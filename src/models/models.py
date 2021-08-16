import torch
from torch.functional import einsum
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class CustomTemporalSignal(StaticGraphTemporalSignal):
    def __init__(self, weather_information, time_encoding, *args, **kwargs):
        super(CustomTemporalSignal, self).__init__(*args, **kwargs)
        self.weather_information = weather_information
        self.time_encoding = time_encoding

    def _get_weather(self, index: int):
        if self.weather_information[index] is None:
            return self.weather_information[index]
        else:
            return torch.FloatTensor(self.weather_information[index])

    def _get_time(self, index: int):
        if self.weather_information[index] is None:
            return self.time_encoding[index]
        else:
            return torch.FloatTensor(self.time_encoding[index])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self._get_features(index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        y = self._get_target(index)
        weather = self._get_weather(index)
        time_encoding = self._get_time(index)

        snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        return snapshot, weather, time_encoding


class ExternalLSTM(torch.nn.Module):
    def __init__(
        self, in_features: int, num_nodes: int, out_features: int = 8, hidden_size: int = 64, num_layers: int = 2
    ):
        super(ExternalLSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        self.lstm = torch.nn.LSTM(
            input_size=self.in_features, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True
        )
        self.embedding_hidden = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_nodes * self.out_features
        )
        self.embedding_state = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_nodes * self.out_features
        )

    def forward(self, data: list):
        x = torch.zeros((1, len(data), self.in_features))

        for i, inp in enumerate(data):
            x[0, i, :] = data[i].view(-1)

        _, (hidden_state, cell_state) = self.lstm(x)

        # only take last hidden state from last layer
        hidden_state = hidden_state[-1, :, :]
        cell_state = cell_state[-1, :, :]

        # reshape hidden_state to size (batch, nodes, features)
        embedding_hidden = self.embedding_hidden(hidden_state)
        embedding_output_hidden = embedding_hidden.reshape(-1, self.num_nodes * self.out_features)

        embedding_state = self.embedding_state(cell_state)
        embedding_output_state = embedding_state.reshape(-1, self.num_nodes * self.out_features)

        return embedding_output_state, embedding_output_hidden


class GraphModel(torch.nn.Module):
    def __init__(self, node_in_features: int, num_nodes: int, node_out_features: int = 8, hidden_size: int = 64):
        super(GraphModel, self).__init__()
        self.node_in_features = node_in_features
        self.num_nodes = num_nodes
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size
        self.conv1_sh = GCNConv(node_in_features, node_out_features)
        self.lstm = torch.nn.LSTM(
            input_size=node_out_features * num_nodes,
            hidden_size=self.num_nodes * self.node_out_features,
            batch_first=True,
        )

    def forward(self, data: list):
        lstm_inputs = torch.zeros((1, len(data), self.node_out_features * self.num_nodes))
        for i, snapshot in enumerate(data):
            x, edge_index, edge_weight = snapshot.x, snapshot.edge_index, snapshot.edge_attr
            x = self.conv1_sh(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            lstm_inputs[0, i, :] = x.view(-1)

        _, (hidden_state, cell_state) = self.lstm(lstm_inputs)

        # only take last hidden state from last layer
        hidden_state = hidden_state[-1, :, :]
        cell_state = cell_state[-1, :, :]

        return cell_state, hidden_state


class PredictionLSTM(torch.nn.Module):
    def __init__(self, in_features: int = 8):
        super(PredictionLSTM, self).__init__()


# TODO SCALE MODEL OUTPUTS UP TO FIT DIMENSIONS: (NODES, HIDDEN_HEATURES)
class Encoder(torch.nn.Module):
    def __init__(
        self,
        node_in_features: int,
        num_nodes: int,
        time_features: int,
        weather_features: int,
        node_out_features: int = 8,
        hidden_size: int = 64,
    ):
        super(Encoder, self).__init__()
        self.node_in_features = node_in_features
        self.num_nodes = num_nodes
        self.time_features = time_features
        self.weather_features = weather_features
        self.node_out_features = node_out_features
        self.hidden_size = hidden_size

        self.weight_graph_cell_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )
        self.weight_graph_hidden_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.weight_weather_cell_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )
        self.weight_weather_hidden_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.weight_time_cell_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )
        self.weight_time_hidden_state = torch.nn.Parameter(
            torch.nn.init.normal_(torch.zeros(self.num_nodes * self.node_out_features)), requires_grad=True
        )

        self.graph_model = GraphModel(
            node_in_features=self.node_in_features,
            num_nodes=self.num_nodes,
            node_out_features=self.node_out_features,
            hidden_size=self.hidden_size,
        )

        self.weather_model = ExternalLSTM(
            in_features=self.weather_features,
            num_nodes=self.num_nodes,
            out_features=self.node_out_features,
            hidden_size=self.hidden_size,
        )

        self.time_model = ExternalLSTM(
            in_features=self.time_features,
            num_nodes=self.num_nodes,
            out_features=self.node_out_features,
            hidden_size=self.hidden_size,
        )

    def forward(self, data_graph: list, data_weather: list, data_time: list):

        cell_state_graph, hidden_state_graph = self.graph_model(data_graph)
        cell_state_weather, hidden_state_weather = self.weather_model(data_weather)
        cell_state_time, hidden_state_time = self.time_model(data_time)

        cell_state_fused = (
            torch.einsum("ab, b -> ab", cell_state_graph, self.weight_graph_cell_state)
            + torch.einsum("ab, b -> ab", cell_state_weather, self.weight_weather_cell_state)
            + torch.einsum("ab, b -> ab", cell_state_time, self.weight_time_cell_state)
        )
        hiden_state_fused = (
            torch.einsum("ab, b -> ab", hidden_state_graph, self.weight_graph_hidden_state)
            + torch.einsum("ab, b -> ab", hidden_state_weather, self.weight_weather_hidden_state)
            + torch.einsum("ab, b -> ab", hidden_state_time, self.weight_time_hidden_state)
        )

        return cell_state_fused, hiden_state_fused


# TODO: IMPLEMENT DECODER LSTM
class Decoder(torch.nn.Module):
    def __init__(self, node_out_features, num_nodes):
        super(Decoder, self).__init__()
        self.node_out_features = node_out_features
        self.num_nodes = num_nodes
        self.hidden_size = self.num_nodes * self.node_out_features

        # DESIGN QUESTION HAVE LSTM OUTPUT HIDDEN STATE FOR EACH NODE
        # OR MAKE IT OUTPUT IT FOR THEM ALL TOGETHER AND THEN RESHAPE/SPLIT
        # MANUALLY USING PYTORCH?
        self.lstm = torch.nn.LSTM(
            input_size=self.num_nodes, hidden_size=self.hidden_size, num_layers=1, batch_first=True
        )
        self.linear = torch.nn.Linear(self.hidden_size, self.num_nodes)
    
    def forward(self, x_input, hidden_state, cell_state):
        lstm_out, (hidden, cell) = self.lstm(x_input.unsqueeze(0), (hidden_state, cell_state))
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, (hidden, cell)


class STGNNModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(STGNNModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, graph: list, weather: list, time: list):
        cell_state_fused, hidden_state_fused = self.encoder(graph, weather, time)

        out, (hidden_state, cell_state) = self.decoder(graph[-1].x.reshape(1, 69), hidden_state_fused.unsqueeze(0), cell_state_fused.unsqueeze(0))

        return out, (hidden_state, cell_state)
