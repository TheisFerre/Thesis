# -*- coding: utf-8 -*-
from src.data.process_dataset import (
    load_csv_dataset,
    create_grid,
    create_grid_ids,
    correlation_adjacency_matrix,
    features_targets_and_externals,
    Dataset,
)
from src.data.encode_externals import Weather_container, time_encoder
import json
import os
import dill
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading .csv files from directory {input_filepath}")
    logger.info("making final data set from raw data")
    outfile = open(output_filepath + "/output.txt", "w")

    for infile in os.listdir(input_filepath):
        if infile[-3:] == "csv":
            logger.info(f"Processing file {input_filepath}/{infile}")
            infile_root = infile[:-4]
            column_dict = json.load(open(f"{input_filepath}/{infile_root}.json"))

            # we dont neccesarily need stations if we makes grids.
            if "STATION_COL" in column_dict:
                station_column = column_dict["STATION_COL"]
            else:
                station_column = None

            df = load_csv_dataset(
                path=f"{input_filepath}/{infile}",
                time_column=column_dict["TIME_COL"],
                location_columns=[column_dict["LNG_COL"], column_dict["LAT_COL"]],
                station_column=station_column,
            )

            df = create_grid(df, lng_col=column_dict["LNG_COL"], lat_col=column_dict["LAT_COL"], splits=10)

            df["grid_id"] = create_grid_ids(
                df, longitude_col=column_dict["LNG_COL"] + "_binned", lattitude_col=column_dict["LAT_COL"] + "_binned"
            )

            region_ordering = df["grid_id"].unique()
            adj_mat = correlation_adjacency_matrix(
                rides_df=df, region_ordering=region_ordering, id_col="grid_id", time_col=column_dict["TIME_COL"]
            )

            # encode time & weather
            mean_lon = df[column_dict["LNG_COL"]].mean()
            mean_lat = df[column_dict["LAT_COL"]].mean()
            weather = Weather_container(longitude=mean_lon, latitude=mean_lat)

            time_enc = time_encoder()

            X, targets, time_encoding, weather_array = features_targets_and_externals(
                df=df, region_ordering=region_ordering, id_col="grid_id", time_col=column_dict["TIME_COL"], time_encoder=time_enc, weather=weather
            )

            
            """weather_array = weather.get_weather_df(
                start=min(df[column_dict["TIME_COL"]]), end=max(df[column_dict["TIME_COL"]])
            )

            time_encoding = encode_times(datetime_series=df[column_dict["TIME_COL"]])"""

            dat = Dataset(
                adjacency_matrix=adj_mat,
                targets=targets,
                X=X,
                weather_information=weather_array,
                time_encoding=time_encoding,
            )

            logger.info(f"SAVING PROCESSED DATA TO {output_filepath}/{infile_root}.pkl")
            outfile = open(f"{output_filepath}/{infile_root}.pkl", "wb")
            dill.dump(dat, outfile)

            outfile.close()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
