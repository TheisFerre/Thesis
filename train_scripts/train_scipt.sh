#!/bin/sh
#BSUB -J FIRST_RUN The name the job will get
#BSUB -q gpuv100 The queue the job will be committed to, here the GPU enabled queue
#BSUB -gpu "num=1:mode=exclusive_process" How the job will be run on the VM, here I request 1 GPU with exclusive access i.e. only my c #BSUB -n 1 How many CPU cores my job request
#BSUB -W 24:00 The maximum runtime my job have note that the queuing might enable shorter jobs earlier due to scheduling.
#BSUB -R "span[hosts=1]" How many nodes the job requests
#BSUB -R "rusage[mem=40GB]" How much RAM the job should have access to
#BSUB -R "select[gpu32gb]" For requesting the extra big GPU w. 32GB of VRAM
#BSUB -N Send an email when done
#BSUB -o logs/%J_Output_nyc_combi.out Log file
#BSUB -e logs/%J_Error_nyc_combi.err Error log file
echo "Starting:"

cd ~/Thesis/models
#cd /Users/theisferre/Documents/SPECIALE/Thesis/src/models

source ~/Thesis/venv-thesis/bin/activate

DATA=../../data/processed/202106-citibike-tripdata.pkl
MODEL=edgeconv
NUM_HISTORY=12
TRAIN_SIZE=0.8
BATCH_SIZE=32
EPOCHS=5
WEIGHT_DECAY=0
LEARNING_RATE=0.001
LR_FACTOR=1
LR_PATIENCE=100



python train_model.py --data $DATA --model $MODEL --num_history $NUM_HISTORY --train_size $TRAIN_SIZE \
--batch_size $BATCH_SIZE --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --learning_rate $LEARNING_RATE \
--lr_factor $LR_FACTOR --lr_patience $LR_PATIENCE 

