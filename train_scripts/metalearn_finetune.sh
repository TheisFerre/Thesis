#!/bin/sh
#BSUB -J FINETUNE #The name the job will get
#BSUB -q gpuv100 #The queue the job will be committed to, here the GPU enabled queue
#BSUB -gpu "num=1:mode=exclusive_process" #How the job will be run on the VM, here I request 1 GPU with exclusive access i.e. only my c #BSUB -n 1 How many CPU cores my job request
#BSUB -W 24:00 #The maximum runtime my job have note that the queuing might enable shorter jobs earlier due to scheduling.
#BSUB -R "span[hosts=1]" #How many nodes the job requests
#BSUB -R "rusage[mem=12GB]" #How much RAM the job should have access to
#BSUB -R "select[gpu32gb]" #For requesting the extra big GPU w. 32GB of VRAM
#BSUB -o logs/OUTPUT.%J #Log file
#BSUB -e logs/ERROR.%J #Error log file
echo "Starting:"

cd ~/Thesis/metalearning
#cd /Users/theisferre/Documents/SPECIALE/Thesis/src/models

source ~/Thesis/venv-thesis/bin/activate

DATA=/zhome/2b/7/117471/Thesis/data/processed/metalearning/yellow-taxi2020-nov-GRID.pkl
MODEL_PATH=/zhome/2b/7/117471/Thesis/metalearning/METALEARN_MODELS/2021-10-04T15:03:33.015728
TRAIN_SIZE=0.9
BATCH_SIZE=20
EPOCHS=150
WEIGHT_DECAY=0.0000000001
LEARNING_RATE=0.001
LR_PATIENCE=15
LR_FACTOR=0.1
OPTIMIZER=SGD




python /zhome/2b/7/117471/Thesis/src/models/finetune_meta.py --data $DATA --model_path $MODEL_PATH --train_size $TRAIN_SIZE --batch_size $BATCH_SIZE --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --learning_rate $LEARNING_RATE --lr_patience $LR_PATIENCE --lr_factor $LR_FACTOR --optimizer $OPTIMIZER --gpu



# TRAINED MODELS
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T13:45:48.841581
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T13:45:52.884588
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T13:45:57.852115
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T13:46:03.133785
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T13:46:10.448014
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T13:46:18.368208
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T14:06:02.254115
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T14:11:42.156506
# /zhome/2b/7/117471/Thesis/metalearning/2021-10-04T15:03:33.015728


