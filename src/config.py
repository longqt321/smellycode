import os
import torch.nn as nn

# Training config
SEED=1206
TEST_SIZE=0.20
VAL_SIZE=0.20
TRAIN_BATCH_SIZE=1024
VAL_BATCH_SIZE=1024
NUM_WORKERS=4


# Dataset 
LABEL_COLUMNS = ["Long method","God class","Feature envy","Data class"]

# Config mount path on cloud
VOLUME_MOUNT_PATH="/mnt/data"
DATASET_PATH=os.path.join(VOLUME_MOUNT_PATH,"dataset.csv")
ARTIFACTS_DIR=os.path.join(VOLUME_MOUNT_PATH,"artifacts")

# Model config
USE_RELU=True
ACTIVATION=nn.ReLU() if USE_RELU else nn.Mish()