import os

# Training settings
BATCH_SIZE = 16
NUM_EPOCHS = 20
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# Data paths
TRAIN_DIR = "train"
VAL_DIR = "valid"
TEST_DIR = "test"

# Model
BEST_MODEL_PATH = "best_resnet18_climber.pth"

# Output
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]