# Climber - Overgripping Classification

## Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries
- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - Computer vision utilities and models
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Metrics and evaluation tools
- `matplotlib>=3.7.0` - Plotting and visualization
- `seaborn>=0.12.0` - Enhanced visualization

### Hardware
- **Recommended**: NVIDIA GPU with CUDA support for faster training
- **Minimum**: CPU (training will be slower)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ClutchKenny/Climber.git
   cd Climber
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The project expects the following directory structure for training, validation, and test datasets:

```
Climber/
├── src/
│   ├── train/
│   │   ├── not_overgripping/
│   │   │   └── [training images]
│   │   └── overgripping/
│   │       └── [training images]
│   ├── valid/
│   │   ├── not_overgripping/
│   │   │   └── [validation images]
│   │   └── overgripping/
│   │       └── [validation images]
│   └── test/
│       ├── not_overgripping/
│       │   └── [test images]
│       └── overgripping/
│           └── [test images]
```

**Note**: Place your climbing images in the appropriate directories before training.

## Configuration

All hyperparameters and settings can be modified in `src/config.py`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `BATCH_SIZE` | 16 | Number of images per batch |
| `NUM_EPOCHS` | 20 | Number of training epochs |
| `HEAD_LR` | 1e-3 | Learning rate for classification head |
| `BACKBONE_LR` | 1e-4 | Learning rate for backbone (layers 3-4) |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization parameter |
| `NUM_WORKERS` | 2 | Number of data loading workers |
| `TRAIN_DIR` | "train" | Path to training data |
| `VAL_DIR` | "valid" | Path to validation data |
| `TEST_DIR` | "test" | Path to test data |
| `BEST_MODEL_PATH` | "best_resnet18_climber.pth" | Checkpoint save path |

## Usage

### Training and Evaluation

To train the model from scratch and evaluate on the test set:

```bash
cd src
python main.py
```

### Individual Scripts

You can also run individual components:

**Training only:**
```bash
cd src
python train.py
```

**Evaluation only (requires trained model):**
```bash
cd src
python evaluate.py
```

**Generate plots:**
```bash
cd src
python plotting.py
```

## Output

After running `main.py`, the following outputs are generated:

### Model Checkpoint
- `src/best_resnet18_climber.pth` - Best model weights based on validation accuracy

### Results Directory (`src/results/`)
- `training_summary.txt` - Text summary of training process
- `figures/training_curves.png` - Loss and accuracy curves over epochs
- `figures/confusion_matrix.png` - Confusion matrix on test set
- `figures/sample_predictions.png` - Sample predictions with ground truth labels

### Console Output
- Training progress (loss and accuracy per epoch)
- Model parameter counts
- Test set metrics:
  - Overall accuracy
  - Precision, Recall, F1-score per class
  - Macro-averaged metrics
  - Confusion matrix
  - Classification report


## Project Structure

```
Climber/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── src/
    ├── main.py                        # Main training and evaluation script
    ├── config.py                      # Configuration and hyperparameters
    ├── data_loader.py                 # Dataset loading and preprocessing
    ├── model.py                       # Model architecture definition
    ├── train.py                       # Training loop
    ├── evaluate.py                    # Test evaluation
    ├── plotting.py                    # Visualization utilities
    ├── best_resnet18_climber.pth      # Trained model checkpoint
    ├── train/                         # Training images
    ├── valid/                         # Validation images
    ├── test/                          # Test images
    └── results/                       # Output directory
        ├── training_summary.txt
        └── figures/
            ├── training_curves.png
            ├── confusion_matrix.png
            └── sample_predictions.png
```

## Troubleshooting

**CUDA out of memory error:**
- Reduce `BATCH_SIZE` in `config.py`
- Use a smaller model or reduce image resolution

**Low accuracy:**
- Ensure balanced class distribution in your dataset
- Increase `NUM_EPOCHS` for longer training
- Adjust learning rates in `config.py`
- Check that images are properly labeled in correct directories

**Data loading errors:**
- Verify dataset directory structure matches expected format
- Ensure image files are valid and readable
- Check file permissions


## Author

Kenny Tran
Matthew Park

---

**Last Updated**: December 4, 2025
