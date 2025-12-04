import torch
import os
import config
from data_loader import create_dataloaders
from model import create_model
from train import train_model
from evaluate import evaluate_test
from plotting import plot_training_curves, visualize_sample_predictions

# Train model on climbing dataset and evaluate it on the test split
def main():
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load data
    try:
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR, 
            config.BATCH_SIZE, config.NUM_WORKERS
        )
        print(f"\nClasses: {class_names}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return
    
    # Create a ResNet18-based classifier with a fine-tuned last block
    model = create_model(num_classes=len(class_names), device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {trainable:,} trainable / {total:,} total params\n")
    
    # Train for a fixed number of epochs and keep the best model
    best_val_acc, history = train_model(model, train_loader, val_loader, config.NUM_EPOCHS, device)
    plot_training_curves(history, os.path.join(config.FIGURES_DIR, "training_curves.png"))

    # Load the best checkpoint based on valid accuracy
    print(f"\nLoading the best model")
    try:
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    test_metrics = evaluate_test(model, test_loader, class_names, device)
    
    # Plot predictions from the model
    visualize_sample_predictions(
        model, test_loader, class_names, device,
        num_samples=12,
        save_path=os.path.join(config.FIGURES_DIR, "sample_predictions.png")
    )
    
    # Summary
    summary_path = os.path.join(config.RESULTS_DIR, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Best Val Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Test F1: {test_metrics['macro_f1']:.4f}\n\n")
        for i, cls_name in enumerate(class_names):
            f.write(f"{cls_name}:\n")
            f.write(f"  Precision: {test_metrics['per_class_precision'][i]:.4f}\n")
            f.write(f"  Recall: {test_metrics['per_class_recall'][i]:.4f}\n")
            f.write(f"  F1: {test_metrics['per_class_f1'][i]:.4f}\n")
    
    print(f"\nFinished")


if __name__ == "__main__":
    main()