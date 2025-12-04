import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import config
from plotting import plot_confusion_matrix

# Run the model on test set and output results
def evaluate_test(model, test_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluating on test")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1
    }
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    for i, cls_name in enumerate(class_names):
        print(f"{cls_name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
    
    print("\n" + classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    plot_confusion_matrix(cm, class_names, os.path.join(config.FIGURES_DIR, "confusion_matrix.png"))
    
    return metrics