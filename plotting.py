import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import config


def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_sample_predictions(model, test_loader, class_names, device, num_samples=12, save_path=None):
    model.eval()
    
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    mean = np.array(config.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(config.IMAGENET_STD).reshape(3, 1, 1)
    
    for idx, ax in enumerate(axes):
        if idx >= num_samples:
            ax.axis('off')
            continue
            
        img_idx = indices[idx]
        img = images[img_idx].cpu().numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        
        true_label = labels[img_idx].item()
        pred_label = preds[img_idx].item()
        confidence = probs[img_idx][pred_label].item()
        
        ax.imshow(img)
        ax.axis('off')
        
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nConf: {confidence:.2f}"
        ax.set_title(title, fontsize=9, color=color)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(config.FIGURES_DIR, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dataset_distribution(stats, save_path):
    splits = list(stats.keys())
    class_names = list(stats[splits[0]].keys())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, split in enumerate(splits):
        counts = [stats[split].get(cls, 0) for cls in class_names]
        ax.bar(x + i * width, counts, width, label=split.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Distribution')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()