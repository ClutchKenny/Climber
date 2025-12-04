import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config

# Run training epoch over entire training set
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward -> loss -> backward -> optimizer step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(train_loader), 100 * correct / total

# Evaluate the model on the validation set
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(val_loader), 100 * correct / total

# Sets up loss function and optimizer, and trains for num_epochs
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()

    # Separate parameter groups for different learning rates for the backbone and classifier head.
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Layers 3 and 4
        if name.startswith("layer3.") or name.startswith("layer4."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": config.BACKBONE_LR},
            {"params": head_params, "lr": config.HEAD_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    # Early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    epochs_since_improvement = 0
    patience = 4
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} loss, {train_acc:.2f}% acc | "
              f"Val: {val_loss:.4f} loss, {val_acc:.2f}% acc")
        
        scheduler.step(val_acc)

        # Save the best model weights seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  -> Saved best model ({val_acc:.2f}%)")
        else:
            epochs_since_improvement +=1
            print(f"No improvement for {epochs_since_improvement} epoch(s)")
            if epochs_since_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc, history