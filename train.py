import os
import torch
import torch.nn as nn
from dataset.dataloader import get_dataloaders
from models.customnet import CustomNet
from eval import evaluate

def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to the selected device
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
         # Forward pass: compute predictions
        outputs = model(inputs)
        # Compute the loss
        loss = criterion(outputs, targets)
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    print(f'Train Epoch: {epoch} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%')
    return train_loss, train_acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Ensure checkpoint folder exists
    os.makedirs('checkpoints', exist_ok=True)

    # Load datasets
    train_loader, val_loader = get_dataloaders(batch_size=32, num_workers=2)

    # Initialize model, loss and optimizer
    model = CustomNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0.0
    num_epochs = 10

    # Training and evaluation loop
    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch, model, train_loader, criterion, optimizer, device)
        _, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"--> Saved best model checkpoint with accuracy: {best_acc:.2f}%")

    print(f'Training finished. Best validation accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()