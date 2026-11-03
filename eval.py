import os
import torch
import torch.nn as nn
from dataset.dataloader import get_dataloaders
from models.customnet import CustomNet

def evaluate(model, val_loader, criterion, device):
    """Evaluates the model over the validation set."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass: compute predictions
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, targets)

            # Accumulate loss
            val_loss += loss.item()
            # Compute accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    print(f'Validation -> Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running evaluation on device: {device}")

    # Load the validation loader
    _, val_loader = get_dataloaders(batch_size=32, num_workers=2)

    # Initialize the model and loss function
    model = CustomNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print("Warning: checkpoint not found. Evaluating with untrained weights.")

    # Run the evaluation
    evaluate(model, val_loader, criterion, device)


if __name__ == '__main__':
    main()