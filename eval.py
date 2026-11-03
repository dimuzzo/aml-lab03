import torch

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