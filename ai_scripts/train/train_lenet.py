import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.le_net import LeNet

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Adjust paths if needed
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
EXPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ai_scripts/export/weights'))
os.makedirs(EXPORT_PATH, exist_ok=True)

train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model setup
model = LeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

# Save the trained model weights
torch.save(model.state_dict(), os.path.join(EXPORT_PATH, 'lenet_weights.pth'))

# Evaluate and print test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
print('Test accuracy: {:.2f}%'.format(100. * correct / total))
