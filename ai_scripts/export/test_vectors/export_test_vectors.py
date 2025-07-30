import os
import sys
import torch
import numpy as np
from torchvision import datasets, transforms

# Add parent directory to sys.path so model.le_net can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.le_net import LeNet

# Paths
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../export/test_vectors'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../export/weights/lenet_weights.pth'))

os.makedirs(EXPORT_DIR, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
sample_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Export the first N test vectors
N = 10
with torch.no_grad():
    for i, (inputs, targets) in enumerate(sample_loader):
        if i >= N:
            break
        inputs = inputs.to(device)
        outputs = model(inputs)
        np.save(os.path.join(EXPORT_DIR, f'input_{i}.npy'), inputs.cpu().numpy())
        np.save(os.path.join(EXPORT_DIR, f'output_{i}.npy'), outputs.cpu().numpy())
        np.save(os.path.join(EXPORT_DIR, f'label_{i}.npy'), targets.cpu().numpy())
