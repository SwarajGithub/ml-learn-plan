import torch
import torch.nn as nn
import torch.optim as optim

from phase1.mlp import hidden_features

# 1. Grounding and reproducibility
seed = 21
torch.manual_seed(seed)

# Scaled inputs (sqft in thousands, rooms)
X = torch.tensor([
    [7.0, 4.0],
    [2.9, 1.5],
    [5.5, 3.0]
], dtype=torch.float32)

y_true = torch.tensor([
    [9.0],
    [4.5],
    [6.8]
], dtype=torch.float32)


# Define mlp class
class HousePriceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=2, out_features=4)
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_features=4, out_features=1)

    # entry point of neural network training
    def forward(self, x):
        raw_hidden = self.hidden(x)
        raw_relu = self.relu(raw_hidden)
        raw_output = self.output(raw_relu)
        return raw_output, raw_relu

model = HousePriceMLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


print("Training started")

# Quick training loop
for epoch in range(300):
    preds, _ = model(X)
    loss = criterion(preds, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("MLP Trained Successfully")

# Freeze the model
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Extract hidden feature
with torch.no_grad():
    _, hidden_features = model(X)

print(f"Extracted Hidden Features Matrix (3 houses x 4 neurons):\n{hidden_features}")

# A probing classifier on hidden neuron index 1

neuron_index = 1
probe_input = hidden_features[:, neuron_index].unsqueeze(1)

# Train a separate linear weight (w_probe) and bias (b_probe) for this probe
w_probe = torch.randn(1, 1, dtype=torch.float32, requires_grad=True)
b_probe = torch.randn(1, dtype=torch.float32, requires_grad=True)
probe_optimizer = optim.SGD([w_probe, b_probe], lr=0.05)

print(f"--- Probing Neuron #{neuron_index} ---")
for epoch in range(100):
    probe_pred = torch.matmul(probe_input, w_probe) + b_probe
    probe_loss = criterion(probe_pred, y_true)

    probe_optimizer.zero_grad()
    probe_loss.backward()
    probe_optimizer.step()

print(f"Probe Training Complete!")
print(f"True House Prices: {y_true.squeeze().tolist()}")
print(f"Predictions made *only* using Neuron #{neuron_index}: {probe_pred.squeeze().tolist()}")
print(f"Probe Final MSE Loss: {probe_loss.item():.4f}")
