import torch
import torch.nn as nn
import torch.optim as optim

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training MLP Model")

for epoch in range(300):

    # passes through forward method
    y_pred, _ = model(X)

    # this is MSE loss
    loss = criterion(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete")

X_new = torch.tensor([[5.5, 3.0]], dtype=torch.float32)

# use the trained model - in evaluation mode now
model.eval()
with torch.no_grad():
    predicted_price, hidden_features = model(X_new)

print(f"\n--- MLP Prediction Result ---")
print(f"Hidden features: {hidden_features}")
print(f"Predicted price for House Features: [5500 sqft, 3 rooms] is ${predicted_price.item() * 100000:,.2f}")
