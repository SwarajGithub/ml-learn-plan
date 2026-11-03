import torch

# This is for grounding and reproducibility.
seed=21

torch.manual_seed(seed)

# Input with dimensions
# Below is the number of inputs and dimensions
# 3 inputs and 2 dimensions
# example house prices.
# 2 dimensions - sqft and number_of_rooms
X=torch.tensor(
    [
        [7000.0, 4.0],
        [2900.0,1.5],
        [5500.0,3.0]
    ], dtype=torch.float32
)

# house prices for each entry
y_true = torch.tensor([
    [900000.0],
    [450000.0],
    [680000.0]
], dtype=torch.float32)

# nothing but random sampling
# size is 2,1 reason X(3,2) * W(2,1) == y_true/y_pred(3,1)
W = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)

# nothing but random start price
b = torch.randn(1, dtype=torch.float32, requires_grad=True)

print(f"Initial Weights:\n{W}")
print(f"Initial Bias:\n{b}\n")

# predicted value based on random sampling
y_pred = torch.matmul(X, W) + b

# loss based on mean square root
loss = torch.mean((y_pred - y_true) ** 2)

print(f"Predictions:\n{y_pred.detach().numpy()}")
print(f"True Values:\n{y_true.numpy()}")
print(f"Calculated MSE Loss: {loss.item():.4f}")

