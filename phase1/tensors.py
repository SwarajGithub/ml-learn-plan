import torch

seed=21

torch.manual_seed(seed)
X=torch.tensor(
    [
        [1.0, 2.0],
        [2.0,3.0],
        [3.0,1.0]
    ], dtype=torch.float32
)

y_true = torch.tensor([
    [5.0],
    [8.0],
    [7.0]
], dtype=torch.float32)

W = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)
b = torch.randn(1, dtype=torch.float32, requires_grad=True)

print(f"Initial Weights:\n{W}")
print(f"Initial Bias:\n{b}\n")

y_pred = torch.matmul(X, W) + b

loss = torch.mean((y_pred - y_true) ** 2)

print(f"Predictions:\n{y_pred.detach().numpy()}")
print(f"True Values:\n{y_true.numpy()}")
print(f"Calculated MSE Loss: {loss.item():.4f}")

