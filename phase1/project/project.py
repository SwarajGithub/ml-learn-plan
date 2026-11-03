import torch

# 1. Grounding and Reproducibility
seed = 21
torch.manual_seed(seed)

def generate_housing_data(samples: int = 200):
    sqft = torch.rand(samples, 1) * 4.0 + 1.0
    rooms = torch.randint(1, 7, (samples, 1)).float()
    age = torch.rand(samples, 1) * 0.5
    distance = torch.rand(samples, 1) * 1.45 + 0.05
    garage = torch.randint(0, 4, (samples, 1)).float()

    X = torch.cat([sqft, rooms, age, distance, garage], dim=1)

    # Price ($ in hundreds of thousands) = 1.2*sqft + 0.5*rooms - 0.3*age - 1.5*distance + 0.4*garage + Base(2.0) + Noise
    true_weights = torch.tensor([[1.2], [0.5], [-0.3], [-1.5], [0.4]])
    base_price = 2.0

    noise = torch.randn(samples, 1) * 0.25  # Random market noise

    y = torch.matmul(X, true_weights) + base_price + noise

    return X, y

X, y = generate_housing_data(samples=200)

X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X_normalized = (X - X_mean)/ X_std

# Layer 1: Hidden Layer (5 inputs -> 8 hidden neurons)
W1 = (torch.randn(5, 8, dtype=torch.float32) * 0.1).requires_grad_(True)
b1 = torch.zeros(1, 8, dtype=torch.float32, requires_grad=True)

# Layer 2: Output Layer (8 hidden neurons -> 1 price output)
W2 = (torch.randn(8, 1, dtype=torch.float32) * 0.1).requires_grad_(True)
b2 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)

learning_rate = 0.01


def manual_relu(tensor):
    return torch.clamp(tensor, min=0.0)


print("Training")

# 3. Training Loop
for epoch in range(500):
    h_linear = torch.matmul(X_normalized, W1) + b1

    h_activated = manual_relu(h_linear)

    y_pred = torch.matmul(h_activated, W2) + b2

    loss = torch.mean((y_pred - y) ** 2)

    loss.backward()

    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad

        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()


print("Regression Training Complete!")

# Test the prediction on a sample house
sample_house = torch.tensor([[3.0, 3.0, 0.1, 0.04, 2.0]], dtype=torch.float32)
sample_normalized = (sample_house - X_mean) / X_std

with torch.no_grad():
    h_sample = manual_relu(torch.matmul(sample_normalized, W1) + b1)
    pred_price = torch.matmul(h_sample, W2) + b2

for i in range(100):
    features = X[i].tolist()
    price = y[i].item() * 100000  # Scaling back to actual dollar amounts
    print(f"House {i+1} -> Features [Sqft, Rooms, Age, Dist, Garage]: {[round(f, 2) for f in features]} | Price: ${price:,.2f}")


print(f"Predicted House Price with Features [Sqft, Rooms, Age, Dist, Garage]: {[round(f, 2) for f in sample_house.tolist()[0]]}: ${pred_price.item() * 100000:,.2f}")