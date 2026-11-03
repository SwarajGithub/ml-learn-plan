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
        [7.0, 4.0],
        [2.9,1.5],
        [5.5,3.0]
    ], dtype=torch.float32
)

# house prices for each entry
y_true = torch.tensor([
    [9.0],
    [4.5],
    [6.8]
], dtype=torch.float32)

# nothing but random sampling
# size is 2,1 reason X(3,2) * W(2,1) == y_true/y_pred(3,1)
W = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)

# nothing but random start price
b = torch.randn(1, dtype=torch.float32, requires_grad=True)

print(f"Initial Weights:\n{W}")
print(f"Initial Bias:\n{b}\n")

# Learning rate - higher the rate, more impact it will have on learning
learning_rate = 0.01

print("Starting Training Loop...\n")

# Learning
for epoch in range(200):
    # predicted value based on random sampling
    y_pred = torch.matmul(X, W) + b

    # loss based on mean square root
    loss = torch.mean((y_pred - y_true) ** 2)
    #Learning
    # backpropagation
    loss.backward()

    #Learning
    with torch.no_grad():
        # gradient going back
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

        # resetting gradient
        W.grad.zero_()
        b.grad.zero_()

        print(f"epoch -> Loss {loss}")

print("\nTraining Complete!")


# final output
y_pred = torch.matmul(X, W) + b

# final loss
loss = torch.mean((y_pred - y_true) ** 2)

print(f"Predictions:\n{y_pred.detach().numpy()}")
print(f"True Values:\n{y_true.numpy()}")
print(f"Calculated MSE Loss: {loss.item():.4f}")


# Let's predict a new house price after
sqft = 5.5
bdrm = 3.0
X_new = torch.tensor([
    [5.5, 3.0]
], dtype=torch.float32)

# predict based on learning
with torch.no_grad():
    predicted_price = torch.matmul(X_new, W) + b

print(f"\n--- Prediction Result ---")
print(f"Predicted price for new House Features: [({sqft * 1000} sqft), {bdrm} rooms] is ${predicted_price.item() * 100000:,.2f}")

