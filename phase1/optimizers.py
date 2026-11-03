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

# House prices (in scaled units, where 9.0 = $900,000)
y_true = torch.tensor([
    [9.0],
    [4.5],
    [6.8]
], dtype=torch.float32)

# 2. Define Weights and Bias
W = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)
b = torch.randn(1, dtype=torch.float32, requires_grad=True)

print(f"Initial Weights:\n{W}")
print(f"Initial Bias:\n{b}\n")

# 3. Define Loss Function - Regression
criterion = nn.MSELoss()

optimizer = optim.Adam([W, b], lr=0.05)

print("Starting Training Loop with ADAM Optimizer...\n")

# 4. Training Loop
for epoch in range(200):
    # --- STEP A: FORWARD PASS ---
    y_pred = torch.matmul(X, W) + b

    # --- STEP B: CALCULATE LOSS ---
    loss = criterion(y_pred, y_true)

    # --- STEP C: BACKPROPAGATION & OPTIMIZATION ---
    optimizer.zero_grad()  # Clears old gradients automatically!
    loss.backward()        # Computes gradients via backpropagation
    optimizer.step()       # Updates W and b automatically based on ADAM's math!

print("Training Complete!")

# --- INFERENCE: Predicting on a New House ---
sqft = 5.5
bdrm = 3.0
X_new = torch.tensor([[sqft, bdrm]], dtype=torch.float32)

with torch.no_grad():
    predicted_price = torch.matmul(X_new, W) + b

print(f"--- Prediction Result ---")
print(f"Predicted price for House Features: [{sqft * 1000} sqft, {bdrm} rooms] is ${predicted_price.item() * 100000:,.2f}")

