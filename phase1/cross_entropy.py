import torch
import torch.nn as nn

# Sentence to finish The cat sat on ...
# Options: mat, apple, moon

# logits na dmodel thinks correct word is mat
logits = torch.tensor([[3.5, 0.2, -0.5]], dtype=torch.float32)

# true word is at index 0
y_true_next_word_index = torch.tensor([0], dtype=torch.long)

criterion = nn.CrossEntropyLoss()

loss = criterion(logits, y_true_next_word_index)

print(f"Model Logits (Scores for [mat, banana, spaceship]): {logits.tolist()}")
print(f"True Next Word Index: {y_true_next_word_index.item()}")
print(f"Calculated Cross Entropy Loss: {loss.item():.4f}")

print("====================================================")
print(f" When the confidence is wrong")
# true word is at index 0
y_true_next_word_index = torch.tensor([2], dtype=torch.long)

criterion = nn.CrossEntropyLoss()

loss = criterion(logits, y_true_next_word_index)

print(f"Model Logits (Scores for [mat, banana, spaceship]): {logits.tolist()}")
print(f"True Next Word Index: {y_true_next_word_index.item()}")
print(f"Calculated Cross Entropy Loss: {loss.item():.4f}")
