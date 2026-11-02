# ML & SLM Engineering Master Roadmap (8GB VRAM Optimized)

This document outlines a structured, hands-on learning roadmap designed to take you from an **LLM Application Developer** (RAG/Agents) to an **ML/SLM Specialist**. It is specifically optimized for an **8GB NVIDIA GPU** to maximize learning depth while staying safely within hardware constraints.

---

## 1. Environment & Setup Requirements

### Software Prerequisites
1. **Python Environment:** Python 3.10 or 3.11 (3.12 can sometimes have wheel incompatibilities with PyTorch CUDA bindings).
2. **NVIDIA Driver & CUDA Toolkit:** Ensure your GPU drivers support CUDA 12.x (`nvidia-smi` in terminal).
3. **C++ Build Tools:** (For Windows users) Visual Studio Build Tools with C++ workload (required for compiling `bitsandbytes` or Triton kernels if needed).

### Core Python Packages to Install
Create a dedicated virtual environment (conda or venv) and install the core stack:

```bash
# 1. PyTorch with CUDA support (adjust for your specific CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Machine Learning & Data Math Foundations
pip install numpy pandas matplotlib scikit-learn

# 3. Hugging Face Ecosystem & Quantization Stack
pip install transformers datasets accelerate peft bitsandbytes trl evaluate

# 4. Optimized Training Library (Massive VRAM saver for 8GB cards)
pip install unsloth

# 5. Reinforcement Learning Stack
pip install gymnasium stable-baselines3
```

### Models & Downloads
* **Existing Setup (Ollama / Qwen 8B):** Excellent for running inference, generating synthetic datasets, and serving local API benchmarks.
* **Models to Download during Phase 3:**
  * `Qwen/Qwen2.5-1.5B-Instruct` (Primary SLM fine-tuning candidate)
  * `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Alternative fast-iteration model)

---

## 2. Theoretical vs. Hardware Matrix

| Metric | What You CAN Do (Local 8GB VRAM) | What You CANNOT Do (Requires Cloud/Multi-GPU) |
| :--- | :--- | :--- |
| **From Scratch Training** | Train 10M–50M parameter models on custom corpora in minutes. | Pre-train 1B+ parameter base models from scratch. |
| **Fine-Tuning** | 4-bit QLoRA on 1.5B–3B SLMs with 1k–2k context windows. | Full-parameter tuning of >1B models or 70B+ QLoRA fine-tuning. |
| **Reinforcement Learning** | Deep Q-Networks (DQN), PPO on OpenAI Gym, Direct Preference Optimization (DPO) on small SLMs. | Full PPO/RLHF pipelines on 7B+ models running simultaneously with reward models. |

---

## 3. Four-Phase Master Learning Plan

### Phase 1: Deep Learning Foundations from Scratch (Weeks 1–3)

* **Objective:** Understand how neural networks learn, how tensors flow, and how backpropagation updates weights at the math level without black-box wrappers.
* **Why it matters:** Eliminates the "magic box" perception of AI. You will understand loss, learning rates, gradients, and activation functions.
* **Key Topics:**
  * Tensors, autograd, forward/backward passes.
  * Multi-Layer Perceptrons (MLPs), loss functions (MSE, Cross-Entropy), and optimizers (SGD, Adam).
  * Feature extraction, overfitting, regularization.
* **Milestone Project:** Build a multi-layer neural network in raw PyTorch tensors to classify synthetic dataset patterns or handwritten digits without using `torch.nn.Module` abstractions.
* **Skills Gained:** Tensor operations, manual gradient flow debugging, intuition for loss curves.

---

### Phase 2: Building a Micro-Transformer from Scratch (Weeks 4–6)

* **Objective:** Code a generative Transformer (Micro-GPT, ~10M parameters) line-by-line in PyTorch and train it on a local text dataset.
* **Why it matters:** Teaches you the exact mathematical mechanics behind LLMs—Self-Attention, Queries/Keys/Values ($Q, K, V$), Multi-Head Attention, Positional Encodings, and Causal Masking.
* **Key Topics:**
  * Matrix multiplication in attention heads: $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
  * Feed-forward layers, layer normalization, residual connections.
  * Tokenization, vocabulary creation, cross-entropy language modeling loss.
* **Milestone Project:** Train a 10M parameter character-level or sub-word language model on Shakespeare or a code snippet dataset entirely on your local GPU.
* **Skills Gained:** Deep architecture knowledge, understanding token probability sampling (temperature, top-k/top-p).

---

### Phase 3: Fine-Tuning Small Language Models (SLMs) with QLoRA (Weeks 7–10)

* **Objective:** Take a pre-trained 1.5B–3B SLM (`Qwen-2.5-1.5B`) and adapt it to follow custom instruction formats, output specific JSON schemas, or perform specialized agentic tasks using 4-bit QLoRA.
* **Why it matters:** Turns you from a prompt engineer into a model customization specialist who can bake behaviors directly into weights while optimizing VRAM.
* **Key Topics:**
  * **Quantization Mechanics:** FP32 vs. FP16/BF16 vs. INT4 / NF4 (`bitsandbytes`).
  * **LoRA/QLoRA:** Low-Rank Adaptation matrices ($W = W_0 + \Delta W$, where $\Delta W = A \times B$).
  * **Memory Optimization:** Gradient Checkpointing, Gradient Accumulation, Mixed Precision training.
  * **Dataset Engineering:** Formatting instruction datasets (ChatML format), tokenizing with attention masks.
* **Milestone Project:** Fine-tune `Qwen-2.5-1.5B` using Unsloth/PEFT on a 1,000-row dataset of domain-specific JSON function calls or multi-step reasoning traces.
* **Skills Gained:** Parameter-Efficient Fine-Tuning (PEFT), VRAM budget management, customized model evaluation.

---

### Phase 4: Reinforcement Learning & Preference Alignment (Weeks 11–14)

* **Objective:** Learn how agents optimize actions through rewards, and explore modern preference alignment techniques (DPO/RL) used in LLMs.
* **Why it matters:** Agent systems rely on decision-making loops. RL provides the foundation for reward modeling, policy iteration, and model alignment.
* **Key Topics:**
  * **Markov Decision Processes (MDP):** States, Actions, Rewards, Discount factor ($\gamma$).
  * **Deep Q-Learning (DQN):** Q-networks, Experience Replay Buffer, Target networks.
  * **Policy Optimization:** Policy Gradients, PPO basics.
  * **LLM Preference Alignment:** Direct Preference Optimization (DPO) vs. standard Supervised Fine-Tuning (SFT).
* **Milestone Project:** 
  1. Train a DQN agent to solve the Gymnasium `LunarLander-v2` environment.
  2. Run a small Direct Preference Optimization (DPO) script on a 1.5B SLM to prefer concise/helpful responses over verbose ones.
* **Skills Gained:** Reward function design, environment interaction loops, preference alignment algorithms.

---

## 4. Immediate Next Step Checklist

1. Verify Python and CUDA availability:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```
2. Set up your virtual environment and install the initial packages (`torch`, `numpy`, `matplotlib`).
3. Begin Phase 1 by writing your first tensor operations script.