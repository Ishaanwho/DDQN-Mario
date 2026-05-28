# 🍄 DDQN Mario Agent

A Deep Reinforcement Learning agent that learns to play **Super Mario Bros** using **Double Deep Q-Networks (DDQN)**, built with **PyTorch**, **OpenAI Gym**, and `gym-super-mario-bros`.

<p align="center">
  <img src="https://media.giphy.com/media/l1KtXmfi3EnjM5zpK/giphy.gif" width="500"/>
</p>

---

## 🚀 Live Demo

🎮 Watch the Trained Agent Play
👉 [DDQN Mario Hugging Face Demo](https://huggingface.co/spaces/isxhxan/DDQN-Mario?utm_source=chatgpt.com)

---

# ✨ Features

* 🧠 **Double DQN (DDQN)** implementation
* 🎮 Real-time Mario gameplay rendering
* 📦 Replay Memory Buffer
* 🎯 Target Network updates
* 🖼️ Frame preprocessing + stacking
* ⚡ CUDA / GPU acceleration support
* 💾 Automatic checkpoint saving/loading
* 📉 Epsilon-greedy exploration
* 🔥 DeepMind-style CNN architecture

---

# 🧠 Model Architecture

The agent uses a convolutional neural network inspired by the original DeepMind Atari DQN paper.

```text
Input (4 stacked grayscale frames: 84x84)

→ Conv2D (32 filters, 8x8 kernel, stride 4)
→ ReLU

→ Conv2D (64 filters, 4x4 kernel, stride 2)
→ ReLU

→ Conv2D (64 filters, 3x3 kernel, stride 1)
→ ReLU

→ Fully Connected (512)
→ ReLU

→ Output Layer (Q-values for actions)
```

---

# 🎯 Why DDQN Instead of DQN?

Traditional DQN tends to **overestimate Q-values**.

This project uses **Double DQN**, where:

* `policy_net` selects the best action
* `target_net` evaluates that action

Which stabilizes training and improves learning performance.

Mathematically:

```math
Q_target(s', argmax_a Q_policy(s', a))
```

instead of:

```math
max_a Q_target(s', a)
```

---

# 🕹️ Action Space

Using:

```python
gym_super_mario_bros.actions.RIGHT_ONLY
```

The agent learns from a simplified movement set including:

* Move Right
* Right + Jump
* Right + Run
* Jump
* Idle

---

# 🛠️ Tech Stack

* Python
* PyTorch
* OpenAI Gym
* gym-super-mario-bros
* NumPy
* CUDA

---

# 📂 Project Structure

```bash
DDQN-Mario/
│
├── checkpoints/
│   └── mario_dqn_latest.pt
│
├── wrappers.py
├── mario_GPU.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/Ishaanwho/DDQN-Mario.git

cd DDQN-Mario
```

---

## 2️⃣ Create a virtual environment (recommended)

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Training the Agent

Run:

```bash
python mario_GPU.py
```

If CUDA is available, training automatically uses the GPU.

---



# 💾 Checkpoint System

The project automatically:

* saves model checkpoints
* resumes training from latest checkpoint
* stores optimizer state
* preserves epsilon value

Checkpoint location:

```bash
checkpoints/mario_dqn_latest.pt
```

---

# 📸 Preprocessing Pipeline

The environment uses several wrappers:

| Wrapper            | Purpose                  |
| ------------------ | ------------------------ |
| `MaxAndSkipEnv`    | Frame skipping           |
| `ProcessFrame84`   | Resize + grayscale       |
| `ImageToPyTorch`   | Channel-first conversion |
| `BufferWrapper`    | Frame stacking           |
| `ScaledFloatFrame` | Normalization            |

---

# 📊 Hyperparameters

| Parameter     | Value            |
| ------------- | ---------------- |
| Episodes      | 500              |
| Batch Size    | 64               |
| Replay Memory | 10,000           |
| Gamma         | 0.99             |
| Learning Rate | 1e-4             |
| Epsilon Min   | 0.1              |
| Target Update | Every 1000 steps |

---

# 🖥️ CUDA Support

The script automatically detects GPU availability:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---


# 📈 Future Improvements

* ✅ Prioritized Experience Replay
* ✅ Dueling DDQN
* ✅ Noisy Networks
* ✅ Rainbow DQN
* ✅ Curriculum Learning
* ✅ Multi-world training
* ✅ TensorBoard metrics
* ✅ Distributed training

---

# 🤝 Contributing

Pull requests are welcome.

If you'd like to improve the agent, optimize training speed, or add new RL techniques, feel free to fork the repo and open a PR.

---

# 👨‍💻 Author

Built by **Ishaan Singh**

* GitHub: [Ishaanwho GitHub](https://github.com/Ishaanwho)
* Hugging Face: [isxhxan Hugging Face](https://huggingface.co/isxhxan)

---

# ⭐ Star the Repo

If you liked this project, consider giving it a star ⭐ :D
