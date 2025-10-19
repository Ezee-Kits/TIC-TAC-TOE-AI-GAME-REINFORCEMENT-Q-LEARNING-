# 🎮 TIC-TAC-TOE-REINFORCEMENT-AI-GAME (Q-LEARNING)

A self-learning **AI-powered Tic Tac Toe game** built using **Q-Learning (Reinforcement Learning)**.  
Instead of being told what the correct move is, this AI learns by **trial and error**, improving over **50,000 self-play episodes**.

---

## 🧠 Overview

This project demonstrates how a simple agent can **learn to play Tic Tac Toe from scratch** using **Reinforcement Learning** — without any prior knowledge of the game.

It explores:
- **State-action learning** (Q-table)
- **Exploration vs. exploitation**
- **Reward-based decision-making**
- **Model persistence using Pickle**

---

## ⚙️ How It Works

### 1️⃣ Training the AI (Q-Learning Phase)

File: `TIC_TAC_TOE_QLEARNING_TRAIN.py`

**Main Idea:**
- The AI plays against a **random opponent** thousands of times.
- It updates its **Q-table** (a memory of good/bad moves) using the **Bellman Equation**:
  
Q(state, action) ← Q(state, action) + α * (reward + γ * max(Q(next_state, all_actions)) - Q(state, action))


- Over time, it discovers the **best strategy to win** or force a draw.

**Rewards:**
- `+1` → AI wins  
- `0` → Draw  
- `-1` → AI loses  

**Parameters:**
| Parameter | Meaning | Value |
|------------|----------|--------|
| α (alpha) | Learning rate | 0.1 |
| γ (gamma) | Discount factor | 0.9 |
| ε (epsilon) | Exploration rate | 1.0 → 0.1 (decays gradually) |
| Episodes | Number of self-play games | 50,000 |

---

### 🧩 Code Highlights

```python
# Q-Learning formula
Q[(state, action)] = Q.get((state, action), 0) + alpha * (
  reward + gamma * max(Q.get((next_state, a), 0) for a in next_moves) - Q.get((state, action), 0)
)


AI learns gradually — starting from random moves to near-perfect play after thousands of rounds.

2️⃣ Saving the Trained Model

After training, the model is saved automatically:

tic_tac_toe_qtable.pkl


This file contains the Q-table, mapping:

(state, action) → learned Q-value

3️⃣ Testing / Playing Against the AI

File: TIC_TAC_TOE_QLEARNING_PLAY.py

This script lets you play against the trained AI interactively in the console.

Gameplay Symbols:

X → AI

O → Player

→ Empty

Board Example:

 X | O | X
---+---+---
 O | X |  
---+---+---
   |   | O

🧮 Reinforcement Learning Logic (Layman’s Explanation)

Think of the AI as a child learning Tic Tac Toe:

It tries random moves at first (exploration).

Every win, loss, or draw gives it feedback (reward).

Over time, it remembers which moves lead to success.

Eventually, it stops guessing and starts making optimal decisions.

Every mistake makes it smarter.
That’s what reinforcement learning is about — learning from experience, not instruction.

🧰 Tech Stack
Component	Technology
Language	Python 3.10+
Algorithm	Q-Learning
Libraries	NumPy, Pickle
Environment	Custom Tic Tac Toe board
Agent Type	Model-Free Reinforcement Learning
📂 Folder Structure
TIC-TAC-TOE-REINFORCEMENT-AI-GAME/
│
├── tic_tac_toe_qtable.pkl                # Saved Q-learning model
├── TIC_TAC_TOE_QLEARNING_TRAIN.py        # Training script
├── TIC_TAC_TOE_QLEARNING_PLAY.py         # Play script
└── README.md                             # Documentation (this file)

⚙️ Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/EzeeKits/TIC-TAC-TOE-REINFORCEMENT-AI-GAME.git
cd TIC-TAC-TOE-REINFORCEMENT-AI-GAME

2️⃣ Install Dependencies
pip install numpy

3️⃣ Train the AI
python TIC_TAC_TOE_QLEARNING_TRAIN.py


After training, you’ll see:

Training completed!
Q-table saved successfully!

4️⃣ Play Against the AI
python TIC_TAC_TOE_QLEARNING_PLAY.py

🧩 Example Output
Training completed!
Q-table saved successfully!

Board:
   |   |  
---+---+---
   | X |  
---+---+---
   |   |  
Enter move (0–8): 0
 O |   |  
---+---+---
   | X |  
---+---+---
   |   |  
AI wins!

🧠 AI Behavior Summary
Situation	AI Behavior
Early game	Explores random moves
Mid training	Learns optimal win/draw strategies
Late training	Consistently avoids losing positions
After convergence	Plays near-perfect Tic Tac Toe
💡 Real-Life Analogy

This project shows how reinforcement learning powers modern AI systems like:

AlphaGo (Google DeepMind) — learns Go by self-play

Self-driving cars — learn through trial and feedback

Robotics — learn movements through rewards and penalties

Your Tic Tac Toe AI is a mini version of that same principle!

🚀 Future Improvements

Add epsilon-greedy visualization (see exploration progress).

Create a GUI (Tkinter or Pygame) for human vs. AI gameplay.

Track win/loss statistics over training epochs.

Upgrade to Deep Q-Learning (DQN) using a Neural Network.

Add AI vs. AI tournaments.

👨‍💻 Author

Ezee Kits
🎓 Electrical & Electronics Engineer | Python & AI Developer
📺 YouTube Channel

📧 Email: ezeekits@gmail.com

💡 Passionate about building intelligent systems that learn through experience.

📜 License

MIT License

MIT License

Copyright (c) 2025 Ezee Kits

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software")...

🌟 Support

If you like this project:

⭐ Star this repo on GitHub

📺 Subscribe to Ezee Kits YouTube Channel

💬 Share it with your friends who love AI & Machine Learning

🧩 Summary
Step	File	Purpose
1️⃣	TIC_TAC_TOE_QLEARNING_TRAIN.py	Trains AI using Q-Learning
2️⃣	tic_tac_toe_qtable.pkl	Stores learned Q-table
3️⃣	TIC_TAC_TOE_QLEARNING_PLAY.py	Lets player test the AI

🎯 Final Output Example:

AI wins! 😎


Your AI has officially learned how to play Tic Tac Toe like a pro!
