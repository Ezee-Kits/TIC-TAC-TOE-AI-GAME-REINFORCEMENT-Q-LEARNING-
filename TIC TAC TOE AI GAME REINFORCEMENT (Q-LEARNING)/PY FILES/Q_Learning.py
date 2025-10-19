import numpy as np
import random
import pickle

# Tic Tac Toe Environment
WIN_COMBOS = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6]
]

def check_winner(board):
    for combo in WIN_COMBOS:
        line = [board[i] for i in combo]
        if line == [-1,-1,-1]: return -1  # AI wins
        if line == [1,1,1]: return 1      # Player wins
    if 0 not in board:
        return 0  # draw
    return None  # ongoing

def available_moves(board):
    return [i for i,v in enumerate(board) if v == 0]

def get_state(board):
    return tuple(board)  # immutable for dictionary key

# Q-learning params
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 1.0    # exploration rate
min_epsilon = 0.1
decay = 0.9995

Q = {}  # Q-table

def choose_action(state, moves):
    if random.uniform(0,1) < epsilon:
        return random.choice(moves)
    q_values = [Q.get((state, a), 0) for a in moves]
    return moves[int(np.argmax(q_values))]

# Training
episodes = 50000

for ep in range(episodes):
    board = [0]*9
    done = False

    while not done:
        state = get_state(board)
        moves = available_moves(board)
        if not moves:
            break
        
        action = choose_action(state, moves)
        board[action] = -1  # AI move

        result = check_winner(board)
        if result is not None:
            reward = 1 if result == -1 else -1 if result == 1 else 0
            Q[(state, action)] = Q.get((state, action), 0) + alpha * (reward - Q.get((state, action), 0))
            done = True
            break

        # Opponent random move
        opp_moves = available_moves(board)
        if not opp_moves:
            break
        opp_action = random.choice(opp_moves)
        board[opp_action] = 1

        result = check_winner(board)
        if result is not None:
            reward = 1 if result == -1 else -1 if result == 1 else 0
            Q[(state, action)] = Q.get((state, action), 0) + alpha * (reward - Q.get((state, action), 0))
            done = True
        else:
            next_state = get_state(board)
            next_moves = available_moves(board)
            future = max([Q.get((next_state, a), 0) for a in next_moves]) if next_moves else 0
            Q[(state, action)] = Q.get((state, action), 0) + alpha * (0 + gamma * future - Q.get((state, action), 0))
    
    # decay epsilon
    epsilon = max(min_epsilon, epsilon * decay)

print("Training completed!")

# Save the learned Q-table
with open("tic_tac_toe_qtable.pkl", "wb") as f:
    pickle.dump(Q, f)
print("Q-table saved successfully!")
