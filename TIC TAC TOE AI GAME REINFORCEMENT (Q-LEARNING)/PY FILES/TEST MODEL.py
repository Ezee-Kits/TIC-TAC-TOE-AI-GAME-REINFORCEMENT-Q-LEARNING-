
import numpy as np
import joblib




class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        # 0 = empty, 1 = AI, -1 = Player
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # AI starts by default
        return self.board.copy()

    def available_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def make_move(self, position, player):
        if self.board[position] == 0:
            self.board[position] = player
            return True
        return False

    def check_winner(self):
        combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for combo in combos:
            total = sum(self.board[i] for i in combo)
            if total == 3:
                return 1   # AI wins
            elif total == -3:
                return -1  # Player wins
        if 0 not in self.board:
            return 0  # Draw
        return None  # No winner yet

    def display(self):
        symbols = {1: "X", -1: "O", 0: " "}
        print("\nBoard:")
        for i in range(0, 9, 3):
            print(f" {symbols[self.board[i]]} | {symbols[self.board[i+1]]} | {symbols[self.board[i+2]]} ")
            if i < 6:
                print("---+---+---")
        print()

    def ai_move(self, model):
        available = self.available_moves()
        if not available:
            return

        # Convert current board to tuple for Q-table lookup
        state = tuple(self.board.tolist())

        if state in model:
            q_values = model[state]
        else:
            q_values = np.zeros(9)  # unseen state → all zeros

        # Sort moves by Q-value descending
        ranked_moves = sorted(
            zip(range(9), q_values), key=lambda x: x[1], reverse=True
        )

        for move, _ in ranked_moves:
            if move in available:
                self.make_move(move, 1)
                break


# Example usage
if __name__ == "__main__":
    from random import choice

    game = TicTacToe()
    game.display()

    # Dummy model that picks random moves for now
    class DummyModel:
        def predict(self, X):
            return np.random.rand(X.shape[0], 9)

    
    model = joblib.load("DATASET/tic_tac_toe_qtable.pkl")


    while True:
        # AI Move
        game.ai_move(model)
        game.display()
        winner = game.check_winner()
        if winner is not None:
            print("AI wins!" if winner == 1 else "Player wins!" if winner == -1 else "Draw!")
            break

        # Player Move
        move = int(input("Enter move (0–8): "))
        if game.make_move(move, -1):
            game.display()
        else:
            print("Invalid move! Try again.")
            continue

        winner = game.check_winner()
        if winner is not None:
            print("AI wins!" if winner == 1 else "Player wins!" if winner == -1 else "Draw!")
            break
