"""
games/tictactoe/game.py

TicTacToe board logic — pure Python, no dependencies.

Board is a flat list of 9 ints:
    0  = empty cell
    1  = X (player 1, the AI agent when training)
   -1  = O (player 2, human or random opponent)

Cells are numbered 0-8 in reading order:
    0 | 1 | 2
   ---+---+---
    3 | 4 | 5
   ---+---+---
    6 | 7 | 8
"""


WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # columns
    [0, 4, 8], [2, 4, 6],              # diagonals
]


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1    # X goes first

    def get_state(self):
        """Return board as a list of 9 floats — the input to the neural network."""
        return [float(c) for c in self.board]

    def get_valid_moves(self):
        """Return indices of empty cells."""
        return [i for i, c in enumerate(self.board) if c == 0]

    def make_move(self, pos):
        """
        Place current_player's mark at pos.

        Returns:
            'ongoing' — game continues, current_player has been switched
            'win'     — current_player just won (NOT yet switched)
            'draw'    — board is full, no winner (NOT yet switched)

        Note: on 'win' and 'draw', current_player is still the player who
        made the last move. The caller uses this to assign rewards correctly.
        """
        assert self.board[pos] == 0, f"Cell {pos} is already taken"
        self.board[pos] = self.current_player

        if self._check_winner(self.current_player):
            return 'win'
        if self._is_draw():
            return 'draw'

        self.current_player *= -1   # switch only if game continues
        return 'ongoing'

    def _check_winner(self, player):
        return any(
            all(self.board[i] == player for i in line)
            for line in WIN_LINES
        )

    def _is_draw(self):
        return all(c != 0 for c in self.board)

    def display(self):
        """Print the board to stdout (for terminal play/debugging)."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in range(3):
            print(' | '.join(symbols[self.board[row * 3 + col]] for col in range(3)))
            if row < 2:
                print('--+---+--')
        print()
