import pytest
import chess
import random

from src.fenstruct import FenStruct

#generate boards by random walk
@pytest.fixture
def board() -> chess.Board:
    board = chess.Board()
    num_moves = random.randint(4,40)
    
    move_counter = 0

    while(True):
        legal_moves = list(board.generate_legal_moves())
        if len(legal_moves) == 0 or move_counter > num_moves:
            return board
        
        move = random.choice(legal_moves)
        board.push(move)

        move_counter += 1

@pytest.mark.parametrize("run", list(range(100))) 
def test_fenstruct(board, run):
    fen_struct = FenStruct.from_board(board)

    print(f'run no: {run}, struct= {fen_struct}')
    
    assert len(fen_struct.piece_str) == 64
    assert len(fen_struct.turn_str) == 1
    assert len(fen_struct.castling_str) == 4
    assert len(fen_struct.ep_str) in [1,2]
    assert len(fen_struct.hc_str) == 2
    assert len(fen_struct.fc_str) == 3