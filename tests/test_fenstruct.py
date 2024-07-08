import pytest

from src.fenstruct import FenStruct

@pytest.mark.parametrize("run", list(range(100))) 
def test_fenstruct(board, run):
    fen = FenStruct.from_board(board)

    print(f'run no: {run}, struct= {fen}')
    
    assert len(fen.piece_str) == 64
    assert len(fen.turn_str) == 1
    assert len(fen.castling_str) == 4
    assert len(fen.ep_str) in [1,2]
    assert len(fen.hc_str) == 2
    assert len(fen.fc_str) == 3