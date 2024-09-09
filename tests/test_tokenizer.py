import pytest

from src.fenstruct import FenStruct
from src.tokenizer import tokenize

@pytest.mark.parametrize("run", list(range(100))) 
def test_tokenizer(board, run):
    fen = FenStruct.from_board(board)
    tokens = tokenize(*fen.__dict__.values())

    print(f'run no: {run}, tokens= {tokens}')

    assert len(tokens) == 77
