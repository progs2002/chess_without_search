from fenstruct import FenStruct
import chess
from typing import List

import torch

possible_ep_strings = ['-'] + [
    chess.square_name(s) for s in list(range(16,24)) + list(range(40,48)) 
]

empty_token = ['.']

possible_piece_symbols = chess.PIECE_SYMBOLS[1:] + [x.upper() for x in chess.PIECE_SYMBOLS[1:]]

possible_turn_symbols = ['w','b']

possible_digits = [str(x) for x in list(range(0,10))]

vocab = empty_token + possible_digits + possible_piece_symbols + possible_turn_symbols + possible_ep_strings

vocab_map = {ch: i for i,ch in enumerate(vocab)}

def tokenize(fen) -> List[int]:
    builder: List[int] = []

    for ch in fen['piece_str']:
        builder.append(vocab_map[ch])

    builder.append(vocab_map[fen['turn_str']])

    for ch in fen['castling_str']:
        builder.append(vocab_map[ch])

    builder.append(vocab_map[fen['ep_str']])

    for ch in (fen['hc_str'] + fen['fc_str']):
        builder.append(vocab_map[ch])

    return torch.tensor(builder)