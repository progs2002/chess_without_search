import chess
from typing import List

import torch

import pandas as pd

empty_token = ['.']

possible_piece_symbols = chess.PIECE_SYMBOLS[1:] + [x.upper() for x in chess.PIECE_SYMBOLS[1:]]

possible_turn_symbols = ['w','b']

possible_digits = [str(x) for x in list(range(0,10))]

possible_letters = ['a','b','c','d','e','f','g','h']

vocab = set(empty_token + possible_digits + possible_piece_symbols + possible_turn_symbols + possible_letters)

vocab_map = {ch: i for i,ch in enumerate(vocab)}

def tokenize(piece_str, turn_str, castling_str, ep_str, hc_str, fc_str) -> List[int]:
    builder: List[int] = []

    for ch in piece_str:
        builder.append(vocab_map[ch])

    builder.append(vocab_map[turn_str])

    for ch in castling_str:
        builder.append(vocab_map[ch])

    for ch in ep_str:
        builder.append(vocab_map[ch])

    for ch in hc_str + fc_str:
        builder.append(vocab_map[ch])

    return torch.tensor(builder)

def tokenize_from_series(s: pd.Series):
    return tokenize(
        s['piece_str'],
        s['turn_str'],
        s['castling_str'],
        s['ep_str'],
        s['hc_str'],
        s['fc_str']
    )

def tokenize_from_struct(fen): 
    return tokenize(*fen.__dict__.values())