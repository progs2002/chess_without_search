import chess
import torch 

from typing import List, Dict

from src.fenstruct import FenStruct
from src.tokenizer import tokenize_from_struct

#the model
from src.gpt import GPT, GPTConfig

n_bin = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = GPTConfig(
    block_size=70,
    vocab_size=42,
    n_layer=6,
    n_head=8,
    n_embd=256,
    n_bin=n_bin,
    bias=False
)

model = GPT(config).to(device)

model.load_state_dict(
    torch.load(
        'weights/new_model/new_ds_new_model_lrdecay_50000.pt'
    )
)

model.eval()

def get_next_legal_board_states(board: chess.Board) -> List[chess.Board]:
    legal_moves = list(board.generate_legal_moves())
    legal_boards = []

    for move in legal_moves:
        t_board = board.copy()
        t_board.push(move)
        legal_boards.append(t_board)

    return legal_moves, legal_boards

def prepare_batch(boards):
    fens = [FenStruct.from_board(board) for board in boards]
    tensors = [tokenize_from_struct(fen) for fen in fens]
    tensors = torch.stack(
        [t for t in tensors]
    )

    return tensors

def get_bins(boards: List[chess.Board]) -> List[float]:
    batch = prepare_batch(boards).to(device)
    with torch.no_grad():
        out = model(batch)[:,-1].max(-1)[1]

    return out

bin_scores = (torch.arange(0,n_bin) + torch.arange(1,n_bin+1))/(n_bin*2)
bin_scores = bin_scores.to(device)

def get_moves(board: chess.Board) -> Dict[chess.Move, float]:
    moves, boards = get_next_legal_board_states(board)
    bins = get_bins(boards)

    assert len(bins) == len(moves)

    scores = bin_scores[bins]

    return moves, scores.cpu().numpy()