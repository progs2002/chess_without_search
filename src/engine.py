import chess
import torch 
import torch.nn.functional as F

from typing import Tuple, List, Dict

from src.fenstruct import FenStruct
from src.tokenizer import tokenize_from_struct

from src.model import Decoder, ModelConfig


model_config = ModelConfig(
    model_dim=256,
    seq_len=77,
    n_layers=6,
    vocab_size=31,
    n_heads=8,
    bias=False,
    dropout=0,
    n_bins=32
)

class Engine:
    def __init__(
        self,
        weights_path: str,
        device = 'cuda'
    ):
        self.device = device
        self.model = Decoder(model_config).to(device)

        self.model.load_state_dict(
            torch.load(weights_path)
        )
        self.model.eval()

    @classmethod
    def get_next_legal_board_states(cls, board: chess.Board) -> Tuple[List[chess.Move], List[chess.Board]]:
        legal_moves = list(board.generate_legal_moves())
        legal_boards = []

        for move in legal_moves:
            t_board = board.copy()
            t_board.push(move)
            legal_boards.append(t_board)

        return legal_moves, legal_boards
    
    @classmethod
    def prepare_batch(cls, boards: List[chess.Board]):
        fens = [FenStruct.from_board(board) for board in boards]
        tensors = [tokenize_from_struct(fen) for fen in fens]
        tensors = torch.stack(
            [t for t in tensors]
        )

        return tensors

    def analyze_board(self, board: chess.Board):
        legal_moves, legal_boards = Engine.get_next_legal_board_states(board)
        batch = Engine.prepare_batch(legal_boards).to(self.device)

        with torch.no_grad():
            out = self.model(batch)[:, -1]
        
        #out_softmaxed = F.softmax(out, -1).to('cpu')
        out_softmaxed = out.to('cpu')
        
        return_obj = [
           (mv.uci(), scores.numpy()) for mv, scores in zip(legal_moves, out_softmaxed)
        ]

        return return_obj

    def analyze_fen(self, fen: str):
        board = chess.Board(fen)
        return self.analyze_board(board)