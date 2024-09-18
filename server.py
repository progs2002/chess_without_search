from src.engine import Engine
from pydantic import BaseModel
from typing import List

from fastapi import FastAPI

weights = "weights/compare_runs/model_decoder.pt"
engine = Engine(weights)

class FenStr(BaseModel):
    fen: str

class MoveScore(BaseModel):
    move: str
    scores: List[float]

app = FastAPI()

@app.post("/analyze/")
def analyze(fen_str: FenStr):
    res_obj = engine.analyze_fen(fen_str.fen)
    print(res_obj)
    return res_obj
