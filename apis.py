from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn

from predict import get_model_rec, get_random_rec, load_model

class Input(BaseModel) : 
    years : List[int]
    key_top_k: int
    key_input_len: int
    status : bool
    key_years: List[int]
    added_movie_ids : List[int]
    input_len : int
    top_k : int
    selected_movie_count : int
    clicked : bool
    
app = FastAPI()

@app.get("/recommend-movies/")
def message():
    return {"message":"hello world"}
    
@app.post("/random-movies/")
def random_movies(session_state:Input):
    return session_state.top_k

@app.post("/model-movies/")
def model_movies(session_state:Input):
    return session_state

if __name__ == "__main__" : 
    uvicorn.run(app, host = "0.0.0.0", port=8100)