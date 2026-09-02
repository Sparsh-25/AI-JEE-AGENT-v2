from fastapi import FastAPI
from pydantic import BaseModel
from integration import response
import time


class Query(BaseModel):
    query : str

app = FastAPI()

@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/chat")
def ask(query: Query):
    start = time.perf_counter()
    answer = response(query.query)
    elapsed = time.perf_counter()-start
    print(f"[TOTAL Chat] elapsed in {elapsed}")
    return {"answer" : answer}

