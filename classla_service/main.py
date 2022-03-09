from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel
from CLASSLA.mark_entities import mark_entities_in_text


class Vertex(BaseModel):
    kbID: str
    tokenpositions: List[int]


class Edge(BaseModel):
    kbID: str
    left: List[int]
    right: List[int]
    sentence: str
    entity1_text: str
    entity2_text: str
    entity1_sentence_position: int
    entity2_sentence_position: int


class MarkedSentence(BaseModel):
    vertexSet: List[Vertex]
    edgeSet: List[Edge]
    tokens: List[str]

app = FastAPI()


@app.get("/mark_entities/{text}", response_model=List[MarkedSentence])
async def root(text: str, call_id: str):
    return mark_entities_in_text(text, call_id)
