from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from RECON.predict_relation_RECON import predict
from uuid import uuid4

app = FastAPI()

class Entity(BaseModel):
    text: str
    sentence_position: int
    
class Relation(BaseModel):
    WikiData_tag: str
    description: str

class MarkedRelation(BaseModel):
    sentence: str
    entity1: Entity
    entity2: Entity
    relation: Relation
    score: float

@app.get("/find_relations/{text}", response_model=List[MarkedRelation])
async def root(text: str):
    call_id = uuid4() # used to differentiate different calls in logs
    return predict(text, call_id)
    
    
@app.get("/find_relations", response_model=List[MarkedRelation])
async def find_relations2(text: str= "France je oče od Anžeta."):
    call_id = uuid4() # used to differentiate different calls in logs
    return predict(text, call_id)
