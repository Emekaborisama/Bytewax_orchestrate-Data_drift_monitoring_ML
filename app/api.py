from typing import Optional
from fastapi import Body, FastAPI,BackgroundTasks,Response,Depends
from app.predict_model import predict_d
from app.db import insert_data,data_from_s3
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse
import json

import pickle
from joblib import dump, load 
import pandas as pd
import numpy as np




from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates








app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)



loaded_model = load("app/model/rdfmodel.joblib.dat")

def predict_d(data):
    res = loaded_model.predict_proba([data])
    result = dict(zip(loaded_model.classes_, [round(x) for x in res[0]]))
    # Retrieve key-value pairs where value > 0.7
    filtered_dict = {str(key): value for key, value in result.items() if value > 0.7}
    return filtered_dict







@app.post("/d_predict")
async def pred(input_value: list):
    result = predict_d(input_value)
    return result
    






@app.get("/monitoring", response_class=HTMLResponse)
async def read_item(request: Request):
    file_path = "app/static/file.html"
    data_from_s3(method="get")
    return HTMLResponse(content=open(file_path, "r").read())
