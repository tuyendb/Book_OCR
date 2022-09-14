from fastapi import FastAPI, Form, UploadFile
from pydantic import BaseModel
from typing import List, Any
app = FastAPI()



@app.get("/")
def read_root(page_list: List[int] = Form(...)):
    print(type(page_list))
    return "a"

# def read_root(pdf_file = UploadFile):
#     return {"Hello": "World"}

