from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
from automl import run_automl_pipeline

app = FastAPI()

@app.post("/train")
async def train_model(file: UploadFile, target: str = Form(...)):
    df = pd.read_csv(file.file)
    result = run_automl_pipeline(df, target)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)