from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from scoring import full_pipeline
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


class InputModel(BaseModel):
    review: str
    ingredients: str
    skin_type: str


@app.post("/analyze")
def analyze(data: InputModel):
    try:
        skin_num = int(data.skin_type)

        result = full_pipeline(
            data.review,
            data.ingredients,
            skin_num
        )

        print("=== FULL PIPELINE RESULT ===")
        print(result)
        print("============================")

        return {
            "최종점수": result.get("최종점수"),
            "피부타입": result.get("피부타입"),
            "예측고민": result.get("예측고민"),
            "성분가이드": result.get("성분가이드"),
        }

    except Exception as e:
        print("### ERROR in /analyze ###")
        print(e)
        return JSONResponse(status_code=500, content={"error": str(e)})
