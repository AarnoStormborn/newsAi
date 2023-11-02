from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import typing as t
import pandas as pd
import uvicorn
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils import read_config
from src import constant
from pydantic import BaseModel

config = read_config(filepath=constant.CONFIG_FILE).model_prediction
predict_pipe = PredictionPipeline(config=config)

class PredictionInputSchema(BaseModel):
    news_article: str

app = FastAPI(name="NewsAi", description="I loveeeeeee NLP")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("home.html", {"request":request})


@app.post("/", response_class=HTMLResponse)
def infer(request: Request, news_article: str=Form(...)):
    data = pd.Series(data=[news_article])
    prediction = predict_pipe.predict(data=data)
    category = prediction['category'][0]
    sentiment = prediction['sentiment']
    context = {
        "request": request,
        "article": news_article,
        "category": category.capitalize(),
        "sentiment": sentiment
    }
    return templates.TemplateResponse("results.html", context=context)

@app.post("/api")
async def api_infer(data: PredictionInputSchema) -> t.Optional[t.Dict]:

    data = pd.Series(data=[data.model_dump()['news_article']])
    prediction = predict_pipe.predict(data=data)
    return {
        "Category":prediction['category'][0],
        "Sentiment": "Positive" if prediction['sentiment_probability'][0] == 1 else "Negative"
    }


if __name__ == "__main__":
    uvicorn.run(app=app, port=8085, host="0.0.0.0")
