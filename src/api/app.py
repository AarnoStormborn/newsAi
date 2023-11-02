from fastapi import FastAPI
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

@app.get("/")
async def index():

    return {"Message: WELCOME TO END-TO-END NEWS CLASSIFICATION & SENTIMENT ANALYSIS"}

@app.post("/infer")
async def infer(data: PredictionInputSchema) -> t.Optional[t.Dict]:

    data = pd.Series(data=[data.model_dump()['news_article']])
    prediction = predict_pipe.predict(data=data)
    print(prediction)
    return {
        "Category":prediction['category'][0],
        "Sentiment": "Positive" if prediction['sentiment_probability'][0] == 1 else "Negative"
    }


if __name__ == "__main__":
    uvicorn.run(app=app, port=8085, host="0.0.0.0")
