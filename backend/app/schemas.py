from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str                  # predicted tumor class
    confidence: float                # probability of the predicted class (0-1)
    probabilities: dict[str, float]  # probability of every class