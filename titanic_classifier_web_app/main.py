import torch
from torch import nn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import joblib
from pathlib import Path

# ─── Model Definition ────────────────────────────────────────────────────────

class TitanicClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


# ─── Load Model & Scaler ─────────────────────────────────────────────────────

BASE = Path(__file__).parent

model = TitanicClassifier()
model.load_state_dict(torch.load(BASE / "titanic_model.pth", map_location="cpu"))
model.eval()

scaler = joblib.load(BASE / "scaler.pkl")

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Titanic Survival Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


# ─── Schemas ─────────────────────────────────────────────────────────────────

class PassengerInput(BaseModel):
    pclass: int   = Field(..., ge=1, le=3)
    sex: int      = Field(..., ge=0, le=1)          # male=0, female=1
    age: float    = Field(..., ge=0, le=120)
    sibsp: int    = Field(..., ge=0)
    parch: int    = Field(..., ge=0)
    fare: float   = Field(..., ge=0)
    embarked_q: int = Field(..., ge=0, le=1)        # Queenstown: 1/0
    embarked_s: int = Field(..., ge=0, le=1)        # Southampton: 1/0


class PredictionResult(BaseModel):
    survived: bool
    survival_probability: float
    death_probability: float


# ─── Predict ─────────────────────────────────────────────────────────────────

# Feature column order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S
# scale_cols order used during training: Age, Fare, SibSp, Parch  (indices 2,5,3,4)

SCALE_IDX = [2, 5, 3, 4]   # positions of Age, Fare, SibSp, Parch in feature vector


@app.post("/predict", response_model=PredictionResult)
def predict(p: PassengerInput):
    fare_log = np.log1p(p.fare)   # same transform applied during training

    features = np.array([
        p.pclass,
        p.sex,
        p.age,
        p.sibsp,
        p.parch,
        fare_log,
        p.embarked_q,
        p.embarked_s,
    ], dtype=np.float64)

    # Apply StandardScaler to [Age, Fare, SibSp, Parch]
    scale_input = features[SCALE_IDX].reshape(1, -1)       # shape: (1, 4)
    features[SCALE_IDX] = scaler.transform(scale_input)[0]

    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        prob = torch.sigmoid(model(tensor)).item()

    return PredictionResult(
        survived=prob >= 0.5,
        survival_probability=round(prob, 4),
        death_probability=round(1 - prob, 4),
    )
