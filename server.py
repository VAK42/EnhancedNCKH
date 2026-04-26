from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import logging
import joblib
import torch
import shap
import time
import json
import sys
import os
import io
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
outputDir = Path("nckh")
ensemblePath = outputDir / "stackingEnsemble.pkl"
scalerPath = outputDir / "scaler.pkl"
LSTMPath = outputDir / "LSTMModel.pt"
seqNormPath = outputDir / "seqNorm.pkl"
seqLen = 30
LSTMHidden = 64
LSTMLayers = 2
LSTMDropout = 0.3
riskThresh = 0.5
topSHAPK = 10
tabularFeatures = [
  'total_clicks', 'mean_clicks', 'std_clicks', 'max_clicks', 'min_clicks',
  'n_sessions', 'mean_gap', 'std_gap', 'max_gap', 'engagement_cliff',
  'first_access', 'last_access', 'active_span', 'activity_rate', 'late_drop',
  'click_slope', 'cv_clicks', 'n_activity_types', 'activity_entropy',
  'quiz_ratio', 'forum_ratio', 'abandon_rate', 'rolling_vol_7d',
  'mean_score', 'std_score', 'min_score', 'max_score',
  'n_submitted', 'fail_rate', 'score_slope', 'late_submit',
  'gender', 'age_band', 'highest_education',
  'num_of_prev_attempts', 'studied_credits', 'disability',
]
seqFeatures = ['sum_click', 'day_of_week', 'week_num', 'activity_enc']
class StressLSTM(nn.Module):
  def __init__(self, inputSize=4, hiddenSize=LSTMHidden, numLayers=LSTMLayers, dropout=LSTMDropout, nClasses=2):
    super().__init__()
    self.proj = nn.Linear(inputSize, hiddenSize)
    self.lstm = nn.LSTM(hiddenSize, hiddenSize, numLayers, batch_first=True, dropout=dropout, bidirectional=True)
    self.attention = nn.Linear(hiddenSize * 2, 1)
    self.classifier = nn.Sequential(
      nn.Linear(hiddenSize * 2, 64),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(64, nClasses)
    )
  def forward(self, x):
    x = torch.relu(self.proj(x))
    out, _ = self.lstm(x)
    attn = torch.softmax(self.attention(out), dim=1)
    ctx = (attn * out).sum(dim=1)
    return self.classifier(ctx)
  def getEmbedding(self, x):
    x = torch.relu(self.proj(x))
    out, _ = self.lstm(x)
    attn = torch.softmax(self.attention(out), dim=1)
    return (attn * out).sum(dim=1)
class ModelRegistry:
  ensemble = None
  scaler = None
  LSTMModel = None
  SHAPExplainer = None
  featureCols = []
  loadedAt = None
  youdenThresh = riskThresh
  seqMean = None
  seqStd = None
  @classmethod
  def load(cls):
    logger.info("Loading Model Artifacts!")
    if not ensemblePath.exists():
      raise RuntimeError(f"Ensemble Not Found At {ensemblePath}! Run main.py First!")
    cls.ensemble = joblib.load(ensemblePath)
    cls.scaler = joblib.load(scalerPath)
    logger.info("Stacking Ensemble Loaded!")
    if LSTMPath.exists():
      cls.LSTMModel = StressLSTM()
      cls.LSTMModel.load_state_dict(torch.load(LSTMPath, map_location='cpu'))
      cls.LSTMModel.eval()
      logger.info("LSTM Model Loaded!")
    else:
      logger.warning(f"LSTM Not Found At {LSTMPath} — Using Tabular Only!")
    if seqNormPath.exists():
      cls.seqMean, cls.seqStd = joblib.load(seqNormPath)
      logger.info("Sequence Normalization Parameters Loaded!")
    else:
      logger.warning(f"seqNorm.pkl Not Found — Falling Back To Local Normalization!")
    try:
      xgbBase = cls.ensemble.named_estimators_['xgb']
      cls.SHAPExplainer = shap.TreeExplainer(xgbBase)
      logger.info("SHAP Explainer Ready!")
    except Exception as e:
      logger.warning(f"SHAP Explainer Unavailable: {e}")
    try:
      nFeats = cls.scaler.n_features_in_
      if nFeats > len(tabularFeatures):
        cls.featureCols = tabularFeatures + ['LSTM_prob']
      else:
        cls.featureCols = tabularFeatures
    except Exception:
      cls.featureCols = tabularFeatures
    cls.loadedAt = datetime.utcnow().isoformat()
    logger.info(f"Model Registry Ready! Features: {len(cls.featureCols)}")
registry = ModelRegistry()
app = FastAPI(
  title = "Stress Detection API",
  description = "Predict Academic Stress From LMS Digital Behavioral Signals!\n\nPipeline: OULAD VLE Features + Bi-LSTM Embeddings → XGBoost + LightGBM Stacking Ensemble!\n\nReturns Stress Probability, Risk Label & Top SHAP Feature Explanations!",
  version = "1.0.0",
  docs_url = "/docs",
  redoc_url = "/redoc",
)
app.add_middleware(
  CORSMiddleware,
  allow_origins = ["*"],
  allow_credentials = True,
  allow_methods = ["*"],
  allow_headers = ["*"],
)
@app.on_event("startup")
async def startupEvent():
  registry.load()
class VLEInteraction(BaseModel):
  date: int = Field(..., description="Day Number From Course Start")
  sumClick: int = Field(..., description="Number Of Clicks In This Session", ge=0)
  activityType: str = Field("resource", description="Activity Type: Resource, Quiz, Forumng, Homepage, …")
  class Config:
    schema_extra = {"example": {"date": 14, "sumClick": 7, "activityType": "quiz"}}
class AssessmentRecord(BaseModel):
  score: float = Field(..., ge=0, le=100)
  dateSubmitted: int = Field(..., description="Day Submitted")
  dueDate: Optional[int] = Field(None, description="Due Date For Late Detection")
class StudentRequest(BaseModel):
  studentId: str = Field(..., description="Your Internal Student ID")
  VLELogs: List[VLEInteraction] = Field(..., description="Chronological LMS Interaction Log")
  assessments: Optional[List[AssessmentRecord]] = Field(default=[])
  gender: Optional[int] = Field(1, ge=0, le=1)
  ageBand: Optional[int] = Field(1, ge=0, le=2)
  highestEducation: Optional[int] = Field(1, ge=0, le=3)
  numPrevAttempts: Optional[int] = Field(0, ge=0)
  studiedCredits: Optional[int] = Field(60, ge=0)
  disability: Optional[int] = Field(0, ge=0, le=1)
  @validator('VLELogs')
  def minInteractions(cls, v):
    if len(v) < 5:
      raise ValueError("Need At Least 5 VLE Interactions For Reliable Prediction!")
    return v
  class Config:
    schema_extra = {
      "example": {
        "studentId": "u12345",
        "VLELogs": [{"date": 1, "sumClick": 12, "activityType": "homepage"}],
        "assessments": [{"score": 38, "dateSubmitted": 10, "dueDate": 8}],
        "gender": 1, "ageBand": 0, "studiedCredits": 60,
      }
    }
class FeatureImportance(BaseModel):
  feature: str
  SHAPValue: float
  direction: str
class PredictionResponse(BaseModel):
  studentId: str
  stressProb: float = Field(..., description="P(Stressed) ∈ [0, 1]")
  riskLabel: str = Field(..., description="'High Risk' | 'Low Risk'")
  riskThreshold: float
  confidence: str = Field(..., description="'High' | 'Medium' | 'Low'")
  topFeatures: List[FeatureImportance]
  nInteractions: int
  processedAt: str
class BatchResponse(BaseModel):
  total: int
  highRiskCount: int
  lowRiskCount: int
  predictions: List[PredictionResponse]
  processedAt: str
class HealthResponse(BaseModel):
  status: str
  modelLoaded: bool
  LSTMLoaded: bool
  SHAPReady: bool
  loadedAt: Optional[str]
class ModelInfoResponse(BaseModel):
  modelType: str
  featureCount: int
  featureNames: List[str]
  youdenThreshold: float
  loadedAt: str
activityMap = {
  'resource': 0, 'quiz': 1, 'forumng': 2, 'homepage': 3,
  'oucontent': 4, 'subpage': 5, 'dataplus': 6
}
def entropyCalc(probs):
  probs = np.array(probs)
  probs = probs[probs > 0]
  return -np.sum(probs * np.log2(probs + 1e-10))
def extractTabularFeatures(req: StudentRequest) -> np.ndarray:
  logs = req.VLELogs
  dates = np.array([l.date for l in logs])
  clicks = np.array([l.sumClick for l in logs], dtype=float)
  actTypes = [l.activityType for l in logs]
  order = np.argsort(dates)
  dates = dates[order]; clicks = clicks[order]
  actTypes = [actTypes[i] for i in order]
  nSess = len(logs)
  totalClicks = clicks.sum()
  meanClicks = clicks.mean()
  stdClicks = clicks.std() if nSess > 1 else 0
  maxClicks = clicks.max()
  minClicks = clicks.min()
  cvClicks = stdClicks / (meanClicks + 1e-6)
  dateDiffs = np.diff(dates) if nSess > 1 else np.array([0])
  meanGap = dateDiffs.mean()
  stdGap = dateDiffs.std() if len(dateDiffs) > 1 else 0
  maxGap = dateDiffs.max() if len(dateDiffs) > 0 else 0
  engagementCliff = maxGap / (meanGap + 1e-6)
  firstAccess = dates.min(); lastAccess = dates.max()
  activeSpan = lastAccess - firstAccess + 1
  activityRate = nSess / (activeSpan + 1e-6)
  mid = activeSpan / 2
  earlySum = clicks[dates <= (firstAccess + mid)].sum()
  lateSum = clicks[dates > (firstAccess + mid)].sum()
  lateDrop = (earlySum - lateSum) / (earlySum + 1e-6)
  clickSlope = np.polyfit(np.arange(nSess), clicks, 1)[0] if nSess > 2 else 0.0
  actCounts = {}
  for a in actTypes:
    actCounts[a] = actCounts.get(a, 0) + 1
  nTypes = len(actCounts)
  totalA = sum(actCounts.values())
  typeEntropy = entropyCalc([v/totalA for v in actCounts.values()])
  quizRatio = actCounts.get('quiz', 0) / (nSess + 1e-6)
  forumRatio = actCounts.get('forumng', 0) / (nSess + 1e-6)
  abandonRate = (clicks <= 1).mean()
  rollingStds = []
  for d in range(int(dates.min()), int(dates.max()) + 1, 7):
    mask = (dates >= d) & (dates < d + 7)
    if mask.sum() > 1:
      rollingStds.append(clicks[mask].std())
  rollingVol7d = np.mean(rollingStds) if rollingStds else 0.0
  assessments = req.assessments or []
  if assessments:
    scores = np.array([a.score for a in assessments])
    meanScore = scores.mean(); stdScore = scores.std() if len(scores) > 1 else 0
    minScore = scores.min(); maxScore = scores.max()
    nSubmitted = len(scores)
    failRate = (scores < 40).mean()
    scoreSlope = (np.polyfit(np.arange(len(scores)), scores, 1)[0] if len(scores) > 2 else 0.0)
    lateSubmit = sum(1 for a in assessments if a.dueDate and a.dateSubmitted > a.dueDate) / (nSubmitted + 1e-6)
  else:
    meanScore = stdScore = minScore = maxScore = 0.0
    nSubmitted = failRate = scoreSlope = lateSubmit = 0.0
  vec = np.array([
    totalClicks, meanClicks, stdClicks, maxClicks, minClicks,
    nSess, meanGap, stdGap, maxGap, engagementCliff,
    firstAccess, lastAccess, activeSpan, activityRate, lateDrop,
    clickSlope, cvClicks, nTypes, typeEntropy,
    quizRatio, forumRatio, abandonRate, rollingVol7d,
    meanScore, stdScore, minScore, maxScore,
    nSubmitted, failRate, scoreSlope, lateSubmit,
    req.gender if req.gender is not None else 1,
    req.ageBand if req.ageBand is not None else 1,
    req.highestEducation if req.highestEducation is not None else 1,
    req.numPrevAttempts if req.numPrevAttempts is not None else 0,
    req.studiedCredits if req.studiedCredits is not None else 60,
    req.disability if req.disability is not None else 0,
  ], dtype=np.float32)
  return vec
def extractSequence(req: StudentRequest) -> np.ndarray:
  logs = req.VLELogs
  dates = np.array([l.date for l in logs])
  clicks = np.array([l.sumClick for l in logs], dtype=np.float32)
  acts = np.array([activityMap.get(l.activityType, 0) for l in logs], dtype=np.float32)
  order = np.argsort(dates)
  dates = dates[order]; clicks = clicks[order]; acts = acts[order]
  weekNum = (dates // 7).clip(0, 52).astype(np.float32)
  dayOfWeek = (dates % 7).astype(np.float32)
  seq = np.stack([clicks, dayOfWeek, weekNum, acts], axis=1)
  if len(seq) >= seqLen:
    seq = seq[-seqLen:]
  else:
    pad = np.zeros((seqLen - len(seq), 4), dtype=np.float32)
    seq = np.vstack([pad, seq])
  if registry.seqMean is not None and registry.seqStd is not None:
    seq = (seq - registry.seqMean) / registry.seqStd
  else:
    seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-6)
  return seq[np.newaxis, :]
def getLSTMProbability(seq: np.ndarray) -> np.ndarray:
  if registry.LSTMModel is None:
    return None
  t = torch.tensor(seq)
  with torch.no_grad():
    logits = registry.LSTMModel(t)
    prob = torch.softmax(logits, dim=1)[:, 1].numpy()
  return prob
def buildFullFeatureVector(tabular: np.ndarray, LSTMProb: Optional[np.ndarray]) -> np.ndarray:
  if LSTMProb is not None:
    return np.concatenate([tabular, LSTMProb])
  return tabular
def getSHAPExplanation(xVec: np.ndarray) -> List[FeatureImportance]:
  if registry.SHAPExplainer is None:
    return []
  try:
    xSc = registry.scaler.transform(xVec.reshape(1, -1))
    sv = registry.SHAPExplainer.shap_values(xSc)[0]
    featN = registry.featureCols[:len(sv)]
    topK = np.argsort(np.abs(sv))[::-1][:topSHAPK]
    return [
      FeatureImportance(
        feature = featN[i] if i < len(featN) else f'feature_{i}',
        SHAPValue = float(sv[i]),
        direction = 'Increases Risk' if sv[i] > 0 else 'Decreases Risk'
      )
      for i in topK
    ]
  except Exception as e:
    logger.warning(f"SHAP Failed: {e}")
    return []
def confidenceLabel(prob: float, thresh: float) -> str:
  margin = abs(prob - thresh)
  if margin > 0.30: return 'High'
  if margin > 0.15: return 'Medium'
  return 'Low'
def predictStudent(req: StudentRequest) -> PredictionResponse:
  tabular = extractTabularFeatures(req)
  seq = extractSequence(req)
  LSTMProb = getLSTMProbability(seq)
  xFull = buildFullFeatureVector(tabular, LSTMProb)
  nExpected = registry.scaler.n_features_in_
  if len(xFull) < nExpected:
    xFull = np.concatenate([xFull, np.zeros(nExpected - len(xFull))])
  elif len(xFull) > nExpected:
    xFull = xFull[:nExpected]
  xSc = registry.scaler.transform(xFull.reshape(1, -1))
  prob = float(registry.ensemble.predict_proba(xSc)[0, 1])
  label = 'High Risk' if prob >= registry.youdenThresh else 'Low Risk'
  shapFeats = getSHAPExplanation(xFull)
  return PredictionResponse(
    studentId = req.studentId,
    stressProb = round(prob, 4),
    riskLabel = label,
    riskThreshold = registry.youdenThresh,
    confidence = confidenceLabel(prob, registry.youdenThresh),
    topFeatures = shapFeats,
    nInteractions = len(req.VLELogs),
    processedAt = datetime.utcnow().isoformat(),
  )
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
  return HealthResponse(
    status = "Ok" if registry.ensemble is not None else "Model Not Loaded!",
    modelLoaded = registry.ensemble is not None,
    LSTMLoaded = registry.LSTMModel is not None,
    SHAPReady = registry.SHAPExplainer is not None,
    loadedAt = registry.loadedAt,
  )
@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def modelInfo():
  if registry.ensemble is None:
    raise HTTPException(503, "Model Not Loaded!")
  return ModelInfoResponse(
    modelType = type(registry.ensemble).__name__,
    featureCount = len(registry.featureCols),
    featureNames = registry.featureCols,
    youdenThreshold = registry.youdenThresh,
    loadedAt = registry.loadedAt or "Unknown",
  )
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(req: StudentRequest):
  if registry.ensemble is None:
    raise HTTPException(503, "Model Not Loaded! Run main.py First!")
  try:
    t0 = time.time()
    result = predictStudent(req)
    logger.info(f"Predicted {req.studentId}: P={result.stressProb:.3f} Label={result.riskLabel} In {(time.time()-t0)*1000:.1f}ms")
    return result
  except ValueError as e:
    raise HTTPException(422, str(e))
  except Exception as e:
    logger.error(f"Prediction Error: {e}")
    raise HTTPException(500, f"Prediction Failed: {e}")
@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predictBatch(file: UploadFile = File(...)):
  if registry.ensemble is None:
    raise HTTPException(503, "Model Not Loaded!")
  try:
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
  except Exception as e:
    raise HTTPException(400, f"Could Not Parse CSV: {e}")
  if 'studentId' not in df.columns:
    raise HTTPException(400, "CSV Must Have A 'studentId' Column!")
  featCols = [c for c in tabularFeatures if c in df.columns]
  missing = [c for c in tabularFeatures if c not in df.columns]
  if missing:
    logger.warning(f"Missing Columns Filled With 0: {missing}")
    for c in missing:
      df[c] = 0
  X = df[tabularFeatures].fillna(0).values
  nExp = registry.scaler.n_features_in_
  if X.shape[1] < nExp:
    X = np.hstack([X, np.zeros((len(X), nExp - X.shape[1]))])
  elif X.shape[1] > nExp:
    X = X[:, :nExp]
  X_sc = registry.scaler.transform(X)
  probas = registry.ensemble.predict_proba(X_sc)[:, 1]
  thresh = registry.youdenThresh
  labels = ['High Risk' if p >= thresh else 'Low Risk' for p in probas]
  predictions = []
  for i, row in df.iterrows():
    shapFeats = getSHAPExplanation(X[i])
    predictions.append(PredictionResponse(
      studentId = str(row['studentId']),
      stressProb = round(float(probas[i]), 4),
      riskLabel = labels[i],
      riskThreshold = thresh,
      confidence = confidenceLabel(float(probas[i]), thresh),
      topFeatures = shapFeats,
      nInteractions = 0,
      processedAt = datetime.utcnow().isoformat(),
    ))
  nHigh = sum(1 for l in labels if l == 'High Risk')
  return BatchResponse(
    total = len(predictions),
    highRiskCount = nHigh,
    lowRiskCount = len(predictions) - nHigh,
    predictions = predictions,
    processedAt = datetime.utcnow().isoformat(),
  )
@app.post("/threshold/update", tags=["System"])
async def updateThreshold(threshold: float):
  if not (0.0 < threshold < 1.0):
    raise HTTPException(422, "Threshold Must Be In (0, 1)!")
  old = registry.youdenThresh
  registry.youdenThresh = threshold
  logger.info(f"Threshold Updated: {old:.3f} → {threshold:.3f}")
  return {"oldThreshold": old, "newThreshold": threshold}
if __name__ == "__main__":
  import uvicorn
  uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)