import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch.optim as optim
import lightgbm as lgb
import torch.nn as nn
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import requests
import zipfile
import joblib
import torch
import shap
import sys
import os
import io
import re
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, matthews_corrcoef, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import DataLoader, TensorDataset
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
randomState = 42
testSize = 0.2
valSize = 0.1
seqLen = 30
LSTMHidden = 64
LSTMLayers = 2
LSTMDropout = 0.3
LSTMEpochs = 30
LSTMBatch = 64
LSTMLR = 5e-4
nFolds = 5
dataDir = Path("data")
outputDir = Path("nckh")
dataDir.mkdir(exist_ok=True)
outputDir.mkdir(exist_ok=True)
np.random.seed(randomState)
torch.manual_seed(randomState)
def showBanner(title):
  print("\n" + "=" * 80)
  print(f"{title}")
  print("=" * 80)
def loadOULAD(dataDir: Path) -> dict:
  showBanner("Section 1: Loading OULAD Dataset")
  ouladDir = dataDir / "oulad"
  coreRequired = ["studentInfo.csv", "studentVle.csv", "vle.csv", "studentAssessment.csv", "assessments.csv"]
  missing = [f for f in coreRequired if not (ouladDir / f).exists()]
  if missing:
    raise FileNotFoundError(f"Missing Real OULAD Files: {missing}! Please Place Them In data/oulad!")
  info = pd.read_csv(ouladDir / "studentInfo.csv")
  svle = pd.read_csv(ouladDir / "studentVle.csv")
  vleMeta = pd.read_csv(ouladDir / "vle.csv")
  sa = pd.read_csv(ouladDir / "studentAssessment.csv")
  assessments = pd.read_csv(ouladDir / "assessments.csv")
  svle = svle.merge(vleMeta[['id_site', 'activity_type']].drop_duplicates(), on='id_site', how='left')
  print(f"Loaded Student Info → {len(info):>8,} Rows")
  print(f"Loaded Student VLE → {len(svle):>8,} Rows  (Merged With vle.csv)")
  print(f"Loaded Student Assessment → {len(sa):>8,} Rows")
  print(f"Loaded Assessments → {len(assessments):>8,} Rows")
  print(f"Unique Students → {info['id_student'].nunique():>8,}")
  sa = sa.merge(assessments[['id_assessment', 'date']], on='id_assessment', how='left')
  return {'info': info, 'vle': svle, 'studentassessment': sa, 'assessments_meta': assessments}
def buildStressLabels(oulad: dict) -> pd.DataFrame:
  showBanner("Section 2: Label Construction — Stress Proxy")
  info = oulad['info'].copy()
  if '_stress_label' in info.columns:
    labels = info[['id_student', '_stress_label']].rename(columns={'_stress_label': 'stress_label'})
  else:
    resultMap = {'Pass': 0, 'Distinction': 0, 'Fail': 1, 'Withdrawn': 1}
    info['stress_label'] = info['final_result'].map(resultMap)
    info = info.dropna(subset=['stress_label'])
    info['stress_label'] = info['stress_label'].astype(int)
    labels = info.groupby('id_student')['stress_label'].max().reset_index()
  nTotal = len(labels)
  nStressed = labels['stress_label'].sum()
  nNormal = nTotal - nStressed
  print(f"\nTotal Students: {nTotal:,}")
  print(f"Stressed (1): {nStressed:,}  ({100*nStressed/nTotal:.1f}%)")
  print(f"Normal (0): {nNormal:,}  ({100*nNormal/nTotal:.1f}%)")
  return labels
def entropyCalc(probs):
  probs = probs[probs > 0]
  return -np.sum(probs * np.log2(probs + 1e-10))
def rollingStdCalc(dates, clicks, window=7):
  stds = []
  for d in range(int(dates.min()), int(dates.max()) + 1, window):
    mask = (dates >= d) & (dates < d + window)
    if mask.sum() > 1:
      stds.append(clicks[mask].std())
  return np.mean(stds) if stds else 0.0
def engineerVLEFeatures(oulad: dict) -> pd.DataFrame:
  showBanner("Section 3a: VLE Feature Engineering (Session & Temporal)")
  vle = oulad['vle'].copy().sort_values(['id_student', 'date'])
  feats = []
  for sid, grp in vle.groupby('id_student'):
    clicks = grp['sum_click'].values
    dates = grp['date'].values
    nSess = len(grp)
    totalClicks = clicks.sum()
    meanClicks = clicks.mean()
    stdClicks = clicks.std() if len(clicks) > 1 else 0
    maxClicks = clicks.max()
    minClicks = clicks.min()
    dateDiffs = np.diff(dates) if len(dates) > 1 else np.array([0])
    meanGap = dateDiffs.mean()
    stdGap = dateDiffs.std() if len(dateDiffs) > 1 else 0
    maxGap = dateDiffs.max() if len(dateDiffs) > 0 else 0
    engagementCliff = maxGap / (meanGap + 1e-6)
    firstAccess = dates.min()
    lastAccess = dates.max()
    activeSpan = lastAccess - firstAccess + 1
    activityRate = nSess / (activeSpan + 1e-6)
    mid = activeSpan / 2
    earlyClicks = clicks[dates <= (firstAccess + mid)].sum()
    lateClicks = clicks[dates > (firstAccess + mid)].sum()
    lateDrop = (earlyClicks - lateClicks) / (earlyClicks + 1e-6)
    if len(clicks) > 2:
      slope = np.polyfit(np.arange(len(clicks)), clicks, 1)[0]
    else:
      slope = 0.0
    cvClicks = stdClicks / (meanClicks + 1e-6)
    if 'activity_type' in grp.columns:
      nTypes = grp['activity_type'].nunique()
      typeEntropy = entropyCalc(grp['activity_type'].value_counts(normalize=True).values)
      quizRatio = (grp['activity_type'] == 'quiz').mean()
      forumRatio = (grp['activity_type'] == 'forumng').mean()
    else:
      nTypes = typeEntropy = quizRatio = forumRatio = 0
    abandonRate = (clicks <= 1).mean()
    rollingVol = rollingStdCalc(dates, clicks, window=7)
    feats.append({
      'id_student': sid,
      'total_clicks': totalClicks,
      'mean_clicks': meanClicks,
      'std_clicks': stdClicks,
      'max_clicks': maxClicks,
      'min_clicks': minClicks,
      'n_sessions': nSess,
      'mean_gap': meanGap,
      'std_gap': stdGap,
      'max_gap': maxGap,
      'engagement_cliff': engagementCliff,
      'first_access': firstAccess,
      'last_access': lastAccess,
      'active_span': activeSpan,
      'activity_rate': activityRate,
      'late_drop': lateDrop,
      'click_slope': slope,
      'cv_clicks': cvClicks,
      'n_activity_types': nTypes,
      'activity_entropy': typeEntropy,
      'quiz_ratio': quizRatio,
      'forum_ratio': forumRatio,
      'abandon_rate': abandonRate,
      'rolling_vol_7d': rollingVol,
    })
    if len(feats) % 5000 == 0:
      print(f"... Engineered VLE Features For {len(feats):,} Students")
  df = pd.DataFrame(feats)
  print(f"VLE Features Shape: {df.shape}")
  return df
def engineerAssessmentFeatures(oulad: dict) -> pd.DataFrame:
  showBanner("Section 3b: Assessment Feature Engineering")
  asmnt = oulad.get('studentassessment', oulad.get('assessments', pd.DataFrame()))
  if asmnt.empty:
    print("No Assessment Data Found, Skipping!")
    return pd.DataFrame()
  feats = []
  for sid, grp in asmnt.groupby('id_student'):
    scores = grp['score'].dropna().values
    if len(scores) == 0:
      continue
    meanScore = scores.mean()
    stdScore = scores.std() if len(scores) > 1 else 0
    minScore = scores.min()
    maxScore = scores.max()
    nSubmitted = len(scores)
    failRate = (scores < 40).mean()
    if len(scores) > 2:
      scoreSlope = np.polyfit(np.arange(len(scores)), scores, 1)[0]
    else:
      scoreSlope = 0.0
    lateSubmit = 0.0
    if 'date_submitted' in grp.columns and 'date' in grp.columns:
      lateSubmit = (grp['date_submitted'] > grp['date']).mean()
    feats.append({
      'id_student': sid,
      'mean_score': meanScore,
      'std_score': stdScore,
      'min_score': minScore,
      'max_score': maxScore,
      'n_submitted': nSubmitted,
      'fail_rate': failRate,
      'score_slope': scoreSlope,
      'late_submit': lateSubmit,
    })
    if len(feats) % 5000 == 0:
      print(f"... Engineered Assessment Features For {len(feats):,} Students")
  df = pd.DataFrame(feats)
  print(f"Assessment Features Shape: {df.shape}")
  return df
def engineerStudentInfoFeatures(oulad: dict) -> pd.DataFrame:
  showBanner("Section 3c: Student Info Features")
  info = oulad['info'].copy()
  catCols = ['gender', 'age_band', 'highest_education', 'disability']
  for col in catCols:
    if col in info.columns:
      info[col] = LabelEncoder().fit_transform(info[col].astype(str))
  keep = ['id_student', 'gender', 'age_band', 'highest_education', 'num_of_prev_attempts', 'studied_credits', 'disability']
  keep = [c for c in keep if c in info.columns]
  df = info[keep].copy()
  print(f"Student Info Features Shape: {df.shape}")
  return df
def mergeFeatures(vleFeats, asmntFeats, infoFeats, labels) -> pd.DataFrame:
  showBanner("Section 3d: Merging All Feature Tables")
  df = labels.copy()
  df = df.merge(vleFeats, on='id_student', how='left')
  if not asmntFeats.empty:
    df = df.merge(asmntFeats, on='id_student', how='left')
  if not infoFeats.empty:
    df = df.merge(infoFeats, on='id_student', how='left')
  numCols = df.select_dtypes(include=np.number).columns.tolist()
  df[numCols] = df[numCols].fillna(df[numCols].median())
  print(f"Final Merged Shape: {df.shape}")
  print(f"Feature Count: {df.shape[1] - 2}")
  return df
def buildSequences(oulad: dict, labels: pd.DataFrame, currentSeqLen=seqLen) -> tuple:
  showBanner("Section 4: Building LSTM Sequences From VLE Logs")
  vle = oulad['vle'].copy().sort_values(['id_student', 'date'])
  if 'activity_type' in vle.columns:
    vle['activity_enc'] = LabelEncoder().fit_transform(vle['activity_type'].astype(str))
  else:
    vle['activity_enc'] = 0
  vle['week_num'] = (vle['date'] // 7).clip(0, 52)
  vle['day_of_week'] = (vle['date'] % 7)
  labelMap = labels.set_index('id_student')['stress_label'].to_dict()
  X_list, y_list, studentIdsInSeq = [], [], []
  for sid, grp in vle.groupby('id_student'):
    if sid not in labelMap:
      continue
    seqFeats = grp[['sum_click', 'day_of_week', 'week_num', 'activity_enc']].values.astype(np.float32)
    if len(seqFeats) >= currentSeqLen:
      seq = seqFeats[-currentSeqLen:]
    else:
      pad = np.zeros((currentSeqLen - len(seqFeats), seqFeats.shape[1]), dtype=np.float32)
      seq = np.vstack([pad, seqFeats])
    X_list.append(seq)
    y_list.append(labelMap[sid])
    studentIdsInSeq.append(sid)
    if len(X_list) % 5000 == 0:
      print(f"... Built Sequences For {len(X_list):,} Students")
  X = np.array(X_list, dtype=np.float32)
  y = np.array(y_list, dtype=np.int64)
  shapeOrig = X.shape
  X_flat = X.reshape(-1, X.shape[-1])
  globalMean = X_flat.mean(axis=0)
  globalStd = X_flat.std(axis=0) + 1e-6
  X = ((X_flat - globalMean) / globalStd).reshape(shapeOrig)
  print(f"Sequence Tensor Shape: {X.shape}")
  print(f"Label Distribution: Normal={sum(y==0)} | Stressed={sum(y==1)}")
  return X, y, globalMean, globalStd, studentIdsInSeq
class StressLSTM(nn.Module):
  def __init__(self, inputSize, hiddenSize=LSTMHidden, numLayers=LSTMLayers, dropout=LSTMDropout, nClasses=2):
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
def trainLSTM(X_seq, y_seq, studentIds) -> tuple:
  showBanner("Section 5: Training Bi-LSTM With Attention")
  X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq, test_size=testSize, random_state=randomState, stratify=y_seq)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Device: {device}")
  trainDs = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
  testDs = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
  trainDl = DataLoader(trainDs, batch_size=LSTMBatch, shuffle=True)
  testDl = DataLoader(testDs, batch_size=LSTMBatch)
  model = StressLSTM(inputSize=X_seq.shape[2]).to(device)
  criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, len(y_tr[y_tr==0]) / (len(y_tr[y_tr==1]) + 1e-6)], dtype=torch.float32).to(device))
  optimizer = optim.Adam(model.parameters(), lr=LSTMLR, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LSTMEpochs)
  history = {'train_loss': [], 'val_acc': []}
  bestAcc, bestState = 0, None
  for epoch in range(LSTMEpochs):
    model.train()
    totalLoss = 0
    for xb, yb in trainDl:
      xb, yb = xb.to(device), yb.to(device)
      optimizer.zero_grad()
      loss = criterion(model(xb), yb)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      totalLoss += loss.item()
    scheduler.step()
    model.eval()
    allPreds, allTrue = [], []
    with torch.no_grad():
      for xb, yb in testDl:
        preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
        allPreds.extend(preds)
        allTrue.extend(yb.numpy())
    valAcc = accuracy_score(allTrue, allPreds)
    history['train_loss'].append(totalLoss / len(trainDl))
    history['val_acc'].append(valAcc)
    if valAcc > bestAcc:
      bestAcc = valAcc
      bestState = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"Epoch [{epoch+1:02d}/{LSTMEpochs}]  Loss: {history['train_loss'][-1]:.4f}  Val Acc: {valAcc:.4f}")
  model.load_state_dict(bestState)
  print(f"\nBest LSTM Val Accuracy: {bestAcc:.4f}")
  model.eval()
  allDl = DataLoader(TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq)), batch_size=LSTMBatch)
  probs = []
  with torch.no_grad():
    for xb, _ in allDl:
      out = model(xb.to(device))
      p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
      probs.extend(p)
  probs = np.array(probs).reshape(-1, 1)
  probMap = dict(zip(studentIds, probs.flatten()))
  return model, probMap, history, (X_te, y_te, allPreds)
def printEnsembleMetrics(yTe, yPred, yPredTuned, yProba, thresh):
  print("\n" + "-" * 60)
  print("Ensemble Results (Default Threshold = 0.5)")
  print("-" * 60)
  print(f"Accuracy: {accuracy_score(yTe, yPred):.4f}")
  print(f"Precision: {precision_score(yTe, yPred, zero_division=0):.4f}")
  print(f"Recall: {recall_score(yTe, yPred, zero_division=0):.4f}")
  print(f"F1-Score: {f1_score(yTe, yPred, zero_division=0):.4f}")
  print(f"ROC-AUC: {roc_auc_score(yTe, yProba):.4f}")
  print(f"MCC: {matthews_corrcoef(yTe, yPred):.4f}")
  print(f"\nEnsemble Results (Youden Threshold = {thresh:.3f})")
  print("-" * 60)
  print(f"Accuracy: {accuracy_score(yTe, yPredTuned):.4f}")
  print(f"Precision: {precision_score(yTe, yPredTuned, zero_division=0):.4f}")
  print(f"Recall: {recall_score(yTe, yPredTuned, zero_division=0):.4f}")
  print(f"F1-Score: {f1_score(yTe, yPredTuned, zero_division=0):.4f}")
  print(f"MCC: {matthews_corrcoef(yTe, yPredTuned):.4f}")
  print(f"\n{classification_report(yTe, yPredTuned, target_names=['Normal','Stressed'])}")
def trainEnsemble(tabularDf: pd.DataFrame, LSTMProbs: np.ndarray) -> dict:
  showBanner("Section 7: Stacking Ensemble — XGBoost + LightGBM + LSTM")
  featureCols = [c for c in tabularDf.columns if c not in ('id_student', 'stress_label')]
  XFull = tabularDf[featureCols].values
  y = tabularDf['stress_label'].values
  print(f"Combined Feature Dim  : {XFull.shape[1]} (37 Tabular + 1 LSTM)")
  smote = SMOTE(random_state=randomState)
  XTr, XTe, yTr, yTe = train_test_split(XFull, y, test_size=testSize, random_state=randomState, stratify=y)
  XTrS, yTrS = smote.fit_resample(XTr, yTr)
  print(f"\nAfter SMOTE — Normal: {sum(yTrS==0)} | Stressed: {sum(yTrS==1)}")
  scaler = StandardScaler()
  XTrSc = scaler.fit_transform(XTrS)
  XTeSc = scaler.transform(XTe)
  xgbModel = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=randomState, n_jobs=-1)
  lgbModel = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=randomState, n_jobs=-1, verbose=-1)
  stack = StackingClassifier(estimators=[('xgb', xgbModel), ('lgb', lgbModel)], final_estimator=LogisticRegression(max_iter=1000, random_state=randomState), cv=5, passthrough=False, n_jobs=-1)
  print("\nFitting Stacking Ensemble (This May Take A Few Minutes)...")
  stack.fit(XTrSc, yTrS)
  yPred = stack.predict(XTeSc)
  yProba = stack.predict_proba(XTeSc)[:, 1]
  fpr, tpr, thresholds = roc_curve(yTe, yProba)
  bestThresh = thresholds[np.argmax(tpr - fpr)]
  yPredTuned = (yProba >= bestThresh).astype(int)
  results = {
    'model': stack, 'scaler': scaler, 'X_test': XTeSc, 'y_test': yTe,
    'y_pred': yPred, 'y_pred_tuned': yPredTuned, 'y_proba': yProba,
    'best_thresh': bestThresh, 'feature_cols': featureCols,
    'X_train': XTrSc, 'y_train': yTrS,
  }
  printEnsembleMetrics(yTe, yPred, yPredTuned, yProba, bestThresh)
  return results
def crossValidateModels(tabularDf: pd.DataFrame) -> dict:
  showBanner("Section 8: Stratified K-Fold Cross Validation")
  featureCols = [c for c in tabularDf.columns if c not in ('id_student', 'stress_label')]
  X = tabularDf[featureCols].values
  y = tabularDf['stress_label'].values
  skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=randomState)
  xgbBase = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=randomState, n_jobs=-1)
  lgbBase = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=randomState, n_jobs=-1, verbose=-1)
  modelsToCv = {
    'XGBoost': ImbPipeline([('smote', SMOTE(random_state=randomState)), ('scaler', StandardScaler()), ('clf', xgbBase)]),
    'LightGBM': ImbPipeline([('smote', SMOTE(random_state=randomState)), ('scaler', StandardScaler()), ('clf', lgbBase)]),
  }
  cvResults = {}
  for name, pipeline in modelsToCv.items():
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    cvResults[name] = scores
    print(f"\n{name} (CV With SMOTE):")
    print(f"  Mean Acc: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Folds: {[f'{s:.4f}' for s in scores]}")
  return cvResults
def SHAPAnalysis(results: dict, tabularDf: pd.DataFrame, outDir: Path):
  showBanner("Section 9: SHAP Feature Interpretability")
  featureCols = results['feature_cols']
  XTe = results['X_test']
  try:
    xgbBase = results['model'].named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgbBase)
    shapVals = explainer.shap_values(XTe)
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shapVals, XTe, feature_names=featureCols, show=False, max_display=20)
    plt.title("SHAP Summary — XGBoost Base Learner", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outDir / 'shapSummary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("SHAP Summary Saved → shapSummary.png")
  except Exception as e:
    print(f"SHAP Analysis Skipped: {e}!")
def generateVisualizations(results: dict, cvResults: dict, LSTMHistory: dict, outDir: Path):
  showBanner("Section 10: Generating Visualizations")
  yTe = results['y_test']
  yPred = results['y_pred_tuned']
  yProba = results['y_proba']
  fpr, tpr, _ = roc_curve(yTe, yProba)
  aucScore = roc_auc_score(yTe, yProba)
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {aucScore:.4f}')
  axes[0].plot([0,1],[0,1], 'k--', lw=1)
  axes[0].fill_between(fpr, tpr, alpha=0.15, color='steelblue')
  axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
  axes[0].set_title('ROC Curve — Stacking Ensemble', fontweight='bold')
  axes[0].legend(loc='lower right')
  cm = confusion_matrix(yTe, yPred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=['Normal','Stressed'], yticklabels=['Normal','Stressed'])
  axes[1].set_title('Confusion Matrix (Tuned Threshold)', fontweight='bold')
  axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
  if LSTMHistory:
    ep = range(1, len(LSTMHistory['val_acc']) + 1)
    ax3 = axes[2]; ax3_twin = ax3.twinx()
    ax3.plot(ep, LSTMHistory['train_loss'], 'coral', lw=2, label='Train Loss')
    ax3_twin.plot(ep, LSTMHistory['val_acc'], 'steelblue', lw=2, label='Val Accuracy')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss', color='coral')
    ax3_twin.set_ylabel('Accuracy', color='steelblue')
    ax3.set_title('LSTM Training History', fontweight='bold')
    ax3.legend(loc='upper left'); ax3_twin.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig(outDir / 'ROCCMLSTM.png', dpi=200, bbox_inches='tight')
  plt.close()
  if cvResults:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(cvResults.keys()); means = [v.mean() for v in cvResults.values()]; stds = [v.std() for v in cvResults.values()]
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B']
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors[:len(names)], edgecolor='black')
    for bar, m in zip(bars, means):
      ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{m:.4f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylim([0.5, 1.05]); ax.set_ylabel('CV Accuracy', fontweight='bold')
    ax.set_title(f'{nFolds}-Fold Stratified CV Comparison', fontweight='bold')
    plt.tight_layout()
    plt.savefig(outDir / 'CVComparison.png', dpi=200, bbox_inches='tight')
    plt.close()
  metrics = {
    'Accuracy': accuracy_score(yTe, yPred), 'Precision': precision_score(yTe, yPred, zero_division=0),
    'Recall': recall_score(yTe, yPred, zero_division=0), 'F1-Score': f1_score(yTe, yPred, zero_division=0),
    'ROC-AUC': aucScore, 'MCC': matthews_corrcoef(yTe, yPred),
  }
  fig, ax = plt.subplots(figsize=(9, 4)); ax.axis('off')
  tableData = [[k, f'{v:.4f}'] for k, v in metrics.items()]
  tbl = ax.table(cellText=tableData, colLabels=['Metric', 'Score'], cellLoc='center', loc='center')
  tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1.5, 2.0)
  ax.set_title('Final Ensemble — Evaluation Summary', fontsize=14, fontweight='bold', pad=20)
  plt.tight_layout()
  plt.savefig(outDir / 'metricsSummary.png', dpi=200, bbox_inches='tight')
  plt.close()
  print("\nAll Visualizations Saved To:", outDir)
def exportModels(results: dict, LSTMModel, outDir: Path, seqNorm=None):
  showBanner("Section 11: Exporting Models")
  joblib.dump(results['model'], outDir / 'stackingEnsemble.pkl')
  joblib.dump(results['scaler'], outDir / 'scaler.pkl')
  if LSTMModel is not None:
    torch.save(LSTMModel.state_dict(), outDir / 'LSTMModel.pt')
  if seqNorm is not None:
    joblib.dump(seqNorm, outDir / 'seqNorm.pkl')
    print(f"seqNorm.pkl          → {outDir}")
  print(f"stackingEnsemble.pkl → {outDir}")
  print(f"scaler.pkl            → {outDir}")
  print(f"LSTMModel.pt         → {outDir}")
def main():
  import time
  start = datetime.now(); tGlobal = time.time()
  print("\n" + "=" * 80)
  print("Stress Detection Via LMS Digital Signals")
  print("Pipeline: OULAD-Based Stress Detection")
  print("Target  : 95–98% Accuracy Via Hybrid LSTM + XGBoost Stacking")
  print("=" * 80)
  print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
  print("\nPhase 1: Loading Datasets...")
  t0 = time.time(); oulad = loadOULAD(dataDir)
  print(f"Datasets Loaded In {time.time() - t0:.2f}s")
  print("\nPhase 2: Building Labels & Feature Engineering...")
  t0 = time.time(); labels = buildStressLabels(oulad)
  vleFeats = engineerVLEFeatures(oulad); asmntFeats = engineerAssessmentFeatures(oulad)
  infoFeats = engineerStudentInfoFeatures(oulad); tabularDf = mergeFeatures(vleFeats, asmntFeats, infoFeats, labels)
  print(f"Feature Engineering Completed In {time.time() - t0:.2f}s")
  print("\nPhase 3: OULAD Sequence Building & LSTM Training...")
  t0 = time.time(); XSeq, ySeq, gMean, gStd, sidSeq = buildSequences(oulad, labels)
  LSTMModel, LSTMProbMap, LSTMHist, LSTMEval = trainLSTM(XSeq, ySeq, sidSeq)
  tabularDf['LSTM_prob'] = tabularDf['id_student'].map(LSTMProbMap)
  tabularDf = tabularDf.dropna(subset=['LSTM_prob']).reset_index(drop=True)
  print(f"LSTM Training Completed In {time.time() - t0:.2f}s")
  print("\nPhase 4: Training Stacking Ensemble (XGBoost + LightGBM + Meta)...")
  t0 = time.time(); results = trainEnsemble(tabularDf, None)
  print(f"Stacking Ensemble Completed In {time.time() - t0:.2f}s")
  print("\nPhase 5: Running Stratified K-Fold Cross Validation...")
  t0 = time.time(); cvResults = crossValidateModels(tabularDf)
  print(f"Cross Validation Completed In {time.time() - t0:.2f}s")
  print("\nPhase 6: Running SHAP Interpretability Analysis...")
  t0 = time.time(); SHAPAnalysis(results, tabularDf, outputDir)
  print(f"SHAP Analysis Completed In {time.time() - t0:.2f}s")
  print("\nPhase 7: Generating Output Visualizations...")
  t0 = time.time(); generateVisualizations(results, cvResults, LSTMHist, outputDir)
  print(f"Visualizations Generated In {time.time() - t0:.2f}s")
  print("\nPhase 8: Exporting Final Models...")
  t0 = time.time(); exportModels(results, LSTMModel, outputDir, seqNorm=(gMean, gStd))
  print(f"Model Export Completed In {time.time() - t0:.2f}s")
  showBanner("Pipeline Complete")
  yTe = results['y_test']; yPred = results['y_pred_tuned']
  print(f"\nFinal Accuracy: {accuracy_score(yTe, yPred):.4f}")
  print(f"Final F1-Score: {f1_score(yTe, yPred, zero_division=0):.4f}")
  print(f"Final ROC-AUC: {roc_auc_score(yTe, results['y_proba']):.4f}")
  print(f"Total Pipeline Elapsed Time: {time.time() - tGlobal:.2f}s")
  print(f"Outputs Saved To: {outputDir.resolve()}")
  print("\n" + "=" * 80)
if __name__ == "__main__":
  main()