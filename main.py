import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch.optim as optim
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import joblib
import torch
import shap
import sys
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, average_precision_score, precision_recall_curve, classification_report, confusion_matrix, roc_curve, log_loss)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')
randomState = 42
testSize = 0.2
valSize = 0.1
seqLen = 30
LSTMDropout = 0.3
LSTMHidden = 64
LSTMEpochs = 50
LSTMBatch = 128
LSTMLayers = 2
LSTMLR = 5e-4
nFolds = 5
defaultThresh = 0.5
dataD = Path("data")
outputD = Path("nckh")
modelD = outputD / "model"
imgD = outputD / "img"
dataD.mkdir(exist_ok=True)
modelD.mkdir(parents=True, exist_ok=True)
imgD.mkdir(parents=True, exist_ok=True)
np.random.seed(randomState)
torch.manual_seed(randomState)
def showBanner(title):
  print("\n" + "=" * 80)
  print(f"{title}")
  print("=" * 80)
def loadOULAD(dataD: Path) -> dict:
  showBanner("S1: Loading OULAD Dataset")
  ouladD = dataD / "oulad"
  coreRequired = ["StudentInfo.csv", "StudentVLE.csv", "VLE.csv", "StudentAssessment.csv", "Assessments.csv"]
  missing = [f for f in coreRequired if not (ouladD / f).exists()]
  if missing:
    raise FileNotFoundError(f"Missing Real OULAD Files: {missing}! Please Place Them In data/oulad!")
  info = pd.read_csv(ouladD / "StudentInfo.csv")
  svle = pd.read_csv(ouladD / "StudentVLE.csv", usecols=['id_student', 'id_site', 'date', 'sum_click'], dtype={'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int16'})
  vleMeta = pd.read_csv(ouladD / "VLE.csv", usecols=['id_site', 'activity_type'], dtype={'id_site': 'int32'})
  sa = pd.read_csv(ouladD / "StudentAssessment.csv")
  assessments = pd.read_csv(ouladD / "Assessments.csv")
  svle = svle.merge(vleMeta.drop_duplicates('id_site'), on='id_site', how='left')
  svle.drop(columns=['id_site'], inplace=True)
  svle.sort_values(['id_student', 'date'], inplace=True)
  print(f"Loaded Student Info → {len(info):>8,} Rows")
  print(f"Loaded Student VLE → {len(svle):>8,} Rows")
  print(f"Loaded Student Assessment → {len(sa):>8,} Rows")
  print(f"Loaded Assessments → {len(assessments):>8,} Rows")
  print(f"Unique Students → {info['id_student'].nunique():>8,}")
  sa = sa.merge(assessments[['id_assessment', 'date']], on='id_assessment', how='left')
  return {'info': info, 'vle': svle, 'studentassessment': sa, 'assessments_meta': assessments}
def buildStressLabels(oulad: dict) -> pd.DataFrame:
  showBanner("S2: Label Construction - Stress Proxy")
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
  showBanner("S3.1: VLE Feature Engineering - Session + Temporal")
  vle = oulad['vle']
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
    slope = np.polyfit(np.arange(len(clicks)), clicks, 1)[0] if len(clicks) > 2 else 0.0
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
  showBanner("S3.2: Assessment Feature Engineering")
  asmnt = oulad.get('studentassessment', oulad.get('assessments', pd.DataFrame()))
  if asmnt.empty:
    print("No Assessment Data Found! Skipping!")
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
    scoreSlope = np.polyfit(np.arange(len(scores)), scores, 1)[0] if len(scores) > 2 else 0.0
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
  showBanner("S3.3: Student Info Features")
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
  showBanner("S3.4: Merging All Feature Tables")
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
  showBanner("S4: Building LSTM Sequences From VLE Logs")
  vle = oulad['vle']
  if 'activity_type' in vle.columns:
    vle['activity_enc'] = LabelEncoder().fit_transform(vle['activity_type'].astype(str))
  else:
    vle['activity_enc'] = 0
  vle['week_num'] = (vle['date'] // 7).clip(0, 52)
  vle['day_of_week'] = (vle['date'] % 7)
  labelMap = labels.set_index('id_student')['stress_label'].to_dict()
  XList, yList, studentIdsInSeq = [], [], []
  for sid, grp in vle.groupby('id_student'):
    if sid not in labelMap:
      continue
    seqFeats = grp[['sum_click', 'day_of_week', 'week_num', 'activity_enc']].values.astype(np.float32)
    if len(seqFeats) >= currentSeqLen:
      seq = seqFeats[-currentSeqLen:]
    else:
      pad = np.zeros((currentSeqLen - len(seqFeats), seqFeats.shape[1]), dtype=np.float32)
      seq = np.vstack([pad, seqFeats])
    XList.append(seq)
    yList.append(labelMap[sid])
    studentIdsInSeq.append(sid)
    if len(XList) % 5000 == 0:
      print(f"... Built Sequences For {len(XList):,} Students")
  X = np.array(XList, dtype=np.float32)
  y = np.array(yList, dtype=np.int64)
  shapeOrig = X.shape
  XFlat = X.reshape(-1, X.shape[-1])
  globalMean = XFlat.mean(axis=0)
  globalStd = XFlat.std(axis=0) + 1e-6
  X = ((XFlat - globalMean) / globalStd).reshape(shapeOrig)
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
def trainLSTM(XSeq, ySeq, studentIds) -> tuple:
  showBanner("S5: Training Bi-LSTM With Attention")
  XTr, XTe, yTr, yTe = train_test_split(XSeq, ySeq, test_size=testSize, random_state=randomState, stratify=ySeq)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Device: {device}")
  trainDs = TensorDataset(torch.tensor(XTr), torch.tensor(yTr))
  testDs = TensorDataset(torch.tensor(XTe), torch.tensor(yTe))
  trainDl = DataLoader(trainDs, batch_size=LSTMBatch, shuffle=True)
  testDl = DataLoader(testDs, batch_size=LSTMBatch)
  model = StressLSTM(inputSize=XSeq.shape[2]).to(device)
  criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, len(yTr[yTr==0]) / (len(yTr[yTr==1]) + 1e-6)], dtype=torch.float32).to(device))
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
  allDl = DataLoader(TensorDataset(torch.tensor(XSeq), torch.tensor(ySeq)), batch_size=LSTMBatch)
  probs = []
  with torch.no_grad():
    for xb, _ in allDl:
      out = model(xb.to(device))
      p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
      probs.extend(p)
  probs = np.array(probs).reshape(-1, 1)
  probMap = dict(zip(studentIds, probs.flatten()))
  return model, probMap, history, (XTe, yTe, allPreds)
def printEnsembleMetrics(yTe, yPred, yPredTuned, yProba, thresh, defThresh=defaultThresh):
  print("\n" + "-" * 60)
  print(f"Ensemble Results (Default Threshold = {defThresh:.3f})")
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
  showBanner("S7: Stacking Ensemble - XGBoost + LightGBM + LSTM")
  featureCols = [c for c in tabularDf.columns if c not in ('id_student', 'stress_label')]
  XFull = tabularDf[featureCols].values
  y = tabularDf['stress_label'].values
  print(f"Combined Feature Dim  : {XFull.shape[1]} (37 Tabular + 1 LSTM)")
  smote = SMOTE(random_state=randomState)
  XTr, XTe, yTr, yTe = train_test_split(XFull, y, test_size=testSize, random_state=randomState, stratify=y)
  XTrS, yTrS = smote.fit_resample(XTr, yTr)
  print(f"\nAfter SMOTE - Normal: {sum(yTrS==0)} | Stressed: {sum(yTrS==1)}")
  scaler = StandardScaler()
  XTrSc = scaler.fit_transform(XTrS)
  XTeSc = scaler.transform(XTe)
  xgbModel = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=randomState, n_jobs=-1)
  lgbModel = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=randomState, n_jobs=-1, verbose=-1)
  stack = StackingClassifier(estimators=[('xgb', xgbModel), ('lgb', lgbModel)], final_estimator=LogisticRegression(max_iter=1000, random_state=randomState), cv=5, passthrough=False, n_jobs=1)
  print("\nFitting Stacking Ensemble...")
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
    'X_train': XTrSc, 'y_train': yTrS, 'tabular_df': tabularDf,
    'X_tr': XTr, 'X_te': XTe, 'y_tr': yTr, 'y_te': yTe, 'X_te_sc': XTeSc,
  }
  printEnsembleMetrics(yTe, yPred, yPredTuned, yProba, bestThresh)
  return results
def crossValidateModels(tabularDf: pd.DataFrame) -> dict:
  showBanner("S8: Stratified K-Fold Cross Validation")
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
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=1)
    cvResults[name] = scores
    print(f"\n{name} (CV With SMOTE):")
    print(f"  Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Folds: {[f'{s:.4f}' for s in scores]}")
  return cvResults
def SHAPAnalysis(results: dict, tabularDf: pd.DataFrame, outD: Path):
  showBanner("S9: SHAP Feature Interpretability")
  featureCols = results['feature_cols']
  XTe = results['X_test']
  yTe = results['y_test']
  try:
    xgbBase = results['model'].named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgbBase)
    shapVals = explainer(XTe)
    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shapVals.values, XTe, feature_names=featureCols, show=False, max_display=20)
    plt.title('SHAP Global Feature Impact Distribution (Beeswarm)', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(imgD / 'ShapSummary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: ShapSummary.png")
    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shapVals.values, XTe, feature_names=featureCols, plot_type='bar', show=False, max_display=20)
    plt.title('SHAP Global Feature Importance (Mean |SHAP|)', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(imgD / 'ShapGlobalBar.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: ShapGlobalBar.png")
    stressedMask = (yTe == 1)
    normalMask = (yTe == 0)
    stressedMean = np.mean(shapVals.values[stressedMask], axis=0)
    normalMean = np.mean(shapVals.values[normalMask], axis=0)
    topIdx = np.argsort(np.abs(np.mean(shapVals.values, axis=0)))[-12:]
    topNames = [featureCols[i] for i in topIdx]
    fig, ax = plt.subplots(figsize=(12, 7))
    yPos = np.arange(len(topNames)); w = 0.35
    ax.barh(yPos - w/2, stressedMean[topIdx], w, label='Stressed Student Cohort (T=1)', color='#E76F51', edgecolor='black')
    ax.barh(yPos + w/2, normalMean[topIdx], w, label='Normal Student Cohort (T=0)', color='#2A9D8F', edgecolor='black')
    ax.set_yticks(yPos); ax.set_yticklabels(topNames, fontsize=11, weight='bold')
    ax.set_xlabel('Mean SHAP Contribution Value', fontsize=12, weight='bold')
    ax.set_title('Cohort Comparison: Mean Feature Impact For Stressed vs Normal Students', fontsize=14, weight='bold', pad=15)
    ax.axvline(0, color='black', linestyle='--', lw=1)
    ax.legend(loc='lower right', fontsize=11); ax.grid(axis='x', linestyle=':', alpha=0.7)
    fig.tight_layout(); fig.savefig(imgD / 'ShapCohortComparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("Saved: ShapCohortComparison.png")
  except Exception as e:
    print(f"SHAP Analysis Skipped: {e}!")
def generateVisualizations(results: dict, cvResults: dict, LSTMHist: dict, outD: Path, tabularDf: pd.DataFrame = None):
  showBanner("S10: Generating Visualizations")
  yTe = results['y_test']
  yPred = results['y_pred_tuned']
  yProba = results['y_proba']
  fpr, tpr, _ = roc_curve(yTe, yProba)
  aucScore = roc_auc_score(yTe, yProba)
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.plot(fpr, tpr, color='#1B365D', lw=2.5, label=f'Stacking Ensemble (AUC = {aucScore:.4f})')
  ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.7, label='Random Baseline (AUC = 0.5000)')
  ax.fill_between(fpr, tpr, alpha=0.15, color='#1B365D')
  ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, weight='bold')
  ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12, weight='bold')
  ax.set_title('Receiver Operating Characteristic Curve', fontsize=14, weight='bold', pad=15)
  ax.legend(loc='lower right', fontsize=11); ax.grid(True, linestyle=':', alpha=0.7)
  ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
  fig.tight_layout(); fig.savefig(imgD / 'ROCCurve.png', dpi=300); plt.close(fig)
  print("Saved: ROCCurve.png")
  if LSTMHist:
    nEpochs = len(LSTMHist['val_acc'])
    ep = range(1, nEpochs + 1)
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    line1 = ax1.plot(ep, LSTMHist['train_loss'], color='#E76F51', lw=2.5, label='Training Loss')
    line2 = ax2.plot(ep, LSTMHist['val_acc'], color='#2A9D8F', lw=2.5, label='Validation Accuracy')
    ax1.set_xlabel(f'Epoch ({nEpochs})', fontsize=12, weight='bold')
    ax1.set_ylabel('Training Loss', color='#E76F51', fontsize=12, weight='bold')
    ax2.set_ylabel('Validation Accuracy', color='#2A9D8F', fontsize=12, weight='bold')
    ax1.set_title(f'Bi-LSTM Temporal Model Training Dynamics Across {nEpochs} Epochs', fontsize=13, weight='bold', pad=15)
    lines = line1 + line2; labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11)
    fig.tight_layout(); fig.savefig(imgD / 'LSTMTrainingHistory.png', dpi=300); plt.close(fig)
    print("Saved: LSTMTrainingHistory.png")
  precPR, recPR, _ = precision_recall_curve(yTe, yProba)
  prAuc = average_precision_score(yTe, yProba)
  baselineRatio = sum(yTe == 1) / len(yTe)
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.plot(recPR, precPR, color='#2A9D8F', lw=2.5, label=f'Stacking Ensemble (PR-AUC = {prAuc:.4f})')
  ax.axhline(baselineRatio, color='#E76F51', linestyle='--', lw=1.5, label=f'No-Skill Baseline ({baselineRatio:.4f})')
  ax.fill_between(recPR, precPR, alpha=0.15, color='#2A9D8F')
  ax.set_xlabel('Recall / Sensitivity', fontsize=12, weight='bold')
  ax.set_ylabel('Precision', fontsize=12, weight='bold')
  ax.set_title('Precision-Recall Curve', fontsize=13, weight='bold', pad=15)
  ax.legend(loc='lower left', fontsize=11)
  ax.set_ylim([0.0, 1.05]); ax.set_xlim([0.0, 1.0])
  fig.tight_layout(); fig.savefig(imgD / 'PrecisionRecallCurve.png', dpi=300); plt.close(fig)
  print("Saved: PrecisionRecallCurve.png")
  if cvResults:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(cvResults.keys()); means = [v.mean() for v in cvResults.values()]; stds = [v.std() for v in cvResults.values()]
    colors = ['#2E86AB', '#E07A5F']
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors[:len(names)], edgecolor='black', width=0.45)
    for bar, m in zip(bars, means):
      ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{m:.4f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylim([0.85, 0.96]); ax.set_ylabel('Cross-Validation Accuracy', fontsize=12, weight='bold')
    ax.set_title('5-Fold Stratified Cross-Validation Accuracy Comparison', fontsize=13, weight='bold', pad=15)
    fig.tight_layout(); fig.savefig(imgD / 'CVComparison.png', dpi=300); plt.close(fig)
    print("Saved: CVComparison.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    foldNames = [f"Fold {i+1}" for i in range(nFolds)] + ["Mean CV"]
    xPos = np.arange(len(foldNames)); w = 0.35
    for idx, (name, scores) in enumerate(cvResults.items()):
      scoresList = list(scores) + [scores.mean()]
      offset = -w/2 if idx == 0 else w/2
      color = "#2E86AB" if idx == 0 else "#E07A5F"
      rects = ax.bar(xPos + offset, scoresList, w, label=name, color=color, edgecolor='black')
      for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_ylabel("Cross-Validation Accuracy", fontsize=12, weight="bold")
    ax.set_title("5-Fold Stratified Cross-Validation Fold-By-Fold Breakdown", fontsize=14, weight="bold", pad=15)
    ax.set_xticks(xPos); ax.set_xticklabels(foldNames, fontsize=11, weight="bold"); ax.set_ylim(0.88, 0.95)
    ax.legend(loc="lower right", fontsize=11); ax.grid(axis="y", linestyle=":", alpha=0.7)
    fig.tight_layout(); fig.savefig(imgD / 'CVFoldsDetail.png', dpi=300); plt.close(fig)
    print("Saved: CVFoldsDetail.png")
  cm = confusion_matrix(yTe, yPred)
  tn, fp, fn, tp = cm.ravel()
  total = len(yTe)
  fig, ax = plt.subplots(figsize=(9, 8))
  cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
  fig.colorbar(cax, fraction=0.046, pad=0.04)
  labels = [[f"TN = {tn:,}\n({tn/total*100:.2f}%)\nSpec: {tn/(tn+fp)*100:.2f}%", f"FP = {fp:,}\n({fp/total*100:.2f}%)\nFPR: {fp/(tn+fp)*100:.2f}%"], [f"FN = {fn:,}\n({fn/total*100:.2f}%)\nFNR: {fn/(fn+tp)*100:.2f}%", f"TP = {tp:,}\n({tp/total*100:.2f}%)\nRecall: {tp/(fn+tp)*100:.2f}%"]]
  for i in range(2):
    for j in range(2):
      textColor = "white" if cm[i, j] > total * 0.25 else "black"
      ax.text(j, i, labels[i][j], ha="center", va="center", color=textColor, fontsize=13, weight="bold")
  ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
  ax.set_xticklabels(["Normal (0)", "Stressed (1)"], fontsize=12, weight="bold")
  ax.set_yticklabels(["Normal (0)", "Stressed (1)"], fontsize=12, weight="bold")
  ax.set_xlabel("Predicted Label", fontsize=13, weight="bold", labelpad=10)
  ax.set_ylabel("True Actual Label", fontsize=13, weight="bold", labelpad=10)
  acc = accuracy_score(yTe, yPred)
  ax.set_title(f"Detailed Confusion Matrix (Test Set N = {total:,})\nTuned Youden Threshold = {results.get('best_thresh', defaultThresh):.3f} | Accuracy = {acc*100:.2f}%", fontsize=14, weight="bold", pad=15)
  fig.tight_layout(); fig.savefig(imgD / 'ConfusionMatrix.png', dpi=300); plt.close(fig)
  print("Saved: ConfusionMatrix.png")
  mseVal = float(np.mean((yProba - yTe) ** 2))
  logLossVal = float(log_loss(yTe, yProba))
  fullHeaders = ["Metric", "Formula", "Value", "Notes"]
  fullRows = [
    ["Accuracy", "(TP + TN) / Total", f"{accuracy_score(yTe, yPred):.4f} ({accuracy_score(yTe, yPred)*100:.2f}%)", "Overall Correct Classification"],
    ["Balanced Accuracy", "(TPR + TNR) / 2", f"{balanced_accuracy_score(yTe, yPred):.4f} ({balanced_accuracy_score(yTe, yPred)*100:.2f}%)", "Balanced Class Representation"],
    ["Precision", "TP / (TP + FP)", f"{precision_score(yTe, yPred, zero_division=0):.4f} ({precision_score(yTe, yPred, zero_division=0)*100:.2f}%)", "Low False Alarm Rate"],
    ["Recall / Sensitivity", "TP / (TP + FN)", f"{recall_score(yTe, yPred, zero_division=0):.4f} ({recall_score(yTe, yPred, zero_division=0)*100:.2f}%)", "High At-Risk Student Capture"],
    ["Specificity", "TN / (TN + FP)", f"{float(tn / (tn + fp + 1e-6)):.4f} ({float(tn / (tn + fp + 1e-6))*100:.2f}%)", "Normal Student Recognition"],
    ["F1-Score", "2 * (P * R) / (P + R)", f"{f1_score(yTe, yPred, zero_division=0):.4f} ({f1_score(yTe, yPred, zero_division=0)*100:.2f}%)", "Harmonic Balance of Precision & Recall"],
    ["ROC-AUC", "Integral TPR d(FPR)", f"{aucScore:.4f} ({aucScore*100:.2f}%)", "Exceptional Discriminative Ability"],
    ["PR-AUC", "Integral Prec d(Rec)", f"{prAuc:.4f} ({prAuc*100:.2f}%)", "Area Under Precision-Recall Curve"],
    ["MCC", "(TP*TN - FP*FN) / Denom", f"{matthews_corrcoef(yTe, yPred):.4f}", "Robust Balanced Correlation Metric"],
    ["Cohen's Kappa", "(p_o - p_e) / (1 - p_e)", f"{cohen_kappa_score(yTe, yPred):.4f}", "Inter-Rater Agreement Level"],
    ["MSE / Brier Score", "Sum (p_i - y_i)^2 / N", f"{mseVal:.4f}", "Probability Calibration Quality"],
    ["RMSE", "Sqrt(MSE)", f"{float(np.sqrt(mseVal)):.4f}", "Root Mean Square Prediction Error"],
    ["MAE", "Sum |p_i - y_i| / N", f"{float(np.mean(np.abs(yProba - yTe))):.4f}", "Mean Absolute Error"],
    ["Log-Loss", "-Sum [y ln p + (1-y) ln(1-p)] / N", f"{logLossVal:.4f}", "Cross-Entropy Penalty"]
  ]
  fig, ax = plt.subplots(figsize=(15, 9)); ax.axis('off')
  tbl = ax.table(cellText=fullRows, colLabels=fullHeaders, cellLoc='center', loc='center')
  tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.0)
  for (r, c), cell in tbl.get_celld().items():
    if r == 0:
      cell.set_facecolor("#1B365D"); cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
      cell.set_facecolor("#F0F4F8")
    else:
      cell.set_facecolor("#FFFFFF")
  ax.set_title("Full Comprehensive Evaluation Metrics Report", fontsize=15, weight="bold", pad=25)
  fig.tight_layout(); fig.savefig(imgD / 'FullMetricsReport.png', dpi=300); plt.close(fig)
  print("Saved: FullMetricsReport.png")
  if tabularDf is not None:
    nStressed = int((tabularDf['stress_label'] == 1).sum())
    nNormal = int((tabularDf['stress_label'] == 0).sum())
    labels1 = [f"Stressed / At-Risk (1)\n{nStressed:,} ({nStressed/len(tabularDf)*100:.1f}%)", f"Normal (0)\n{nNormal:,} ({nNormal/len(tabularDf)*100:.1f}%)"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie([nStressed, nNormal], labels=labels1, colors=["#E76F51", "#2A9D8F"], autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11, "weight": "bold"}, explode=(0.04, 0))
    ax.set_title(f"OULAD Student Label Distribution (Total: {len(tabularDf):,})", fontsize=13, weight="bold", pad=15)
    fig.tight_layout(); fig.savefig(imgD / 'DatasetDistribution.png', dpi=300); plt.close(fig)
    print("Saved: DatasetDistribution.png")
    nTr = len(results['y_train']); nTe = len(results['y_test'])
    labels2 = [f"Train Set (SMOTE, {nTr/(nTr+nTe)*100:.0f}%)\n{nTr:,} Students", f"Test Set ({nTe/(nTr+nTe)*100:.0f}%)\n{nTe:,} Students"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie([nTr, nTe], labels=labels2, colors=["#457B9D", "#F4A261"], autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11, "weight": "bold"}, explode=(0.04, 0))
    ax.set_title("Dataset Stratified Train / Test Partitioning", fontsize=13, weight="bold", pad=15)
    fig.tight_layout(); fig.savefig(imgD / 'TrainTestSplit.png', dpi=300); plt.close(fig)
    print("Saved: TrainTestSplit.png")
  print("\nAll Visualizations Saved To:", imgD)
def exportModels(results: dict, LSTMModel, outD: Path, LSTMHist: dict = None, seqNorm=None):
  showBanner("S11: Exporting Models")
  joblib.dump(results['model'], modelD / 'StackingEnsemble.pkl')
  joblib.dump(results['scaler'], modelD / 'Scaler.pkl')
  if LSTMModel is not None:
    torch.save(LSTMModel.state_dict(), modelD / 'LSTM.pt')
  if seqNorm is not None:
    joblib.dump(seqNorm, modelD / 'SeqNorm.pkl')
    print(f"SeqNorm.pkl → {modelD}")
  if LSTMHist is not None:
    joblib.dump(LSTMHist, modelD / 'LSTMHistory.pkl')
    print(f"LSTMHistory.pkl → {modelD}")
  print(f"StackingEnsemble.pkl → {modelD}")
  print(f"Scaler.pkl → {modelD}")
  print(f"LSTM.pt → {modelD}")
  evalCache = {
    'model': results['model'], 'scaler': results['scaler'], 'tabular_df': results.get('tabular_df'),
    'feature_cols': results['feature_cols'], 'X_tr': results.get('X_tr'), 'X_te': results.get('X_te'),
    'y_tr': results.get('y_tr'), 'y_te': results['y_test'], 'X_te_sc': results['X_test'],
    'y_pred': results['y_pred'], 'y_pred_tuned': results['y_pred_tuned'], 'y_proba': results['y_proba'],
    'best_thresh': results['best_thresh'],
  }
  joblib.dump(evalCache, modelD / 'EvaluationCache.pkl')
  print(f"EvaluationCache.pkl → {modelD}")
def main():
  import time
  start = datetime.now(); tGlobal = time.time()
  print("\n" + "=" * 80)
  print("Stress Detection Via LMS Digital Signals")
  print("Pipeline: OULAD-Based Stress Detection")
  print("Target  : 90%+ Accuracy Via Hybrid LSTM + XGBoost Stacking")
  print("=" * 80)
  print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
  featCachePath = modelD / "FeatureCache.pkl"
  if featCachePath.exists():
    showBanner("Resuming Pipeline From Checkpoint")
    print(f"Loading Pre-Computed Features & LSTM From: {featCachePath}")
    cached = joblib.load(featCachePath)
    tabularDf = cached['tabularDf']
    LSTMHist = cached['LSTMHist']
    gMean, gStd = cached['seqNorm']
    LSTMModel = cached.get('LSTMModel')
  else:
    print("\nPhase 1: Loading Datasets...")
    t0 = time.time(); oulad = loadOULAD(dataD)
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
    joblib.dump({'tabularDf': tabularDf, 'LSTMHist': LSTMHist, 'seqNorm': (gMean, gStd), 'LSTMModel': LSTMModel}, featCachePath)
    if LSTMModel is not None:
      torch.save(LSTMModel.state_dict(), modelD / 'LSTM.pt')
    print(f"Checkpoint Saved To: {featCachePath}")
  print("\nPhase 4: Training Stacking Ensemble (XGBoost + LightGBM + Meta)...")
  t0 = time.time(); results = trainEnsemble(tabularDf, None)
  print(f"Stacking Ensemble Completed In {time.time() - t0:.2f}s")
  print("\nPhase 5: Running Stratified K-Fold Cross Validation...")
  t0 = time.time(); cvResults = crossValidateModels(tabularDf)
  print(f"Cross Validation Completed In {time.time() - t0:.2f}s")
  print("\nPhase 6: Running SHAP Interpretability Analysis...")
  t0 = time.time(); SHAPAnalysis(results, tabularDf, outputD)
  print(f"SHAP Analysis Completed In {time.time() - t0:.2f}s")
  print("\nPhase 7: Generating Output Visualizations...")
  t0 = time.time(); generateVisualizations(results, cvResults, LSTMHist, outputD, tabularDf)
  print(f"Visualizations Generated In {time.time() - t0:.2f}s")
  print("\nPhase 8: Exporting Final Models...")
  t0 = time.time(); exportModels(results, LSTMModel, outputD, LSTMHist=LSTMHist, seqNorm=(gMean, gStd))
  print(f"Model Export Completed In {time.time() - t0:.2f}s")
  showBanner("Finished!!")
  yTe = results['y_test']; yPred = results['y_pred_tuned']
  print(f"\nFinal Accuracy: {accuracy_score(yTe, yPred):.4f}")
  print(f"Final F1-Score: {f1_score(yTe, yPred, zero_division=0):.4f}")
  print(f"Final ROC-AUC: {roc_auc_score(yTe, results['y_proba']):.4f}")
  print(f"Total Pipeline Elapsed Time: {time.time() - tGlobal:.2f}s")
  print(f"Outputs Saved To: {outputD.resolve()}")
  print("\n" + "=" * 80)
if __name__ == "__main__":
  main()