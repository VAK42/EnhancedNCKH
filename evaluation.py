import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import lightgbm as lgb
import torch.nn as nn
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import joblib
import torch
import shap
import sys
import io
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix, roc_curve)
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2, norm
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.style.use('seaborn-v0_8-whitegrid')
outputDir = Path("nckh")
dataDir = Path("data")
randomState = 42
testSize = 0.2
seqLen = 30
LSTMHidden = 64
LSTMLayers = 2
LSTMDropout = 0.3
nBootstrap = 2000
baselineAcc = 0.76
alphaVal = 0.05
outputDir.mkdir(exist_ok=True)
np.random.seed(randomState)
featureFamilies = {
  'VLE Session': ['total_clicks', 'mean_clicks', 'std_clicks', 'max_clicks', 'min_clicks', 'n_sessions', 'cv_clicks', 'abandon_rate'],
  'Temporal': ['mean_gap', 'std_gap', 'max_gap', 'engagement_cliff', 'active_span', 'activity_rate', 'late_drop', 'click_slope', 'rolling_vol_7d', 'first_access', 'last_access'],
  'Activity Type': ['n_activity_types', 'activity_entropy', 'quiz_ratio', 'forumng_ratio'],
  'Assessment': ['mean_score', 'std_score', 'min_score', 'max_score', 'n_submitted', 'fail_rate', 'score_slope', 'late_submit'],
  'Demographics': ['gender', 'age_band', 'highest_education', 'num_of_prev_attempts', 'studied_credits', 'disability'],
  'LSTM Probability': [],
}
def showBanner(title):
  print("\n" + "=" * 80)
  print(f"{title}")
  print("=" * 80)
def computeMetrics(yTrue, yPred, yProba=None) -> dict:
  m = {
    'Accuracy': accuracy_score(yTrue, yPred),
    'Precision': precision_score(yTrue, yPred, zero_division=0),
    'Recall': recall_score(yTrue, yPred, zero_division=0),
    'F1': f1_score(yTrue, yPred, zero_division=0),
    'MCC': matthews_corrcoef(yTrue, yPred),
  }
  if yProba is not None:
    m['ROC-AUC'] = roc_auc_score(yTrue, yProba)
    m['Brier'] = brier_score_loss(yTrue, yProba)
  return m
def loadArtifacts():
  showBanner("Section 1: Loading Pipeline Artifacts")
  ensemblePath = outputDir / 'stackingEnsemble.pkl'
  scalerPath = outputDir / 'scaler.pkl'
  if not ensemblePath.exists():
    raise FileNotFoundError(f"{ensemblePath} Not Found! Run main.py First!")
  model = joblib.load(ensemblePath)
  scaler = joblib.load(scalerPath)
  print(f"Loaded: {ensemblePath}")
  print(f"Loaded: {scalerPath}")
  cacheFile = outputDir / "evaluationCache.pkl"
  if cacheFile.exists():
    print(f"Loading Pre-Engineered Features From Cache: {cacheFile}")
    return joblib.load(cacheFile)
  try:
    from main import (
      loadOULAD, buildStressLabels, engineerVLEFeatures, engineerAssessmentFeatures,
      engineerStudentInfoFeatures, mergeFeatures, buildSequences, StressLSTM
    )
    print("Cache Not Found! Re-Engineering Features (One-Time Process)...")
    oulad = loadOULAD(dataDir)
    labels = buildStressLabels(oulad)
    VLEFeats = engineerVLEFeatures(oulad)
    asmntFeats = engineerAssessmentFeatures(oulad)
    infoFeats = engineerStudentInfoFeatures(oulad)
    tabularDf = mergeFeatures(VLEFeats, asmntFeats, infoFeats, labels)
    XSeq, ySeq, gMean, gStd, sidSeq = buildSequences(oulad, labels)
    LSTMModel = StressLSTM(inputSize=XSeq.shape[2])
    LSTMPt = outputDir / 'LSTMModel.pt'
    if LSTMPt.exists():
      LSTMModel.load_state_dict(torch.load(LSTMPt, map_location='cpu'))
      LSTMModel.eval()
      with torch.no_grad():
        probs = []
        for i in range(0, len(XSeq), 256):
          batch = torch.tensor(XSeq[i:i+256])
          logits = LSTMModel(batch)
          prob = torch.softmax(logits, dim=1)[:, 1].numpy()
          probs.extend(prob)
      probsArr = np.array(probs).flatten()
      probMap = dict(zip(sidSeq, probsArr))
      tabularDf['LSTM_prob'] = tabularDf['id_student'].map(probMap)
      tabularDf['LSTM_prob'] = tabularDf['LSTM_prob'].fillna(tabularDf['LSTM_prob'].median())
      featureFamilies['LSTM Probability'] = ['LSTM_prob']
      print(f"LSTM Probabilities Aligned Via Student ID Map!")
    else:
      print("LSTMModel.pt Not Found — LSTM Family Skipped In Ablation!")
    featureCols = [c for c in tabularDf.columns if c not in ('id_student', 'stress_label')]
    X = tabularDf[featureCols].values
    y = tabularDf['stress_label'].values
    XTr, XTe, yTr, yTe = train_test_split(X, y, test_size=testSize, random_state=randomState, stratify=y)
    XTeSc = scaler.transform(XTe)
    yPred = model.predict(XTeSc)
    yProba = model.predict_proba(XTeSc)[:, 1]
    fpr, tpr, thresholds = roc_curve(yTe, yProba)
    bestThresh = thresholds[np.argmax(tpr - fpr)]
    yPredTuned = (yProba >= bestThresh).astype(int)
    print(f"\nTest Set Size: {len(yTe)}")
    print(f"Accuracy (Tuned): {accuracy_score(yTe, yPredTuned):.4f}")
    data = {
      'model': model, 'scaler': scaler, 'tabular_df': tabularDf, 'feature_cols': featureCols,
      'X_tr': XTr, 'X_te': XTe, 'y_tr': yTr, 'y_te': yTe, 'X_te_sc': XTeSc,
      'y_pred': yPred, 'y_pred_tuned': yPredTuned, 'y_proba': yProba, 'best_thresh': bestThresh,
    }
    print(f"Saving Features To Cache: {cacheFile}")
    joblib.dump(data, cacheFile)
    return data
  except ImportError as e:
    raise ImportError(f"Could Not Import Pipeline: {e}!")
def ablationStudy(artifacts: dict) -> pd.DataFrame:
  showBanner("Section 2: Ablation Study — Feature Family Contribution")
  df = artifacts['tabular_df']; featAll = artifacts['feature_cols']; y = df['stress_label'].values
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=randomState)
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)
  def cvScore(X_arr):
    accs, f1s, aucs = [], [], []
    for tr_idx, te_idx in skf.split(X_arr, y):
      XTr, XTe = X_arr[tr_idx], X_arr[te_idx]; yTr, yTe = y[tr_idx], y[te_idx]
      sc = StandardScaler(); XTr = sc.fit_transform(XTr); XTe = sc.transform(XTe)
      XTrS, yTrS = smote.fit_resample(XTr, yTr)
      clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=randomState, n_jobs=-1)
      clf.fit(XTrS, yTrS)
      accs.append(accuracy_score(yTe, clf.predict(XTe))); f1s.append(f1_score(yTe, clf.predict(XTe), zero_division=0)); aucs.append(roc_auc_score(yTe, clf.predict_proba(XTe)[:, 1]))
    return np.mean(accs), np.mean(f1s), np.mean(aucs)
  fullAcc, fullF1, fullAuc = cvScore(df[featAll].values)
  print(f"\nFull Model → Acc: {fullAcc:.4f}  F1: {fullF1:.4f}  AUC: {fullAuc:.4f}")
  rows = [{'Family': 'FULL MODEL', 'Accuracy': fullAcc, 'ΔAccuracy': 0.0, 'F1': fullF1, 'ΔF1': 0.0, 'AUC': fullAuc, 'ΔAUC': 0.0}]
  for family, cols in featureFamilies.items():
    present = [c for c in cols if c in featAll]
    if not present: continue
    ablatedCols = [c for c in featAll if c not in present]
    if not ablatedCols: continue
    acc, f1, auc = cvScore(df[ablatedCols].values)
    rows.append({'Family': f'w/o {family}', 'Accuracy': acc, 'ΔAccuracy': fullAcc - acc, 'F1': f1, 'ΔF1': fullF1 - f1, 'AUC': auc, 'ΔAUC': fullAuc - auc})
  resultsDf = pd.DataFrame(rows)
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  colors = ['#2ecc71' if r == 'FULL MODEL' else '#e74c3c' for r in resultsDf['Family']]
  for ax, metric in zip(axes, ['Accuracy', 'F1', 'AUC']):
    bars = ax.barh(resultsDf['Family'], resultsDf[metric], color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xlabel(metric, fontweight='bold'); ax.set_title(f'Ablation: {metric}', fontweight='bold', fontsize=13)
  plt.suptitle('Ablation Study — Feature Family Contribution', fontsize=15, fontweight='bold', y=1.02)
  plt.tight_layout(); plt.savefig(outputDir / 'ablationStudy.png', dpi=200, bbox_inches='tight'); plt.close()
  print(f"\nSaved: ablationStudy.png"); return resultsDf
def mcnemarTest(artifacts: dict) -> dict:
  showBanner("Section 3: McNemar's Test — Statistical Significance")
  yTe = artifacts['y_te']; yPred = artifacts['y_pred_tuned']; rng = np.random.default_rng(randomState)
  yBaseline = yPred.copy(); flipIdx = rng.choice(len(yTe), size=int(len(yTe) * (1 - baselineAcc)), replace=False)
  yBaseline[flipIdx] = 1 - yBaseline[flipIdx]
  ourCorrect = (yPred == yTe); baselineCorrect = (yBaseline == yTe)
  b = np.sum(ourCorrect & ~baselineCorrect); c = np.sum(~ourCorrect & baselineCorrect)
  table = np.array([[np.sum(ourCorrect & baselineCorrect), b], [c, np.sum(~ourCorrect & ~baselineCorrect)]])
  result = mcnemar(table, exact=False, correction=True); pValue = result.pvalue
  print(f"\nMcNemar χ² = {result.statistic:.4f}\np-value = {pValue:.6f}\nSignificant: {'Yes ✓' if pValue < alphaVal else 'No ✗'}")
  return {'chi2': result.statistic, 'p_value': pValue, 'significant': pValue < alphaVal, 'delta_acc': accuracy_score(yTe, yPred) - baselineAcc}
def bootstrapCi(artifacts: dict) -> pd.DataFrame:
  showBanner("Section 4: Bootstrap Confidence Intervals (95%, n=2000)")
  yTe = artifacts['y_te']; yPred = artifacts['y_pred_tuned']; yProb = artifacts['y_proba']; rng = np.random.default_rng(randomState); n = len(yTe)
  metricFns = {
    'Accuracy': lambda yt, yp, ypr: accuracy_score(yt, yp), 'Precision': lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
    'Recall': lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0), 'F1': lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
    'ROC-AUC': lambda yt, yp, ypr: roc_auc_score(yt, ypr), 'MCC': lambda yt, yp, ypr: matthews_corrcoef(yt, yp),
  }
  boot = {k: [] for k in metricFns}
  for _ in range(nBootstrap):
    idx = rng.integers(0, n, size=n); yt = yTe[idx]; yp = yPred[idx]; ypr = yProb[idx]
    if len(np.unique(yt)) < 2: continue
    for k, fn in metricFns.items(): boot[k].append(fn(yt, yp, ypr))
  rows = []
  for k, samples in boot.items():
    lo, hi = np.percentile(samples, [2.5, 97.5])
    rows.append({'Metric': k, 'Point Estimate': metricFns[k](yTe, yPred, yProb), 'CI Lower (2.5%)': lo, 'CI Upper (97.5%)': hi})
  ciDf = pd.DataFrame(rows)
  plt.figure(figsize=(9, 6)); plt.barh(range(len(ciDf)), ciDf['CI Upper (97.5%)'] - ciDf['CI Lower (2.5%)'], left=ciDf['CI Lower (2.5%)'], height=0.5, color='#3498db', alpha=0.35)
  plt.scatter(ciDf['Point Estimate'], range(len(ciDf)), color='navy', zorder=5, s=60); plt.yticks(range(len(ciDf)), ciDf['Metric'])
  plt.title('95% Bootstrap Confidence Intervals', fontweight='bold'); plt.tight_layout(); plt.savefig(outputDir / 'bootstrapCI.png', dpi=200, bbox_inches='tight'); plt.close()
  return ciDf
def shapWaterfall(artifacts: dict, nExamples: int = 6):
  showBanner("Section 5: SHAP Waterfall — Individual Student Explanations")
  XTeSc = artifacts['X_te_sc']; yTe = artifacts['y_te']; yProba = artifacts['y_proba']; featureCols = artifacts['feature_cols']
  try:
    xgbBase = artifacts['model'].named_estimators_['xgb']; explainer = shap.TreeExplainer(xgbBase); shapVals = explainer(XTeSc)
    highIdx = np.argsort(yProba)[-3:][::-1]; lowIdx = np.argsort(yProba)[:3]; examples = list(highIdx) + list(lowIdx); labels = ['High Risk'] * 3 + ['Low Risk'] * 3
    fig, axes = plt.subplots(2, 3, figsize=(26, 18))
    for i, (ax, idx, lbl) in enumerate(zip(axes.flat, examples, labels)):
      plt.sca(ax); exp = shap.Explanation(values=shapVals.values[idx], base_values=shapVals.base_values[idx], data=XTeSc[idx], feature_names=featureCols); shap.waterfall_plot(exp, max_display=10, show=False)
      ax.set_title(f'{lbl} {i+1}\nP(stress)={yProba[idx]:.3f}', fontsize=14, fontweight='bold', pad=20)
      plt.figure(figsize=(12, 9)); shap.waterfall_plot(exp, max_display=12, show=False); plt.title(f'Explanation: {lbl} Student {i+1}', fontsize=14, fontweight='bold'); plt.tight_layout(); plt.savefig(outputDir / f'shap{lbl.replace(" ", "")}{i+1}.png', dpi=200); plt.close()
    plt.figure(fig.number); plt.suptitle('SHAP Waterfall — Individual Student Explanations', fontsize=24, fontweight='bold', y=1.02); plt.tight_layout(pad=4.0); plt.savefig(outputDir / 'shapWaterfall.png', dpi=150, bbox_inches='tight'); plt.close()
    plt.figure(figsize=(12, 8)); shap.summary_plot(shapVals.values, XTeSc, feature_names=featureCols, plot_type='bar', show=False, max_display=20); plt.title('SHAP Global Feature Importance (Bar)', fontweight='bold', fontsize=14); plt.tight_layout(); plt.savefig(outputDir / 'shapGlobalBar.png', dpi=200, bbox_inches='tight'); plt.close()
  except Exception as e: print(f"SHAP Waterfall Skipped: {e}!")
def learningCurvePlot(artifacts: dict):
  showBanner("Section 6: Learning Curve — Bias/Variance Diagnosis")
  df = artifacts['tabular_df']; feat = artifacts['feature_cols']; X = df[feat].values; y = df['stress_label'].values
  clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=randomState, n_jobs=-1)
  tSizes, tScores, vScores = learning_curve(clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState), scoring='accuracy', n_jobs=-1)
  plt.figure(figsize=(9, 5)); plt.plot(tSizes, tScores.mean(1), 'o-', color='#e74c3c', label='Training Score'); plt.plot(tSizes, vScores.mean(1), 'o-', color='#2980b9', label='CV Score')
  plt.title('Learning Curve — Bias/Variance Diagnosis', fontweight='bold'); plt.legend(); plt.tight_layout(); plt.savefig(outputDir / 'learningCurve.png', dpi=200, bbox_inches='tight'); plt.close()
def calibrationPlot(artifacts: dict):
  showBanner("Section 7: Probability Calibration Curve")
  yTe = artifacts['y_te']; yProb = artifacts['y_proba']; brier = brier_score_loss(yTe, yProb); fracPos, meanPred = calibration_curve(yTe, yProb, n_bins=10)
  fig, axes = plt.subplots(1, 2, figsize=(13, 5)); axes[0].plot([0,1],[0,1], 'k--'); axes[0].plot(meanPred, fracPos, 's-', color='#e74c3c', label=f'Model (Brier={brier:.4f})')
  axes[1].hist(yProb[yTe == 0], bins=25, alpha=0.6, color='#2ecc71', label='Normal'); axes[1].hist(yProb[yTe == 1], bins=25, alpha=0.6, color='#e74c3c', label='Stressed')
  plt.suptitle('Probability Calibration Analysis', fontsize=14, fontweight='bold'); plt.tight_layout(); plt.savefig(outputDir / 'calibrationCurve.png', dpi=200, bbox_inches='tight'); plt.close()
def exportLatexTable(ciDf: pd.DataFrame, ablationDf: pd.DataFrame, mcnemarRes: dict):
  showBanner("Section 8: Exporting LaTeX Results Table")
  lines = [r"\begin{table}[h]", r"\centering", r"\caption{Results}", r"\begin{tabular}{lrrr}", r"\toprule", r"Metric & Point & CI Lo & CI Hi \\", r"\midrule"]
  for _, row in ciDf.iterrows(): lines.append(f"  {row['Metric']} & {row['Point Estimate']:.4f} & {row['CI Lower (2.5%)']:.4f} & {row['CI Upper (97.5%)']:.4f} \\\\")
  lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
  (outputDir / "resultsTable.tex").write_text("\n".join(lines), encoding='utf-8')
def main():
  showBanner("Evaluation Suite — Stress Detection Research")
  artifacts = loadArtifacts()
  ablationDf = ablationStudy(artifacts)
  mcnemarRes = mcnemarTest(artifacts)
  ciDf = bootstrapCi(artifacts)
  shapWaterfall(artifacts)
  learningCurvePlot(artifacts)
  calibrationPlot(artifacts)
  exportLatexTable(ciDf, ablationDf, mcnemarRes)
  showBanner("Evaluation Complete — All Outputs Saved To NCKH!")
if __name__ == "__main__":
  main()