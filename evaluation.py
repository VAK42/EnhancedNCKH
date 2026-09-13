import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, average_precision_score, precision_recall_curve, brier_score_loss, mean_squared_error, mean_absolute_error, confusion_matrix, roc_curve, log_loss)
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split, cross_val_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2, norm
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sns.set_theme(style='whitegrid')
outputD = Path("nckh")
modelD = outputD / "model"
imgD = outputD / "img"
dataD = Path("data")
randomState = 42
testSize = 0.2
seqLen = 30
LSTMDropout = 0.3
LSTMHidden = 64
LSTMLayers = 2
nBootstrap = 2000
alphaVal = 0.05
defaultThresh = 0.5
modelD.mkdir(parents=True, exist_ok=True)
imgD.mkdir(parents=True, exist_ok=True)
np.random.seed(randomState)
featureFamilies = {
  'VLE Session': ['total_clicks', 'mean_clicks', 'std_clicks', 'max_clicks', 'min_clicks', 'n_sessions', 'cv_clicks', 'abandon_rate'],
  'Temporal': ['mean_gap', 'std_gap', 'max_gap', 'engagement_cliff', 'active_span', 'activity_rate', 'late_drop', 'click_slope', 'rolling_vol_7d', 'first_access', 'last_access'],
  'Activity Type': ['n_activity_types', 'activity_entropy', 'quiz_ratio', 'forum_ratio'],
  'Assessment': ['mean_score', 'std_score', 'min_score', 'max_score', 'n_submitted', 'fail_rate', 'score_slope', 'late_submit'],
  'Demographics': ['gender', 'age_band', 'highest_education', 'num_of_prev_attempts', 'studied_credits', 'disability'],
  'LSTM Probability': [],
}
def showBanner(title):
  print("\n" + "=" * 80)
  print(f"{title}")
  print("=" * 80)
def computeMetrics(yTrue, yPred, yProba=None) -> dict:
  cm = confusion_matrix(yTrue, yPred)
  tn, fp, fn, tp = cm.ravel()
  metrics = {
    'Accuracy': accuracy_score(yTrue, yPred),
    'Balanced Accuracy': balanced_accuracy_score(yTrue, yPred),
    'Precision': precision_score(yTrue, yPred, zero_division=0),
    'Recall': recall_score(yTrue, yPred, zero_division=0),
    'Specificity': float(tn / (tn + fp + 1e-6)),
    'F1': f1_score(yTrue, yPred, zero_division=0),
    'MCC': matthews_corrcoef(yTrue, yPred),
    'Cohen Kappa': cohen_kappa_score(yTrue, yPred),
  }
  if yProba is not None:
    metrics['ROC-AUC'] = roc_auc_score(yTrue, yProba)
    metrics['PR-AUC'] = average_precision_score(yTrue, yProba)
    metrics['Brier'] = brier_score_loss(yTrue, yProba)
    metrics['MSE'] = float(mean_squared_error(yTrue, yProba))
    metrics['RMSE'] = float(np.sqrt(metrics['MSE']))
    metrics['MAE'] = float(mean_absolute_error(yTrue, yProba))
    metrics['LogLoss'] = float(log_loss(yTrue, yProba))
  return metrics
def loadArtifacts():
  showBanner("S1: Loading Pipeline Artifacts")
  ensemblePath = modelD / "StackingEnsemble.pkl"
  scalerPath = modelD / "Scaler.pkl"
  if not ensemblePath.exists():
    raise FileNotFoundError(f"{ensemblePath} Not Found! Run main.py First!")
  model = joblib.load(ensemblePath)
  scaler = joblib.load(scalerPath)
  print(f"Loaded: {ensemblePath}")
  print(f"Loaded: {scalerPath}")
  cacheFile = modelD / "EvaluationCache.pkl"
  if cacheFile.exists():
    print(f"Loading Pre-Engineered Features From Cache: {cacheFile}")
    return joblib.load(cacheFile)
  try:
    from main import (
      loadOULAD, buildStressLabels, engineerVLEFeatures, engineerAssessmentFeatures,
      engineerStudentInfoFeatures, mergeFeatures, buildSequences, StressLSTM
    )
    print("Cache Not Found! Re-Engineering Features (One-Time Process)...")
    oulad = loadOULAD(dataD)
    labels = buildStressLabels(oulad)
    vleFeats = engineerVLEFeatures(oulad)
    infoFeats = engineerStudentInfoFeatures(oulad)
    asmntFeats = engineerAssessmentFeatures(oulad)
    tabularDf = mergeFeatures(vleFeats, asmntFeats, infoFeats, labels)
    XSeq, ySeq, gMean, gStd, sidSeq = buildSequences(oulad, labels)
    LSTMModel = StressLSTM(inputSize=XSeq.shape[2])
    LSTMPytorch = modelD / "LSTM.pt"
    if LSTMPytorch.exists():
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      LSTMModel.load_state_dict(torch.load(LSTMPytorch, map_location=device))
      LSTMModel.to(device)
      LSTMModel.eval()
      with torch.no_grad():
        probs = []
        for batchIdx in range(0, len(XSeq), 256):
          batch = torch.tensor(XSeq[batchIdx:batchIdx+256]).to(device)
          logits = LSTMModel(batch)
          prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
          probs.extend(prob)
      probsArr = np.array(probs).flatten()
      probMap = dict(zip(sidSeq, probsArr))
      tabularDf['LSTM_prob'] = tabularDf['id_student'].map(probMap)
      tabularDf['LSTM_prob'] = tabularDf['LSTM_prob'].fillna(tabularDf['LSTM_prob'].median())
      featureFamilies['LSTM Probability'] = ['LSTM_prob']
      print(f"LSTM Probabilities Aligned Via Student ID Map!")
    else:
      print("LSTM.pt Not Found - LSTM Family Skipped In Ablation!")
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
    print(f"Accuracy: {accuracy_score(yTe, yPredTuned):.4f}")
    data = {
      'model': model, 'scaler': scaler, 'tabular_df': tabularDf, 'feature_cols': featureCols,
      'X_tr': XTr, 'X_te': XTe, 'y_tr': yTr, 'y_te': yTe, 'X_te_sc': XTeSc,
      'y_pred': yPred, 'y_pred_tuned': yPredTuned, 'y_proba': yProba, 'best_thresh': bestThresh,
    }
    targetCache = modelD / "EvaluationCache.pkl"
    print(f"Saving Features To Cache: {targetCache}")
    joblib.dump(data, targetCache)
    return data
  except ImportError as e:
    raise ImportError(f"Could Not Import Pipeline: {e}!")
def ablationStudy(artifacts: dict) -> pd.DataFrame:
  showBanner("S2: Ablation Study - Feature Family Contribution")
  df = artifacts['tabular_df']; featAll = artifacts['feature_cols']; y = df['stress_label'].values
  smote = SMOTE(random_state=randomState)
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)
  def cvScore(XArr):
    accs, f1s, aucs = [], [], []
    for trIdx, teIdx in skf.split(XArr, y):
      XTr, XTe = XArr[trIdx], XArr[teIdx]; yTr, yTe = y[trIdx], y[teIdx]
      sc = StandardScaler(); XTr = sc.fit_transform(XTr); XTe = sc.transform(XTe)
      XTrS, yTrS = smote.fit_resample(XTr, yTr)
      clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=randomState, n_jobs=-1)
      clf.fit(XTrS, yTrS)
      accs.append(accuracy_score(yTe, clf.predict(XTe))); f1s.append(f1_score(yTe, clf.predict(XTe), zero_division=0)); aucs.append(roc_auc_score(yTe, clf.predict_proba(XTe)[:, 1]))
    return np.mean(accs), np.mean(f1s), np.mean(aucs)
  fullAcc, fullF1, fullAuc = cvScore(df[featAll].values)
  print(f"\nFull Model → Acc: {fullAcc:.4f}  F1: {fullF1:.4f}  AUC: {fullAuc:.4f}")
  rows = [{'Family': 'Full Model', 'Accuracy': fullAcc, 'F1': fullF1, 'AUC': fullAuc}]
  for family, cols in featureFamilies.items():
    present = [c for c in cols if c in featAll]
    if not present: continue
    ablatedCols = [c for c in featAll if c not in present]
    if not ablatedCols: continue
    acc, f1, auc = cvScore(df[ablatedCols].values)
    rows.append({'Family': f'w/o {family}', 'Accuracy': acc, 'F1': f1, 'AUC': auc})
  resultsDf = pd.DataFrame(rows)
  fig, ax = plt.subplots(figsize=(12, 7))
  yPos = np.arange(len(resultsDf)); height = 0.25
  ax.barh(yPos - height, resultsDf['Accuracy'], height, label='Accuracy', color='#2A9D8F', edgecolor='black')
  ax.barh(yPos, resultsDf['F1'], height, label='F1-Score', color='#E76F51', edgecolor='black')
  ax.barh(yPos + height, resultsDf['AUC'], height, label='ROC-AUC', color='#264653', edgecolor='black')
  ax.set_yticks(yPos); ax.set_yticklabels(resultsDf['Family'], fontsize=11, weight='bold')
  ax.set_xlabel('Performance Score', fontsize=12, weight='bold'); ax.set_xlim(0.80, 1.0)
  ax.set_title('Ablation Study - Feature Family Impact Across Evaluation Metrics', fontsize=14, weight='bold', pad=15)
  ax.legend(loc='lower left', fontsize=11); ax.grid(axis='x', linestyle=':', alpha=0.7)
  fig.tight_layout(); fig.savefig(imgD / 'AblationStudy.png', dpi=300); plt.close(fig)
  print("Saved: AblationStudy.png"); return resultsDf
def McNemarTest(artifacts: dict) -> dict:
  showBanner("S3: McNemar's Test - Statistical Significance")
  yTe = artifacts['y_te']; yPred = artifacts['y_pred_tuned']
  XTrSc = artifacts['scaler'].transform(artifacts['X_tr'])
  baselineClf = LogisticRegression(max_iter=1000, random_state=randomState)
  baselineClf.fit(XTrSc, artifacts['y_tr'])
  yBaseline = baselineClf.predict(artifacts['X_te_sc'])
  baseAcc = float(accuracy_score(yTe, yBaseline))
  ourCorrect = (yPred == yTe); baselineCorrect = (yBaseline == yTe)
  b = int(np.sum(ourCorrect & ~baselineCorrect)); c = int(np.sum(~ourCorrect & baselineCorrect))
  a = int(np.sum(ourCorrect & baselineCorrect)); d = int(np.sum(~ourCorrect & ~baselineCorrect))
  table = np.array([[a, b], [c, d]])
  result = mcnemar(table, exact=False, correction=True); pValue = float(result.pvalue)
  print(f"\nBaseline Model Accuracy: {baseAcc:.4f}")
  print(f"McNemar χ² = {result.statistic:.4f}\np-Value = {pValue:.6f}\nSignificant: {'Yes ✓' if pValue < alphaVal else 'No ✗'}")
  return {'chi2': float(result.statistic), 'p_value': pValue, 'significant': pValue < alphaVal, 'delta_acc': float(accuracy_score(yTe, yPred) - baseAcc), 'table': table, 'baseline_acc': baseAcc}
def McNemarPlot(artifacts: dict, McNemarRes: dict):
  fig, ax = plt.subplots(figsize=(8, 7))
  contingency = McNemarRes['table']
  cax = ax.imshow(contingency, cmap="Purples", interpolation="nearest")
  fig.colorbar(cax, fraction=0.046, pad=0.04)
  cellText = [[f"Both Correct\na = {contingency[0,0]:,}", f"Proposed Correct / Base Wrong\nb = {contingency[0,1]:,}"], [f"Proposed Wrong / Base Correct\nc = {contingency[1,0]:,}", f"Both Wrong\nd = {contingency[1,1]:,}"]]
  for i in range(2):
    for j in range(2):
      col = "white" if contingency[i, j] > (len(artifacts['y_te']) * 0.25) else "black"
      ax.text(j, i, cellText[i][j], ha="center", va="center", color=col, fontsize=12, weight="bold")
  ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
  ax.set_xticklabels(["Baseline Correct", "Baseline Incorrect"], fontsize=11, weight="bold")
  ax.set_yticklabels(["Proposed Correct", "Proposed Incorrect"], fontsize=11, weight="bold")
  pStr = f"{McNemarRes['p_value']:.6f}" if McNemarRes['p_value'] >= 1e-6 else "< 1e-6"
  ax.set_title(f"McNemar 2x2 Contingency Matrix (N = {len(artifacts['y_te']):,})\nChi^2 = {McNemarRes['chi2']:.4f} | p-Value = {pStr}", fontsize=13, weight="bold", pad=15)
  fig.tight_layout(); fig.savefig(imgD / "McNemarSignificance.png", dpi=300); plt.close(fig)
  print("Saved: McNemarSignificance.png")
def bootstrapCI(artifacts: dict) -> pd.DataFrame:
  showBanner("S4: Bootstrap Confidence Intervals (95% N=2000)")
  yTe = artifacts['y_te']; yPred = artifacts['y_pred_tuned']; yProb = artifacts['y_proba']; rng = np.random.default_rng(randomState); n = len(yTe)
  metricFns = {
    'Accuracy': lambda yt, yp, ypr: accuracy_score(yt, yp), 'Balanced Accuracy': lambda yt, yp, ypr: balanced_accuracy_score(yt, yp),
    'Precision': lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0), 'Recall': lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
    'F1-Score': lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0), 'ROC-AUC': lambda yt, yp, ypr: roc_auc_score(yt, ypr),
    'PR-AUC': lambda yt, yp, ypr: average_precision_score(yt, ypr), 'MCC': lambda yt, yp, ypr: matthews_corrcoef(yt, yp),
    "Cohen's Kappa": lambda yt, yp, ypr: cohen_kappa_score(yt, yp),
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
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.barh(range(len(ciDf)), ciDf['CI Upper (97.5%)'] - ciDf['CI Lower (2.5%)'], left=ciDf['CI Lower (2.5%)'], height=0.45, color='#2A9D8F', alpha=0.45, edgecolor='black')
  ax.scatter(ciDf['Point Estimate'], range(len(ciDf)), color='#1B365D', zorder=5, s=70)
  ax.set_yticks(range(len(ciDf))); ax.set_yticklabels(ciDf['Metric'], fontsize=11, weight='bold')
  ax.set_xlabel('Point Estimate + 95% Confidence Bounds', fontsize=12, weight='bold')
  ax.set_title('95% Empirical Bootstrap Confidence Intervals (2,000 Resamples)', fontsize=13, weight='bold', pad=15)
  fig.tight_layout(); fig.savefig(imgD / 'BootstrapCI.png', dpi=300); plt.close(fig)
  print("Saved: BootstrapCI.png"); return ciDf
def shapAnalysis(artifacts: dict):
  showBanner("S5: SHAP Global + Cohort Interpretability Analysis")
  XTeSc = artifacts['X_te_sc']; yTe = artifacts['y_te']; featureCols = artifacts['feature_cols']
  try:
    xgbBase = artifacts['model'].named_estimators_['xgb']; explainer = shap.TreeExplainer(xgbBase); shapVals = explainer(XTeSc)
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shapVals.values, XTeSc, feature_names=featureCols, show=False, max_display=15)
    plt.title('SHAP Global Feature Impact Distribution (Beeswarm)', fontsize=14, weight='bold', pad=15)
    plt.tight_layout(); plt.savefig(imgD / 'ShapSummary.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("Saved: ShapSummary.png")
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shapVals.values, XTeSc, feature_names=featureCols, plot_type='bar', show=False, max_display=15)
    plt.title('SHAP Global Feature Importance (Mean |SHAP|)', fontsize=14, weight='bold', pad=15)
    plt.tight_layout(); plt.savefig(imgD / 'ShapGlobalBar.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("Saved: ShapGlobalBar.png")
    stressedMask = (yTe == 1); normalMask = (yTe == 0)
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
  except Exception as e: print(f"SHAP Analysis Skipped: {e}!")
def learningCurvePlot(artifacts: dict):
  showBanner("S6: Learning Curve - Bias/Variance Diagnosis")
  df = artifacts['tabular_df']; feat = artifacts['feature_cols']; X = df[feat].values; y = df['stress_label'].values
  print("Calculating Learning Curve (5-Fold Stratified CV)...")
  clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=randomState, n_jobs=-1)
  tSizes, tScores, vScores = learning_curve(clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState), scoring='accuracy', n_jobs=1)
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(tSizes, tScores.mean(1), 'o-', color='#E74C3C', label='Training Score', lw=2.5)
  ax.plot(tSizes, vScores.mean(1), 'o-', color='#2980B9', label='Cross-Validation Score', lw=2.5)
  ax.fill_between(tSizes, tScores.mean(1) - tScores.std(1), tScores.mean(1) + tScores.std(1), alpha=0.1, color='#E74C3C')
  ax.fill_between(tSizes, vScores.mean(1) - vScores.std(1), vScores.mean(1) + vScores.std(1), alpha=0.1, color='#2980B9')
  ax.set_title('Learning Curve - Bias/Variance Convergence Diagnosis', fontweight='bold', fontsize=14, pad=15)
  ax.set_xlabel('Training Sample Size', fontsize=12, weight='bold'); ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
  ax.legend(loc='lower right', fontsize=11); ax.grid(True, linestyle=':', alpha=0.7)
  fig.tight_layout(); fig.savefig(imgD / 'LearningCurve.png', dpi=300); plt.close(fig)
  print("Saved: LearningCurve.png")
def calibrationCurvePlot(artifacts: dict):
  showBanner("S7: Probability Calibration Curve")
  yTe = artifacts['y_te']; yProb = artifacts['y_proba']; brier = brier_score_loss(yTe, yProb); fracPos, meanPred = calibration_curve(yTe, yProb, n_bins=10)
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration (y = x)')
  ax.plot(meanPred, fracPos, 's-', color='#E76F51', lw=2.5, markersize=8, label=f'Stacking Ensemble (Brier = {brier:.4f})')
  ax.set_xlabel('Mean Predicted Probability', fontsize=12, weight='bold')
  ax.set_ylabel('Fraction Of Positives - Empirical', fontsize=12, weight='bold')
  ax.set_title('Probability Calibration Reliability Diagram', fontsize=14, weight='bold', pad=15)
  ax.legend(loc='lower right', fontsize=11); ax.grid(True, linestyle=':', alpha=0.7)
  ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.0])
  fig.tight_layout(); fig.savefig(imgD / 'CalibrationCurve.png', dpi=300); plt.close(fig)
  print("Saved: CalibrationCurve.png")
def probabilityDistributionPlot(artifacts: dict):
  showBanner("S8: Predicted Probability Distribution By Class")
  yTe = artifacts['y_te']; yProb = artifacts['y_proba']
  fig, ax = plt.subplots(figsize=(9, 6))
  ax.hist(yProb[yTe == 0], bins=35, alpha=0.65, color='#2A9D8F', label='Normal Students (T=0)', edgecolor='black')
  ax.hist(yProb[yTe == 1], bins=35, alpha=0.65, color='#E76F51', label='Stressed Students (T=1)', edgecolor='black')
  ax.set_xlabel('Predicted Stress Probability', fontsize=12, weight='bold')
  ax.set_ylabel('Student Frequency Count', fontsize=12, weight='bold')
  ax.set_title('Model Predicted Probability Separation Distribution', fontsize=14, weight='bold', pad=15)
  ax.legend(loc='upper center', fontsize=11); ax.grid(axis='y', linestyle=':', alpha=0.7)
  fig.tight_layout(); fig.savefig(imgD / 'ProbabilityDistribution.png', dpi=300); plt.close(fig)
  print("Saved: ProbabilityDistribution.png")
def detailedConfusionMatrixPlot(artifacts: dict):
  showBanner("S9: Detailed Confusion Matrix Plot")
  yTe = artifacts['y_te']; yPredTuned = artifacts['y_pred_tuned']
  cm = confusion_matrix(yTe, yPredTuned)
  tn, fp, fn, tp = cm.ravel(); total = len(yTe)
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
  acc = accuracy_score(yTe, yPredTuned)
  ax.set_title(f"Detailed Confusion Matrix (Test Set N = {total:,})\nTuned Youden Threshold = {artifacts.get('best_thresh', defaultThresh):.3f} | Accuracy = {acc*100:.2f}%", fontsize=14, weight="bold", pad=15)
  fig.tight_layout(); fig.savefig(imgD / "ConfusionMatrix.png", dpi=300); plt.close(fig)
  print("Saved: ConfusionMatrix.png")
def fullMetricsReportPlot(artifacts: dict, ciDf: pd.DataFrame):
  showBanner("S10: Full Metrics Report Plot")
  yTe = artifacts['y_te']; yPred = artifacts['y_pred']; yPredTuned = artifacts['y_pred_tuned']; yProba = artifacts['y_proba']
  mDef = computeMetrics(yTe, yPred, yProba)
  mTun = computeMetrics(yTe, yPredTuned, yProba)
  ciMap = {row['Metric']: f"{row['Point Estimate']:.4f} [{row['CI Lower (2.5%)']:.4f} - {row['CI Upper (97.5%)']:.4f}]" for _, row in ciDf.iterrows()}
  headers = ["Metric", "Formula", f"Default Threshold ({defaultThresh:.3f})", "Youden Threshold (Tuned)", "Bootstrap 95% CI"]
  rows = [
    ["Accuracy", "(TP + TN) / Total", f"{mDef['Accuracy']:.4f} ({mDef['Accuracy']*100:.2f}%)", f"{mTun['Accuracy']:.4f} ({mTun['Accuracy']*100:.2f}%)", ciMap.get('Accuracy', 'N/A')],
    ["Balanced Accuracy", "(TPR + TNR) / 2", f"{mDef['Balanced Accuracy']:.4f} ({mDef['Balanced Accuracy']*100:.2f}%)", f"{mTun['Balanced Accuracy']:.4f} ({mTun['Balanced Accuracy']*100:.2f}%)", ciMap.get('Balanced Accuracy', 'N/A')],
    ["Precision", "TP / (TP + FP)", f"{mDef['Precision']:.4f} ({mDef['Precision']*100:.2f}%)", f"{mTun['Precision']:.4f} ({mTun['Precision']*100:.2f}%)", ciMap.get('Precision', 'N/A')],
    ["Recall / Sensitivity", "TP / (TP + FN)", f"{mDef['Recall']:.4f} ({mDef['Recall']*100:.2f}%)", f"{mTun['Recall']:.4f} ({mTun['Recall']*100:.2f}%)", ciMap.get('Recall', 'N/A')],
    ["Specificity", "TN / (TN + FP)", f"{mDef['Specificity']:.4f} ({mDef['Specificity']*100:.2f}%)", f"{mTun['Specificity']:.4f} ({mTun['Specificity']*100:.2f}%)", "N/A"],
    ["F1-Score", "2 * (P * R) / (P + R)", f"{mDef['F1']:.4f} ({mDef['F1']*100:.2f}%)", f"{mTun['F1']:.4f} ({mTun['F1']*100:.2f}%)", ciMap.get('F1-Score', 'N/A')],
    ["ROC-AUC", "Integral TPR d(FPR)", f"{mDef['ROC-AUC']:.4f} ({mDef['ROC-AUC']*100:.2f}%)", f"{mTun['ROC-AUC']:.4f} ({mTun['ROC-AUC']*100:.2f}%)", ciMap.get('ROC-AUC', 'N/A')],
    ["PR-AUC (Average Precision)", "Integral Prec d(Rec)", f"{mDef['PR-AUC']:.4f} ({mDef['PR-AUC']*100:.2f}%)", f"{mTun['PR-AUC']:.4f} ({mTun['PR-AUC']*100:.2f}%)", ciMap.get('PR-AUC', 'N/A')],
    ["MCC", "(TP*TN - FP*FN) / Denom", f"{mDef['MCC']:.4f}", f"{mTun['MCC']:.4f}", ciMap.get('MCC', 'N/A')],
    ["Cohen's Kappa", "(p_o - p_e) / (1 - p_e)", f"{mDef['Cohen Kappa']:.4f}", f"{mTun['Cohen Kappa']:.4f}", ciMap.get("Cohen's Kappa", 'N/A')],
    ["MSE / Brier Score", "Sum (p_i - y_i)^2 / N", f"{mDef['MSE']:.4f}", f"{mTun['MSE']:.4f}", "N/A"],
    ["RMSE", "Sqrt(MSE)", f"{mDef['RMSE']:.4f}", f"{mTun['RMSE']:.4f}", "N/A"],
    ["MAE", "Sum |p_i - y_i| / N", f"{mDef['MAE']:.4f}", f"{mTun['MAE']:.4f}", "N/A"],
    ["Log-Loss", "-Sum [y ln p + (1-y) ln(1-p)] / N", f"{mDef['LogLoss']:.4f}", f"{mTun['LogLoss']:.4f}", "N/A"]
  ]
  fig, ax = plt.subplots(figsize=(15, 9)); ax.axis("off")
  tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
  tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.0)
  for (r, c), cell in tbl.get_celld().items():
    if r == 0:
      cell.set_facecolor("#1B365D"); cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
      cell.set_facecolor("#F0F4F8")
    else:
      cell.set_facecolor("#FFFFFF")
  ax.set_title("Full Comprehensive Evaluation Metrics Report", fontsize=15, weight="bold", pad=25)
  fig.tight_layout(); fig.savefig(imgD / "FullMetricsReport.png", dpi=300); plt.close(fig)
  print("Saved: FullMetricsReport.png")
def ROCCurvePlot(artifacts: dict):
  showBanner("S11: ROC Curve Plot")
  yTe = artifacts['y_te']; yProba = artifacts['y_proba']
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
def precisionRecallCurvePlot(artifacts: dict):
  showBanner("S12: Precision-Recall Curve Plot")
  yTe = artifacts['y_te']; yProba = artifacts['y_proba']
  prec, rec, _ = precision_recall_curve(yTe, yProba)
  prAuc = average_precision_score(yTe, yProba)
  baselineRatio = sum(yTe == 1) / len(yTe)
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.plot(rec, prec, color='#2A9D8F', lw=2.5, label=f'Stacking Ensemble (PR-AUC = {prAuc:.4f})')
  ax.axhline(baselineRatio, color='#E76F51', linestyle='--', lw=1.5, label=f'No-Skill Baseline ({baselineRatio:.4f})')
  ax.fill_between(rec, prec, alpha=0.15, color='#2A9D8F')
  ax.set_xlabel('Recall / Sensitivity', fontsize=12, weight='bold')
  ax.set_ylabel('Precision', fontsize=12, weight='bold')
  ax.set_title('Precision-Recall Curve', fontsize=13, weight='bold', pad=15)
  ax.legend(loc='lower left', fontsize=11)
  ax.set_ylim([0.0, 1.05]); ax.set_xlim([0.0, 1.0])
  fig.tight_layout(); fig.savefig(imgD / 'PrecisionRecallCurve.png', dpi=300); plt.close(fig)
  print("Saved: PrecisionRecallCurve.png")
def cvComparisonPlot(artifacts: dict):
  showBanner("S13: 5-Fold CV Comparison Plot")
  df = artifacts['tabular_df']; feat = artifacts['feature_cols']; X = df[feat].values; y = df['stress_label'].values
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)
  xgbBase = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=randomState, n_jobs=-1)
  lgbBase = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=randomState, n_jobs=-1, verbose=-1)
  xgbPipe = ImbPipeline([('smote', SMOTE(random_state=randomState)), ('scaler', StandardScaler()), ('clf', xgbBase)])
  lgbPipe = ImbPipeline([('smote', SMOTE(random_state=randomState)), ('scaler', StandardScaler()), ('clf', lgbBase)])
  xgbScores = cross_val_score(xgbPipe, X, y, cv=skf, scoring='accuracy', n_jobs=1)
  lgbScores = cross_val_score(lgbPipe, X, y, cv=skf, scoring='accuracy', n_jobs=1)
  names = ['XGBoost', 'LightGBM']
  means = [xgbScores.mean(), lgbScores.mean()]
  stds = [xgbScores.std(), lgbScores.std()]
  fig, ax = plt.subplots(figsize=(8, 6))
  bars = ax.bar(names, means, yerr=stds, capsize=6, color=['#2E86AB', '#E07A5F'], edgecolor='black', width=0.45)
  for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{m:.4f}', ha='center', fontweight='bold', fontsize=11)
  ax.set_ylim([0.85, 0.96]); ax.set_ylabel('Cross-Validation Accuracy', fontsize=12, weight='bold')
  ax.set_title('5-Fold Stratified Cross-Validation Accuracy Comparison', fontsize=13, weight='bold', pad=15)
  ax.grid(axis='y', linestyle=':', alpha=0.7)
  fig.tight_layout(); fig.savefig(imgD / 'CVComparison.png', dpi=300); plt.close(fig)
  print("Saved: CVComparison.png")
def cvFoldsDetailPlot(artifacts: dict):
  showBanner("S14: 5-Fold Cross-Validation Fold Detail Plot")
  df = artifacts['tabular_df']; feat = artifacts['feature_cols']; X = df[feat].values; y = df['stress_label'].values
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)
  xgbBase = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=randomState, n_jobs=-1)
  lgbBase = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=randomState, n_jobs=-1, verbose=-1)
  xgbPipe = ImbPipeline([('smote', SMOTE(random_state=randomState)), ('scaler', StandardScaler()), ('clf', xgbBase)])
  lgbPipe = ImbPipeline([('smote', SMOTE(random_state=randomState)), ('scaler', StandardScaler()), ('clf', lgbBase)])
  xgbScores = cross_val_score(xgbPipe, X, y, cv=skf, scoring='accuracy', n_jobs=1)
  lgbScores = cross_val_score(lgbPipe, X, y, cv=skf, scoring='accuracy', n_jobs=1)
  folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean CV"]
  xgbAll = list(xgbScores) + [xgbScores.mean()]
  lgbAll = list(lgbScores) + [lgbScores.mean()]
  fig, ax = plt.subplots(figsize=(11, 6)); xPos = np.arange(len(folds)); w = 0.35
  rects1 = ax.bar(xPos - w/2, xgbAll, w, label="XGBoost", color="#2E86AB", edgecolor="black")
  rects2 = ax.bar(xPos + w/2, lgbAll, w, label="LightGBM", color="#E07A5F", edgecolor="black")
  ax.set_ylabel("Cross-Validation Accuracy", fontsize=12, weight="bold")
  ax.set_title("5-Fold Stratified Cross-Validation Fold-By-Fold Breakdown", fontsize=14, weight="bold", pad=15)
  ax.set_xticks(xPos); ax.set_xticklabels(folds, fontsize=11, weight="bold"); ax.set_ylim(0.88, 0.95)
  ax.axhline(xgbScores.mean(), color="#2E86AB", linestyle="--", alpha=0.6)
  ax.axhline(lgbScores.mean(), color="#E07A5F", linestyle="--", alpha=0.6)
  for rect in rects1:
    h = rect.get_height(); ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")
  for rect in rects2:
    h = rect.get_height(); ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")
  ax.legend(loc="lower right", fontsize=11); ax.grid(axis="y", linestyle=":", alpha=0.7)
  fig.tight_layout(); fig.savefig(imgD / "CVFoldsDetail.png", dpi=300); plt.close(fig)
  print("Saved: CVFoldsDetail.png")
def baselineComparisonPlot(artifacts: dict):
  showBanner("S15: Baseline Comparison Benchmark Plot")
  XTrSc = artifacts['scaler'].transform(artifacts['X_tr'])
  yTr = artifacts['y_tr']
  XTeSc = artifacts['X_te_sc']
  yTe = artifacts['y_te']
  models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=randomState),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=randomState),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=randomState, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=randomState, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=randomState, n_jobs=-1, verbose=-1),
  }
  accList, f1List, aucList = [], [], []
  names = list(models.keys()) + ['Proposed Hybrid']
  for name, clf in models.items():
    clf.fit(XTrSc, yTr)
    pred = clf.predict(XTeSc)
    prob = clf.predict_proba(XTeSc)[:, 1] if hasattr(clf, 'predict_proba') else pred
    accList.append(accuracy_score(yTe, pred))
    f1List.append(f1_score(yTe, pred, zero_division=0))
    aucList.append(roc_auc_score(yTe, prob))
  accList.append(accuracy_score(yTe, artifacts['y_pred_tuned']))
  f1List.append(f1_score(yTe, artifacts['y_pred_tuned'], zero_division=0))
  aucList.append(roc_auc_score(yTe, artifacts['y_proba']))
  fig, ax = plt.subplots(figsize=(13, 7)); xPos = np.arange(len(names)); w = 0.26
  r1 = ax.bar(xPos - w, accList, w, label="Accuracy", color="#3D5A80", edgecolor="black")
  r2 = ax.bar(xPos, f1List, w, label="F1-Score", color="#EE6C4D", edgecolor="black")
  r3 = ax.bar(xPos + w, aucList, w, label="ROC-AUC", color="#293241", edgecolor="black")
  ax.set_ylabel("Score", fontsize=12, weight="bold")
  ax.set_title("Benchmarking: Proposed Hybrid Architecture vs All Baseline Models", fontsize=14, weight="bold", pad=15)
  ax.set_xticks(xPos); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10, weight="bold"); ax.set_ylim(0.65, 1.02)
  for rect in r1:
    h = rect.get_height(); ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
  for rect in r2:
    h = rect.get_height(); ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
  for rect in r3:
    h = rect.get_height(); ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
  ax.legend(loc="lower right", fontsize=11); ax.grid(axis="y", linestyle=":", alpha=0.7)
  fig.tight_layout(); fig.savefig(imgD / "BaselineComparison.png", dpi=300); plt.close(fig)
  print("Saved: BaselineComparison.png")
def datasetDistributionPlot(artifacts: dict):
  showBanner("S16: Dataset Distribution Plot")
  df = artifacts['tabular_df']
  nStressed = int((df['stress_label'] == 1).sum())
  nNormal = int((df['stress_label'] == 0).sum())
  labels = [f"Stressed / At-Risk (1)\n{nStressed:,} ({nStressed/len(df)*100:.1f}%)", f"Normal (0)\n{nNormal:,} ({nNormal/len(df)*100:.1f}%)"]
  fig, ax = plt.subplots(figsize=(7, 6))
  ax.pie([nStressed, nNormal], labels=labels, colors=["#E76F51", "#2A9D8F"], autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11, "weight": "bold"}, explode=(0.04, 0))
  ax.set_title(f"OULAD Student Label Distribution (Total: {len(df):,})", fontsize=13, weight="bold", pad=15)
  fig.tight_layout(); fig.savefig(imgD / "DatasetDistribution.png", dpi=300); plt.close(fig)
  print("Saved: DatasetDistribution.png")
def trainTestSplitPlot(artifacts: dict):
  showBanner("S17: Train / Test Partitioning Plot")
  nTr = len(artifacts['X_tr']); nTe = len(artifacts['X_te'])
  labels = [f"Train Set ({nTr/(nTr+nTe)*100:.0f}%)\n{nTr:,} Students", f"Test Set ({nTe/(nTr+nTe)*100:.0f}%)\n{nTe:,} Students"]
  fig, ax = plt.subplots(figsize=(7, 6))
  ax.pie([nTr, nTe], labels=labels, colors=["#457B9D", "#F4A261"], autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11, "weight": "bold"}, explode=(0.04, 0))
  ax.set_title("Dataset Stratified Train / Test Partitioning", fontsize=13, weight="bold", pad=15)
  fig.tight_layout(); fig.savefig(imgD / "TrainTestSplit.png", dpi=300); plt.close(fig)
  print("Saved: TrainTestSplit.png")
def LSTMTrainingHistoryPlot():
  histPath = modelD / 'LSTMHistory.pkl'
  if not histPath.exists(): return
  history = joblib.load(histPath)
  nEpochs = len(history['val_acc'])
  ep = range(1, nEpochs + 1)
  fig, ax1 = plt.subplots(figsize=(9, 6))
  ax2 = ax1.twinx()
  line1 = ax1.plot(ep, history['train_loss'], color='#E76F51', lw=2.5, label='Training Loss')
  line2 = ax2.plot(ep, history['val_acc'], color='#2A9D8F', lw=2.5, label='Validation Accuracy')
  ax1.set_xlabel(f'Epoch ({nEpochs})', fontsize=12, weight='bold')
  ax1.set_ylabel('Training Loss', color='#E76F51', fontsize=12, weight='bold')
  ax2.set_ylabel('Validation Accuracy', color='#2A9D8F', fontsize=12, weight='bold')
  ax1.set_title(f'Bi-LSTM Temporal Model Training Dynamics Across {nEpochs} Epochs', fontsize=13, weight='bold', pad=15)
  lines = line1 + line2; labels = [l.get_label() for l in lines]
  ax1.legend(lines, labels, loc='center right', fontsize=11)
  fig.tight_layout(); fig.savefig(imgD / 'LSTMTrainingHistory.png', dpi=300); plt.close(fig)
  print("Saved: LSTMTrainingHistory.png")
def main():
  showBanner("Evaluation Suite - Stress Detection Research")
  artifacts = loadArtifacts()
  ablationDf = ablationStudy(artifacts)
  McNemarRes = McNemarTest(artifacts)
  McNemarPlot(artifacts, McNemarRes)
  ciDf = bootstrapCI(artifacts)
  shapAnalysis(artifacts)
  learningCurvePlot(artifacts)
  calibrationCurvePlot(artifacts)
  probabilityDistributionPlot(artifacts)
  detailedConfusionMatrixPlot(artifacts)
  fullMetricsReportPlot(artifacts, ciDf)
  ROCCurvePlot(artifacts)
  precisionRecallCurvePlot(artifacts)
  cvComparisonPlot(artifacts)
  cvFoldsDetailPlot(artifacts)
  baselineComparisonPlot(artifacts)
  datasetDistributionPlot(artifacts)
  trainTestSplitPlot(artifacts)
  LSTMTrainingHistoryPlot()
  showBanner("Finished!!")
if __name__ == "__main__":
  main()