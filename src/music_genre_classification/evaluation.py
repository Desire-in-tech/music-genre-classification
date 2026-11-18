"""Evaluation helpers with consistent multiclass metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score

def evaluate_predictions(y_true,y_pred,labels):
    return {"accuracy":float(accuracy_score(y_true,y_pred)),
      "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
      "macro_precision":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
      "macro_recall":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
      "classification_report":classification_report(y_true,y_pred,labels=labels,output_dict=True,zero_division=0),
      "confusion_matrix":confusion_matrix(y_true,y_pred,labels=labels).tolist(),"labels":list(labels)}

def evaluate_classifier(model,X_test,y_test,*,labels=None):
    return evaluate_predictions(y_test,model.predict(X_test),labels or sorted(set(y_test)))

def feature_matrix(features):
    required={"track_id","genre"}; missing=required.difference(features.columns)
    if missing: raise ValueError(f"features is missing required columns: {sorted(missing)}")
    X=features.drop(columns=["track_id","genre"])
    if X.empty or not np.isfinite(X.to_numpy(dtype=float)).all(): raise ValueError("Feature matrix must contain finite numeric predictors.")
    return X,features.genre.copy()
