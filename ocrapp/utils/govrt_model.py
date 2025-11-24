import joblib
import numpy as np
import re

# ---------------------------------------------
# Load all models once
# ---------------------------------------------
def load_models(model_dir="./models"):
    tfidf = joblib.load(f"{model_dir}/tfidf.joblib")
    svd = joblib.load(f"{model_dir}/svd_300.joblib")
    mlp = joblib.load(f"{model_dir}/ann_mlp.joblib")
    scaler = joblib.load(f"{model_dir}/embed_scaler.joblib")
    svm = joblib.load(f"{model_dir}/svm_on_ann.joblib")
    return tfidf, svd, mlp, scaler, svm

# ---------------------------------------------
# Clean Text
# ---------------------------------------------
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ---------------------------------------------
# ANN hidden embedding
# ---------------------------------------------
def mlp_hidden_embedding(mlp, X):
    relu = lambda a: np.maximum(a, 0)
    A = relu(X.dot(mlp.coefs_[0]) + mlp.intercepts_[0])
    A = relu(A.dot(mlp.coefs_[1]) + mlp.intercepts_[1])
    return A

# ---------------------------------------------
# Predict GOVT / NON-GOVT (1 = GOVT)
# ---------------------------------------------
def predict_texts(text_list, tfidf, svd, mlp, scaler, svm, return_probs=False):
    texts = [clean_text(t) for t in text_list]
    X_tfidf = tfidf.transform(texts)
    X_svd = svd.transform(X_tfidf)
    embed = mlp_hidden_embedding(mlp, X_svd)
    embed_scaled = scaler.transform(embed)
    preds = svm.predict(embed_scaled).astype(int)

    if return_probs and hasattr(svm, "predict_proba"):
        return preds, svm.predict_proba(embed_scaled)
    return preds, None
