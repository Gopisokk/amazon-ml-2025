"""
Minimal text-only baseline runner (log-space XGBoost).
Expects dataset/train.csv and dataset/test.csv with:
- sample_id
- catalog_content
- price (train only)
Outputs: submissions/baseline_textonly.csv
"""
import os, hashlib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA = os.path.join(BASE, "dataset")
OUT  = os.path.join(BASE, "submissions")
os.makedirs(OUT, exist_ok=True)

def norm_text(s: pd.Series):
    return s.fillna("").str.lower().str.replace(r"\s+"," ",regex=True).str.strip()

def make_groups(text: pd.Series):
    t = norm_text(text)
    return t.map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:12]).values

def smape(y_true, y_pred):
    y_true=np.asarray(y_true,float); y_pred=np.asarray(y_pred,float)
    denom=(np.abs(y_true)+np.abs(y_pred))/2.0
    m=denom>0
    return (np.abs(y_true[m]-y_pred[m])/denom[m]).mean()*100

def main():
    train = pd.read_csv(os.path.join(DATA,"train.csv"))
    test  = pd.read_csv(os.path.join(DATA,"test.csv"))
    y = train["price"].astype("float32").values
    sid_te = test["sample_id"].values

    tr = norm_text(train["catalog_content"]); te = norm_text(test["catalog_content"])
    X_tr = tr.str.len().to_numpy(np.int32).reshape(-1,1).astype("float32")
    X_te = te.str.len().to_numpy(np.int32).reshape(-1,1).astype("float32")

    groups = make_groups(train["catalog_content"])
    gkf = GroupKFold(n_splits=5).split(X_tr, y, groups)

    oof_log = np.zeros_like(y, dtype="float32")
    test_logs = []
    params = dict(objective="reg:squarederror", eval_metric="rmse",
                  tree_method="hist", device="cpu", max_depth=6, eta=0.05,
                  subsample=0.9, colsample_bytree=0.9)
    for f,(tr_idx,va_idx) in enumerate(gkf):
        dtr = xgb.DMatrix(X_tr[tr_idx], label=np.log1p(y[tr_idx]))
        dva = xgb.DMatrix(X_tr[va_idx], label=np.log1p(y[va_idx]))
        dte = xgb.DMatrix(X_te)
        bst = xgb.train(params, dtr, num_boost_round=1200,
                        evals=[(dtr,"tr"),(dva,"va")],
                        early_stopping_rounds=100, verbose_eval=False)
        oof_log[va_idx] = bst.predict(dva, iteration_range=(0, bst.best_iteration+1))
        test_logs.append(bst.predict(dte, iteration_range=(0, bst.best_iteration+1)))

    oof_price = np.expm1(oof_log)
    print("OOF SMAPE (toy):", round(smape(y, oof_price), 3))

    test_pred = np.expm1(np.vstack(test_logs).mean(0))
    sub = pd.DataFrame({"sample_id": sid_te, "price": np.clip(test_pred, 0.01, None)})
    out_csv = os.path.join(OUT, "baseline_textonly.csv")
    sub.to_csv(out_csv, index=False, float_format="%.6f")
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()
