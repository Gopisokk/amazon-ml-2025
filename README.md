# Smart Product Pricing — Amazon ML Challenge 2025

Price prediction for e-commerce products using **text-only** features and a **lightweight 2-model ensemble**.

- **Public LB Score:** `54.6631 SMAPE`
- **Final file submitted:** `sub_ensemble-v1.csv`
- **Metric:** SMAPE (Symmetric Mean Absolute Percentage Error, lower is better)

---

## 1) Problem Statement

Given a catalog of products with text fields, predict the **price** of each product in the hidden test set.

- **Input:** `sample_id`, `catalog_content` (title/description text)
- **Output:** `sample_id, price` for all items in `test.csv`
- **Goal:** minimize **SMAPE** between predictions and true prices

---

## 2) Dataset & Submission Format

Put raw CSVs (not committed to git) in the local `dataset/` directory:
dataset/
├─ train.csv # 75,000 rows: sample_id, catalog_content, price
└─ test.csv # 75,000 rows: sample_id, catalog_content


**Columns (train.csv)**

| column          | type    | notes                                  |
|-----------------|---------|----------------------------------------|
| sample_id       | int/str | unique row id                          |
| catalog_content | text    | title/description text                 |
| price           | float   | target to predict                      |

**Submission CSV**

sample_id,price
10000001,123.45
10000002, 89.99
...

We ensure all predicted prices are **positive** (clipped at `0.01`) and the row order follows `test.csv`.

---

## 3) Approach (High Level)

A fast, robust **text-only** pipeline that avoids leakage:

1. **Normalize text** (lowercase, collapse spaces).
2. Build two complementary feature families:
   - **TF-IDF n-grams** (word 1–2, char 3–5; ~300k dims).
   - **TextNumerics**: simple regex/engineered stats from text (lengths, digit counts/ranges, “pack × qty”, etc.).
3. Train **Level-0** models with grouped 5-fold CV:
   - **Ridge (L2)** on TF-IDF.
   - **XGBoost** on TextNumerics (trained on `log1p(price)`).
4. **Median-ratio calibration** in price space.
5. **Weighted blend** of the two Level-0 predictions → final submission.

### Architecture

```mermaid
flowchart LR
  A[train.csv / test.csv] --> B[Normalize Text]
  B --> C1[TF-IDF (word 1-2, char 3-5)]
  B --> C2[Text Numerics\nlen, words, digits, ranges,\nIPQ, pack×qty, qty_count]
  C1 --> D1[Ridge (L2)]
  C2 --> D2[XGBoost (hist)]
  subgraph CV[GroupKFold by text-hash]
    D1 --> OOF1[OOF Ridge preds]
    D2 --> OOF2[OOF XGB preds]
  end
  OOF1 --> E[Blend + Calibrate]
  OOF2 --> E
  E --> F[sub_ensemble-v1.csv]
```

4) Cross-Validation & Calibration

Grouping: group = SHA1(normalized_catalog_content)[:12]
Ensures near-duplicates stay in the same fold (no leakage).

Target space: train models on log1p(price); convert back with expm1.

Calibration: scale = median(train_price) / median(expm1(OOF_log)) applied to test predictions.

5) How to Reproduce
Environment (Windows PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

Place train.csv and test.csv into dataset/.

A) Build TF-IDF & train Ridge (saves a submission)
python - << "PY"
import os, numpy as np, pandas as pd, scipy.sparse as sp, hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

BASE="."; DATA=f"{BASE}/dataset"; FEAT=f"{BASE}/features/tfidf"; SUB=f"{BASE}/submissions"
os.makedirs(FEAT, exist_ok=True); os.makedirs(SUB, exist_ok=True)

def norm(s): s=s.fillna("").str.lower(); return s.str.replace(r"\s+"," ",regex=True).str.strip()
def groups(text): t=norm(text); return t.map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:12]).values

# load
train=pd.read_csv(f"{DATA}/train.csv"); test=pd.read_csv(f"{DATA}/test.csv")
tr=norm(train["catalog_content"]); te=norm(test["catalog_content"])

# TF-IDF
w=TfidfVectorizer(ngram_range=(1,2), max_features=200_000, min_df=2)
c=TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=100_000, min_df=2)
X_tr=sp.hstack([w.fit_transform(tr), c.fit_transform(tr)], format="csr")
X_te=sp.hstack([w.transform(te), c.transform(te)], format="csr")
sp.save_npz(f"{FEAT}/X_tr_tfidf.npz", X_tr); sp.save_npz(f"{FEAT}/X_te_tfidf.npz", X_te)
np.savez_compressed(f"{FEAT}/labels_ids.npz",
  y=train["price"].astype("float32").values,
  sid_tr=train["sample_id"].values, sid_te=test["sample_id"].values)

# CV Ridge
y=np.load(f"{FEAT}/labels_ids.npz")["y"]; sid_te=np.load(f"{FEAT}/labels_ids.npz")["sid_te"]
g=groups(train["catalog_content"]); gkf=GroupKFold(5)
oof=np.zeros_like(y,dtype="float32"); test_preds=[]
for f,(tr_idx,va_idx) in enumerate(gkf.split(X_tr,y,g)):
  m=Ridge(alpha=1.0, random_state=42)
  m.fit(X_tr[tr_idx], np.log1p(y[tr_idx]))
  oof[va_idx]=m.predict(X_tr[va_idx])
  test_preds.append(m.predict(X_te).astype("float32"))

pred=np.clip(np.expm1(np.mean(test_preds,0)), 0.01, None)
pd.DataFrame({"sample_id": sid_te, "price": pred}).to_csv(f"{SUB}/sub_tfidf_ridge.csv", index=False, float_format="%.6f")
print("Wrote:", f"{SUB}/sub_tfidf_ridge.csv")
PY

B) Train XGBoost on TextNumerics (saves a submission)

A minimal end-to-end runner is provided:
python src/models/run_textonly_baseline.py --data_dir dataset --out_dir artifacts --nfolds 5
# writes: submissions/sub_textnum_xgb.csv

C) Blend to the final file
python - << "PY"
import pandas as pd, numpy as np, os
BASE="."; SUB=f"{BASE}/submissions"; os.makedirs(SUB, exist_ok=True)
xgb=pd.read_csv(f"{SUB}/sub_textnum_xgb.csv")
rid=pd.read_csv(f"{SUB}/sub_tfidf_ridge.csv").set_index("sample_id").reindex(xgb["sample_id"]).reset_index()
w_xgb, w_ridge = 0.6, 0.4
ens = np.clip(w_xgb*xgb["price"] + w_ridge*rid["price"], 0.01, None)
pd.DataFrame({"sample_id": xgb["sample_id"], "price": ens}).to_csv(f"{SUB}/sub_ensemble-v1.csv", index=False, float_format="%.6f")
print("Wrote:", f"{SUB}/sub_ensemble-v1.csv")
PY

Submit submissions/sub_ensemble-v1.csv.

6) Repo Layout
amazon-ml-2025/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ submissions/
│  ├─ sub_textnum_xgb.csv
│  ├─ sub_tfidf_ridge.csv
│  └─ sub_ensemble-v1.csv   # final submitted
├─ src/
│  └─ models/
│     └─ run_textonly_baseline.py
├─ dataset/     # put train.csv & test.csv here (ignored by git)
├─ artifacts/ features/ meta/ lvl0/ logs/   # local-only (ignored)

7) Notes & Future Work

Replacing/augmenting TextNumerics with Sentence-Transformers or OpenCLIP embeddings (with TTA) is a natural extension.

A Level-1 meta-learner (e.g., LightGBM on OOF predictions) can further stabilize ranking.

Current pipeline is intentionally lightweight and reproducible on Colab/local CPU.


8) License

Released under the MIT License — see LICENSE
