import json, os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

USE_RAIN = True   # set False to run without rainfall

P = "data/processed/"
w = pd.read_csv(P + "features_wells.csv")
CAND = ["elev", "twi", "slope", "d_flood_km", "flooded",
        "d_waterway_km", "d_industry_km", "d_wastewater_km"] + (["rain_mm"] if USE_RAIN else [])
F = [c for c in CAND if c in w and w[c].notna().any() and w[c].nunique() > 1]
print("features used:", F)

med = w[F].median()
X, y = w[F].fillna(med).values, w["label_iron"].values
grp = (np.floor(w.lat / 0.15).astype(int) * 1000 + np.floor(w.lon / 0.15).astype(int)).values
print("wells:", len(w), "| positives:", int(y.sum()), "| spatial groups:", len(set(grp)))

def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced", random_state=0, n_jobs=-1)

oof = np.zeros(len(w))
for tr, te in GroupKFold(n_splits=min(5, len(set(grp)))).split(X, y, grp):
    oof[te] = rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

k = max(1, int(round(0.2 * len(w))))
top = np.argsort(-oof)[:k]
found = y[top].sum() / y.sum()
auc = roc_auc_score(y, oof)
print(f"top 20% ({k} wells) holds {found:.0%} of contaminated wells (random = 20%) | AUC {auc:.2f}")
json.dump({"n_wells": len(w), "n_pos": int(y.sum()), "top20_recall": float(found),
           "random": 0.2, "auc": float(auc), "features": F},
          open(P + "metrics.json", "w"), indent=2)

m = rf().fit(X, y)
print("importance:", dict(sorted(zip(F, m.feature_importances_.round(3)), key=lambda t: -t[1])))

def score(df):
    Z = df[F].fillna(med).values
    pr = np.array([t.predict_proba(Z)[:, 1] for t in m.estimators_])
    return pr.mean(0), pr.std(0)

wu = np.array([t.predict_proba(X)[:, 1] for t in m.estimators_]).std(0)
rows = [pd.DataFrame({"id": [f"well_{i}" for i in range(len(w))], "name": w.station,
                      "type": "well", "lat": w.lat, "lon": w.lon, "risk": oof, "uncertainty": wu})]
if os.path.exists(P + "features_places.csv"):
    p = pd.read_csv(P + "features_places.csv")
    r, u = score(p)
    rows.append(pd.DataFrame({"id": p.id, "name": p.name, "type": p.type,
                              "lat": p.lat, "lon": p.lon, "risk": r, "uncertainty": u}))
out = pd.concat(rows, ignore_index=True)
out["rank"] = out.risk.rank(ascending=False, method="first").astype(int)
pct, um = out.risk.rank(pct=True), out.uncertainty.median()
out["action"] = np.select(
    [(pct >= .9) & (out.uncertainty <= um), pct >= .9, pct >= .7, pct >= .4],
    ["Immediate testing", "Test and verify", "Priority testing", "Water testing"], "Monitoring")
out = out.sort_values("rank")[["id", "name", "type", "lat", "lon", "risk", "uncertainty", "rank", "action"]]
out.to_csv(P + "risk_scores.csv", index=False)
print(len(out), "rows saved to risk_scores.csv"); print(out.head(8).to_string())