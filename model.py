import json, os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score

P = "data/processed/"
TEST_SIZE = 0.20      # static 80/20 split
SPLIT_SEED = 42       # fixed -> the same wells are held out every run
BLOCK_DEG = 0.15      # wells in the same ~16 km block stay together (no spatial leakage)

w = pd.read_csv(P + "features_wells.csv")
w["lat_r"], w["lon_r"] = w.lat.round(4), w.lon.round(4)

# ---- other contaminant labels (made by make_labels.py); iron stays the main target ----
LAB = P + "labels_multi.csv"
if os.path.exists(LAB):
    lab = pd.read_csv(LAB).drop(columns=["label_iron"], errors="ignore")
    lab = lab.drop_duplicates(["lat_r", "lon_r"])
    n0 = len(w)
    w = w.merge(lab, on=["lat_r", "lon_r"], how="left")
    assert len(w) == n0, "duplicate coordinates in labels_multi.csv"
    others = [c for c in lab.columns if c.startswith("label_")]
else:
    print("labels_multi.csv not found: running iron only (run src\\make_labels.py to add more)")
    others = []
TARGETS = [("iron", "label_iron")] + [(c[6:], c) for c in others]

# ---- features ----
CAND = ["elev", "twi", "slope", "d_flood_km", "flooded",
        "d_waterway_km", "d_industry_km", "d_wastewater_km", "rain_mm"]
F = [c for c in CAND if c in w and w[c].notna().any() and w[c].nunique() > 1]
print("features used:", F)
med = w[F].median()
Xall = w[F].fillna(med).values
lat_all, lon_all = w.lat.values, w.lon.values
print("wells:", len(w), "| iron positives:", int(w["label_iron"].sum()))

p = pd.read_csv(P + "features_places.csv") if os.path.exists(P + "features_places.csv") else None
if p is not None:
    for c in F:
        if c not in p.columns:
            print("WARNING: places file has no", c, "- filled with the wells' median")
            p[c] = np.nan
    Xp = p[F].fillna(med).values

def rf(seed=0):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)

def _tree_p(t, Z):
    pr = t.predict_proba(Z)
    return pr[:, list(t.classes_).index(1)] if 1 in t.classes_ else np.zeros(len(Z))

def tree_probs(m, Z):
    return np.array([_tree_p(t, Z) for t in m.estimators_])

def oob_probs(m, X):
    """Out-of-bag mean/std: each training well is scored only by trees that never saw it."""
    n = len(X)
    P_ = np.full((len(m.estimators_), n), np.nan)
    for i, (t, s) in enumerate(zip(m.estimators_, m.estimators_samples_)):
        oob = np.ones(n, bool); oob[s] = False
        if oob.any():
            P_[i, oob] = _tree_p(t, X[oob])
    return np.nanmean(P_, 0), np.nanstd(P_, 0)

def static_split(y, g):
    """Fixed 80/20 split by spatial block; first seed (from SPLIT_SEED) giving both classes in test."""
    for s in range(SPLIT_SEED, SPLIT_SEED + 100):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=s).split(y, y, g))
        if y[te].sum() >= 2 and (len(te) - y[te].sum()) >= 2 and 0 < y[tr].sum() < len(tr):
            return tr, te, s
    return None, None, None

def run_target(col):
    mask = w[col].notna().values
    y = w.loc[mask, col].astype(int).values
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos < 5 or nneg < 5:
        return None, npos, nneg
    X, lat, lon = Xall[mask], lat_all[mask], lon_all[mask]
    g = np.floor(lat / BLOCK_DEG).astype(int) * 1000 + np.floor(lon / BLOCK_DEG).astype(int)
    tr, te, used_seed = static_split(y, g)
    if tr is None:
        return None, npos, nneg

    # --- train on the 80 %, evaluate once on the held-out 20 % ---
    m = rf(0).fit(X[tr], y[tr])
    pt = tree_probs(m, X[te])
    p_te, u_te = pt.mean(0), pt.std(0)
    k = max(1, int(round(0.2 * len(te))))
    recall20 = float(y[te][np.argsort(-p_te)[:k]].sum() / y[te].sum())
    auc = float(roc_auc_score(y[te], p_te))

    # --- scores for the ranking ---
    final = rf(0).fit(X, y)                     # all labelled wells -> unlabelled wells + places
    pw = tree_probs(final, Xall)
    risk_w, unc_w = pw.mean(0), pw.std(0)
    ri, ui = np.zeros(len(y)), np.zeros(len(y))
    ri[tr], ui[tr] = oob_probs(m, X[tr])        # train wells: out-of-bag (not in-sample)
    ri[te], ui[te] = p_te, u_te                 # test wells: genuinely held out
    risk_w[mask], unc_w[mask] = ri, ui
    risk, unc = risk_w, unc_w
    if p is not None:
        pp = tree_probs(final, Xp)
        risk, unc = np.concatenate([risk_w, pp.mean(0)]), np.concatenate([unc_w, pp.std(0)])
    met = {"n_wells": int(len(y)), "n_pos": npos, "pos_rate": round(npos / len(y), 3),
           "split": f"static {int(100 - TEST_SIZE * 100)}/{int(TEST_SIZE * 100)} by {BLOCK_DEG} deg block",
           "split_seed": int(used_seed), "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_test_pos": int(y[te].sum()),
           "top20_recall": round(recall20, 3), "random": 0.2, "auc": round(auc, 3)}
    return (risk, unc, final, met), npos, nneg

def repeated_cv(col, seeds=5, offsets=(0.0, 0.05, 0.10)):
    """15 repeated spatial CV runs: the more stable number to quote."""
    mask = w[col].notna().values
    y = w.loc[mask, col].astype(int).values
    X, lat, lon = Xall[mask], lat_all[mask], lon_all[mask]
    k = max(1, int(round(0.2 * len(y))))
    runs = []
    for seed in range(seeds):
        for off in offsets:
            g = (np.floor((lat + off) / BLOCK_DEG).astype(int) * 1000 +
                 np.floor((lon + off) / BLOCK_DEG).astype(int))
            o = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(X, y, g):
                if len(set(y[tr])) < 2:
                    o[te] = y[tr].mean(); continue
                o[te] = tree_probs(rf(seed).fit(X[tr], y[tr]), X[te]).mean(0)
            runs.append((y[np.argsort(-o)[:k]].sum() / y.sum(), roc_auc_score(y, o)))
    rec, auc = np.array(runs).T
    return {"n_runs": len(runs), "top20_recall_mean": round(float(rec.mean()), 3),
            "top20_recall_sd": round(float(rec.std()), 3), "random": 0.2,
            "auc_mean": round(float(auc.mean()), 3), "auc_sd": round(float(auc.std()), 3)}

base = [pd.DataFrame({"id": [f"well_{i}" for i in range(len(w))], "name": w.station,
                      "type": "well", "lat": w.lat, "lon": w.lon})]
if p is not None:
    base.append(pd.DataFrame({"id": p.id.astype(str), "name": p.name, "type": p.type,
                              "lat": p.lat, "lon": p.lon}))
out = pd.concat(base, ignore_index=True)

metrics = {"features": F}
for t, col in TARGETS:
    res, npos, nneg = run_target(col)
    if res is None:
        print(f"{t}: skipped (positives={npos}, negatives={nneg}, or no valid split)"); continue
    risk, unc, final, met = res
    print(f"{t:<10} wells={met['n_wells']} pos={npos} ({met['pos_rate']:.0%}) | "
          f"train {met['n_train']} / test {met['n_test']} ({met['n_test_pos']} pos) | "
          f"top 20% holds {met['top20_recall']:.0%} (random 20%) | AUC {met['auc']:.2f}")
    if t == "iron":
        out["risk"], out["uncertainty"] = risk, unc
        metrics.update(met)
        print("importance (iron):", {a: float(b) for a, b in
              sorted(zip(F, final.feature_importances_.round(3)), key=lambda z: -z[1])})
    else:
        out["risk_" + t], out["unc_" + t] = risk.round(4), unc.round(4)
        metrics.setdefault("other_contaminants", {})[t] = met

cv = repeated_cv("label_iron")
metrics["iron_repeated_cv"] = cv
print(f"iron repeated spatial CV ({cv['n_runs']} runs): top 20% holds {cv['top20_recall_mean']:.0%} "
      f"+/- {cv['top20_recall_sd']:.0%} (random 20%) | AUC {cv['auc_mean']:.2f} +/- {cv['auc_sd']:.2f}")

assert "risk" in out, "iron model was skipped, cannot build the main ranking"
out["rank"] = out.risk.rank(ascending=False, method="first").astype(int)
out["rank_in_type"] = out.groupby("type").risk.rank(ascending=False, method="first").astype(int)
pct, um = out.risk.rank(pct=True), out.uncertainty.median()
out["action"] = np.select(
    [(pct >= .9) & (out.uncertainty <= um), pct >= .9, pct >= .7, pct >= .4],
    ["Immediate testing", "Test and verify", "Priority testing", "Water testing"], "Monitoring")
out[["risk", "uncertainty"]] = out[["risk", "uncertainty"]].round(4)
extra_cols = [c for c in out.columns if c.startswith(("risk_", "unc_"))]
out = out.sort_values("rank")[["id", "name", "type", "lat", "lon", "risk", "uncertainty",
                               "rank", "rank_in_type", "action"] + extra_cols]
out.to_csv(P + "risk_scores_full.csv", index=False)
json.dump(metrics, open(P + "metrics.json", "w"), indent=2)
print(len(out), "rows saved to risk_scores_full.csv |", len(TARGETS), "contaminants attempted")
print(out.head(8)[["id", "name", "type", "risk", "uncertainty", "rank", "action"]].to_string())
