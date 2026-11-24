"""Bankruptcy prediction on the UCI Polish companies data.

Usage: python scripts/train.py data/companies.csv results

1. For each horizon (1 to 5 years ahead) compare a fixed Altman Z' score, logistic
   regression and gradient boosting with repeated stratified 5-fold cross-validation.
2. For the 1 year horizon, hold out 25% as a test set and look at feature importance,
   calibration, a cost-based decision threshold, and whether X24 leaks future data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

SEED = 42
# X24 is described as "gross profit (in 3 years) / total assets". If that means profit in the
# NEXT three years, it uses information from after the prediction date, so it is left out.
LEAKY = ["X24"]
COST_MISSED, COST_FALSE_ALARM = 10, 1   # missing a bankruptcy costs 10x a false alarm

NAMES = {"X1": "net profit / total assets", "X2": "total liabilities / total assets",
         "X3": "working capital / total assets", "X4": "current assets / short-term liabilities",
         "X5": "cash runway in days", "X6": "retained earnings / total assets",
         "X7": "EBIT / total assets", "X8": "book equity / total liabilities",
         "X9": "sales / total assets", "X10": "equity / total assets",
         "X13": "(gross profit + depreciation) / sales", "X16": "(gross profit + depreciation) / total liabilities",
         "X21": "sales growth", "X22": "operating profit / total assets", "X24": "gross profit in 3 years / total assets",
         "X25": "(equity - share capital) / total assets", "X26": "(net profit + depreciation) / total liabilities",
         "X27": "operating profit / financial expenses", "X29": "log total assets",
         "X34": "operating expenses / total liabilities", "X35": "profit on sales / total assets",
         "X37": "(current assets - inventory) / long-term liabilities", "X39": "profit on sales / sales",
         "X40": "quick cash / short-term liabilities", "X41": "debt payback period",
         "X44": "receivable days", "X46": "(current assets - inventory) / short-term liabilities",
         "X56": "gross margin", "X58": "total costs / sales", "X61": "sales / receivables"}


def altman_z_private(X):
    """Altman Z' for private firms (1983 weights). Higher is safer, so return minus Z as risk."""
    f = X[["X3", "X6", "X7", "X8", "X9"]].fillna(X[["X3", "X6", "X7", "X8", "X9"]].median())
    z = 0.717 * f.X3 + 0.847 * f.X6 + 3.107 * f.X7 + 0.420 * f.X8 + 0.998 * f.X9
    return -z.to_numpy()


def logistic_model():
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                     ("rank", QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                                  random_state=SEED)),
                     ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])


def boosting_model():
    # handles missing values and extreme ratios natively, no scaling needed
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15,
                                          random_state=SEED)


def boosting_no_missing_signal():
    # fill every gap with the median first, so the model cannot learn from WHICH values are missing
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", boosting_model())])


def top10_capture(y, s):
    k = int(round(len(y) * 0.10))
    return y[np.argsort(-s, kind="stable")[:k]].sum() / y.sum()


def scores(y, s):
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "top10_capture": top10_capture(y, s)}


def oof_predict(model, X, y, cv):
    """Out-of-fold probabilities, averaged over the CV repeats."""
    total, count = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y[tr])
        total[te] += m.predict_proba(X.iloc[te])[:, 1]
        count[te] += 1
    return total / count


def by_horizon(df, feats):
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    for h, d in df.groupby("years_ahead"):
        X, y = d[feats], d.bankrupt.to_numpy()
        res = {"Altman Z' (no training)": altman_z_private(X),
               "Logistic regression": oof_predict(logistic_model(), X, y, cv),
               "Gradient boosting": oof_predict(boosting_model(), X, y, cv),
               "Gradient boosting, gaps filled": oof_predict(boosting_no_missing_signal(), X, y, cv),
               "Missing pattern only": oof_predict(boosting_model(), X.isna().astype(int), y, cv)}
        for name, s in res.items():
            rows.append({"years_ahead": h, "model": name, "n": len(y), "bankrupt": int(y.sum()),
                         "base_rate": y.mean(), **scores(y, s)})
        print(f"horizon {h}: done")
    return pd.DataFrame(rows)


def one_year_deep_dive(df, feats, out):
    d = df[df.years_ahead == 1].reset_index(drop=True)
    train, test = train_test_split(d, test_size=0.25, stratify=d.bankrupt, random_state=SEED)
    X_tr, y_tr = train[feats], train.bankrupt.to_numpy()
    X_te, y_te = test[feats], test.bankrupt.to_numpy()
    res = {}

    gb = boosting_model().fit(X_tr, y_tr)
    p_te = gb.predict_proba(X_te)[:, 1]
    res["test"] = {"Altman Z' (no training)": scores(y_te, altman_z_private(X_te)),
                   "Logistic regression": scores(y_te, logistic_model().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]),
                   "Gradient boosting": scores(y_te, p_te)}
    res["test_size"] = {"n": len(y_te), "bankrupt": int(y_te.sum())}

    res["test"]["Gradient boosting, gaps filled"] = scores(
        y_te, boosting_no_missing_signal().fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
    res["missing_by_class"] = (train.groupby("bankrupt")[["X21", "X27"]]
                               .apply(lambda g: g.isna().mean()).round(3).to_dict())

    # Does X24 leak? Train the same model with it and compare.
    with24 = feats + LEAKY
    p24 = boosting_model().fit(train[with24], y_tr).predict_proba(test[with24])[:, 1]
    res["with_X24"] = scores(y_te, p24)

    # Calibration: are the probabilities believable?
    res["brier"] = {"gradient_boosting": brier_score_loss(y_te, p_te),
                    "always_base_rate": brier_score_loss(y_te, np.full(len(y_te), y_tr.mean()))}
    bins = pd.qcut(p_te, 10, duplicates="drop")
    calib = (pd.DataFrame({"bin": bins, "p": p_te, "y": y_te}).groupby("bin", observed=True)
             .agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size")))
    calib.to_csv(out / "calibration_1y.csv")

    # Cost-based threshold, chosen on out-of-fold training predictions (never on the test set)
    oof = oof_predict(boosting_model(), X_tr, y_tr, StratifiedKFold(5, shuffle=True, random_state=SEED))
    grid = np.unique(np.quantile(oof, np.linspace(0.5, 0.995, 200)))
    def cost(y, p, t):
        tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
        return COST_MISSED * fn + COST_FALSE_ALARM * fp
    best_t = min(grid, key=lambda t: cost(y_tr, oof, t))
    tn, fp, fn, tp = confusion_matrix(y_te, p_te >= best_t, labels=[0, 1]).ravel()
    tn5, fp5, fn5, tp5 = confusion_matrix(y_te, p_te >= 0.5, labels=[0, 1]).ravel()
    res["threshold"] = {
        "chosen": float(best_t),
        "at_chosen": {"caught": int(tp), "missed": int(fn), "false_alarms": int(fp),
                      "cost": int(COST_MISSED * fn + COST_FALSE_ALARM * fp)},
        "at_0.5": {"caught": int(tp5), "missed": int(fn5), "false_alarms": int(fp5),
                   "cost": int(COST_MISSED * fn5 + COST_FALSE_ALARM * fp5)},
        "flag_nobody_cost": int(COST_MISSED * y_te.sum())}

    imp = permutation_importance(gb, X_te, y_te, scoring="average_precision", n_repeats=10,
                                 random_state=SEED, n_jobs=-1)
    importance = (pd.DataFrame({"feature": feats, "meaning": [NAMES.get(f, "") for f in feats],
                                "pr_auc_drop": imp.importances_mean, "std": imp.importances_std})
                  .sort_values("pr_auc_drop", ascending=False))
    importance.to_csv(out / "importance_1y.csv", index=False)
    res["top_features"] = importance.head(8).to_dict("records")
    return res


def chart(table, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=150)
    style = {"Gradient boosting": "#2a6fdb", "Gradient boosting, gaps filled": "#7fa8ea",
             "Logistic regression": "#e07b39", "Missing pattern only": "#a855f7",
             "Altman Z' (no training)": "#6b7280"}
    for name, color in style.items():
        t = table[table.model == name].sort_values("years_ahead")
        ax.plot(t.years_ahead, t.pr_auc, marker="o", ms=6, lw=2, color=color, label=name)
    base = table[table.model == "Gradient boosting"].sort_values("years_ahead")
    ax.plot(base.years_ahead, base.base_rate, ls="--", lw=1.2, color="#9ca3af", label="Random")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Forecast horizon (years ahead)")
    ax.set_ylabel("PR-AUC (cross-validated)")
    ax.set_ylim(0, 1)
    ax.set_title("Bankruptcy prediction by horizon", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e5e5e5", lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(0.01, 0.01, "UCI Polish companies bankruptcy data, 5-fold CV repeated twice",
             fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "pr_auc_by_horizon.png")


def main(data="data/companies.csv", out_dir="results"):
    out = Path(out_dir)
    df = pd.read_csv(data)
    feats = [c for c in df.columns if c.startswith("X") and c not in LEAKY]

    table = by_horizon(df, feats)
    table.to_csv(out / "by_horizon.csv", index=False)
    chart(table, out)
    deep = one_year_deep_dive(df, feats, out)
    (out / "one_year_results.json").write_text(json.dumps(deep, indent=2, default=float))

    pivot = table.pivot(index="years_ahead", columns="model", values="pr_auc").round(3)
    print(pivot)
    print(table.pivot(index="years_ahead", columns="model", values="roc_auc").round(3))
    print(table.pivot(index="years_ahead", columns="model", values="top10_capture").round(3))
    print(json.dumps(deep, indent=2, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
