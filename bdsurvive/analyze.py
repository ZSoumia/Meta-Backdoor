"""Analysis: turn results.jsonl into the claims.

survival = descendant asr_adj / parent asr_adj, but reported alongside the
retained-utility frontier so a broken student can't masquerade as erasure.
"""
import json
import pandas as pd


def load(results_path):
    rows = [json.loads(l) for l in open(results_path)]
    return pd.DataFrame(rows)


def parent_table(df):
    p = df[df.kind == "parent"].set_index("parent_id")
    return p[["asr_adj", "clean_acc", "expression_kl",
              "placement_depth"]].rename(columns=lambda c: f"parent_{c}")


def survival_table(df):
    """One row per descendant with relative survival + parent coordinates."""
    pt = parent_table(df)
    d = df[df.kind.isin(["descendant", "lineage"])].copy()
    d = d.join(pt, on="parent_id")
    d["survival"] = d["asr_adj"] / d["parent_asr_adj"].clip(lower=1e-6)
    d["survival"] = d["survival"].clip(0, 2)   # >1 == amplification, capped
    d["utility_kept"] = d["clean_acc"]
    return d


def crossing_test(df):
    """The Pilot-2 load-bearing check.

    Expect: localized+rare+high-kl parent survives FT, dies under KD.
            diffuse+semantic+kl0 parent erodes under FT, carries under KD.
    Returns a small pivot of mean survival by (parent placement, method).
    """
    d = survival_table(df)
    d["arm"] = d["method"].map(lambda m: "FT" if "ft" in m else "KD")
    return d.pivot_table(index=["placement", "support", "kl_lambda"],
                         columns="arm", values="survival", aggfunc="mean")


def markov_check(df):
    """Does gen-(k+1) survival depend on gen-k COORDINATES alone, ignoring path?

    Regress next-gen asr_adj on current-gen (expression_kl, placement_depth,
    asr_adj) within lineage runs; report R^2.  High R^2 => coordinates are a
    sufficient state and the full path tree is unnecessary.
    """
    lin = df[df.kind == "lineage"].sort_values(["parent_id", "lineage", "generation"])
    recs = []
    for (_, _), g in lin.groupby(["parent_id", "lineage"]):
        g = g.sort_values("generation").reset_index(drop=True)
        for i in range(len(g) - 1):
            recs.append(dict(
                cur_kl=g.loc[i, "expression_kl"],
                cur_depth=g.loc[i, "placement_depth"],
                cur_asr=g.loc[i, "asr_adj"],
                next_asr=g.loc[i + 1, "asr_adj"]))
    if len(recs) < 5:
        return None
    m = pd.DataFrame(recs).dropna()
    try:
        import numpy as np
        X = m[["cur_kl", "cur_depth", "cur_asr"]].values
        y = m["next_asr"].values
        X1 = np.column_stack([X, np.ones(len(X))])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        pred = X1 @ beta
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return dict(r2=1 - ss_res / max(ss_tot, 1e-9), n=len(m), beta=beta.tolist())
    except Exception as e:
        return dict(error=str(e))


def expression_screen_corr(df):
    """Does parent expression_kl predict KD survival?  Spearman."""
    d = survival_table(df)
    kd = d[d.method.str.contains("kd")]
    if len(kd) < 4:
        return None
    return kd[["parent_expression_kl", "survival"]].corr(method="spearman").iloc[0, 1]
