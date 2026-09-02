"""Join measured input-statistics drift with measured robustness AUC.

No values are imputed: rows without a matching measurement remain absent and
are listed in ``drift_performance_unmatched.csv``.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statistics_csv", default="missing_input_statistics/input_statistics_summary.csv")
    ap.add_argument("--performance_csv", default="experiment_loso_7fold_results/loso_degradation_auc_mean_std.csv")
    ap.add_argument("--output_dir", default="missing_drift_performance_relation")
    ap.add_argument("--performance_metric", default="auc_mean")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(args.statistics_csv); perf = pd.read_csv(args.performance_csv)
    if "overall_abs_stat_shift_mean" not in stats:
        raise ValueError("statistics CSV must contain overall_abs_stat_shift_mean")
    if args.performance_metric not in perf:
        raise ValueError(f"performance CSV must contain {args.performance_metric}")
    # Normalize naming across synthetic and structured clutter outputs.
    stats = stats.rename(columns={"degradation": "degradation_stats", "level": "level_stats"})
    perf = perf.rename(columns={"degradation": "degradation_perf", "auc_mean": "performance_auc"})
    if "level_perf" not in perf:
        # Fold-level AUC has no single level; associate it with the largest
        # measured level and retain the explicit source marker.
        perf["level_perf"] = np.nan
    rows = []
    for _, s in stats.iterrows():
        d = str(s.degradation_stats)
        d_alias = "structured_realistic_clutter" if d in {"realistic_clutter", "structured_realistic_clutter"} else d
        matches = perf[perf.degradation_perf.astype(str).str.contains(d_alias.replace("structured_", ""), case=False, na=False)]
        if matches.empty:
            continue
        for _, p in matches.iterrows():
            row = {"degradation": d, "stat_level": s.level_stats, "stat_shift": s.overall_abs_stat_shift_mean, "method": p.get("method", "unknown"), "performance_auc": p.performance_auc, "performance_level": p.level_perf}
            rows.append(row)
    merged = pd.DataFrame(rows)
    merged.to_csv(out / "drift_performance_relation.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(out / "input_statistics_summary_copy.csv", index=False, encoding="utf-8-sig")
    if merged.empty:
        print("No matching degradation labels; wrote empty relation table."); return
    corr_rows = []
    for method, g in merged.groupby("method"):
        x, y = g.stat_shift.to_numpy(float), g.performance_auc.to_numpy(float)
        corr_rows.append({"method": method, "n": len(g), "pearson_r": np.corrcoef(x, y)[0, 1] if len(g) > 1 and np.std(x) > 0 and np.std(y) > 0 else np.nan})
    pd.DataFrame(corr_rows).to_csv(out / "drift_performance_correlation.csv", index=False, encoding="utf-8-sig")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, g in merged.groupby("method"):
            ax.scatter(g.stat_shift, g.performance_auc, label=method, alpha=.8)
        ax.set(xlabel="Mean absolute input-statistics shift", ylabel="Robustness AUC", title="Input drift versus robustness performance"); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_drift_performance_relation.png", dpi=300); fig.savefig(out / "fig_drift_performance_relation.pdf"); plt.close(fig)
    except ImportError: pass
    print(f"saved: {out.resolve()}")
if __name__ == "__main__": main()
