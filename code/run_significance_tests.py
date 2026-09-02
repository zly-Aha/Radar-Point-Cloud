"""Fold-level significance tests and bootstrap confidence intervals."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fold_csv",default="experiment_loso_7fold_results/loso_degradation_auc_by_fold.csv"); ap.add_argument("--output_dir",default="missing_significance"); ap.add_argument("--metric",default="auc"); ap.add_argument("--bootstrap",type=int,default=5000); args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(args.fold_csv)
    rows=[]; rng=np.random.default_rng(42)
    for deg,g in df.groupby("degradation"):
        piv=g.pivot_table(index="fold",columns="method",values=args.metric,aggfunc="mean").dropna(axis=1,how="all"); methods=list(piv.columns)
        for m in methods:
            v=piv[m].dropna().to_numpy(float); boot=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(args.bootstrap)]) if len(v) else np.array([]); rows.append({"degradation":deg,"method":m,"n_folds":len(v),"mean":np.mean(v) if len(v) else np.nan,"ci95_low":np.quantile(boot,.025) if len(boot) else np.nan,"ci95_high":np.quantile(boot,.975) if len(boot) else np.nan})
        try:
            from scipy.stats import friedmanchisquare, wilcoxon
            complete=piv.dropna();
            if complete.shape[1]>=3 and len(complete)>=2:
                stat,p=friedmanchisquare(*[complete[m] for m in complete.columns]); rows.append({"degradation":deg,"test":"friedman","statistic":stat,"p_value":p})
            if len(complete)>=2:
                ref=complete.columns[-1]
                for m in complete.columns[:-1]:
                    stat,p=wilcoxon(complete[m],complete[ref],zero_method="wilcox",alternative="two-sided"); rows.append({"degradation":deg,"test":f"wilcoxon:{m}_vs_{ref}","statistic":stat,"p_value":p})
        except ImportError: pass
    pd.DataFrame(rows).to_csv(out/"significance_results.csv",index=False,encoding="utf-8-sig"); print(f"saved: {out.resolve()}")
if __name__=="__main__": main()
