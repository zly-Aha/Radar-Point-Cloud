"""Convenience entry point for the complete nine-model, seven-fold LOSO run."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

# Matches the nine-model comparison described in the revised manuscript.
ALL = ["stats_svm", "stats_rf", "stats_knn", "pointnet_gru", "pointnet_tcn", "robhar_like", "pct_gru", "dgcnn_gru", "radar_stnet"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset_path",default="Processed_Dataset_NPY"); ap.add_argument("--output_dir",default="experiment_loso_all_models"); ap.add_argument("--device",default="auto"); ap.add_argument("--num_workers",type=int,default=8); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--max_windows_per_file",type=int,default=0); ap.add_argument("--skip_clean_train",action="store_true"); ap.add_argument("--skip_degradation",action="store_true"); ap.add_argument("--run_realistic_clutter",action="store_true"); ap.add_argument("--realistic_mode",choices=["mixed","static","aid"],default="mixed"); ap.add_argument("--extra_args",nargs=argparse.REMAINDER); args=ap.parse_args()
    script=Path(__file__).with_name("run_loso_7fold.py"); cmd=[sys.executable,str(script),"--dataset_path",args.dataset_path,"--output_dir",args.output_dir,"--methods",*ALL,"--device",args.device,"--num_workers",str(args.num_workers),"--epochs",str(args.epochs),"--max_windows_per_file",str(args.max_windows_per_file),"--run_clutter","--run_realistic_clutter","--realistic_mode",args.realistic_mode]
    if args.skip_clean_train: cmd.append("--skip_clean_train")
    if args.skip_degradation: cmd.append("--skip_degradation")
    if args.extra_args: cmd.extend(args.extra_args)
    print("Executing:"," ".join(cmd)); raise SystemExit(subprocess.call(cmd,cwd=script.parent.parent))
if __name__=="__main__": main()
