"""Evaluate temporal-persistence clutter filtering before resampling.

The filter removes low-motion points that have spatial neighbours in at least
``min_persistence`` frames of the same window. Synthetic and structured clutter
are both supported; clean and unfiltered controls are always reported.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torch.utils.data import Dataset, DataLoader
SCRIPT_DIR=Path(__file__).resolve().parent; PROJECT_ROOT=SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from dataset_loader import DatasetConfig, Transforms
from experiment1_action_classification import build_deep_model, collect_records
from experiment2_degradation_robustness import DegradeConfig, DegradedWindowDataset
from experiment2_realistic_clutter import ClutterConfig, apply_realistic_clutter, make_trial_profile

def temporal_persistence_filter(frames, radius=0.035, min_persistence=4, max_motion=0.015):
    """Return frames with persistent, low-motion clutter points removed."""
    arr=[np.asarray(f,dtype=np.float32)[:,:4].copy() for f in frames]; out=[]
    for i, pts in enumerate(arr):
        if len(pts)==0: out.append(pts); continue
        xyz=pts[:,:3]; keep=np.ones(len(pts),dtype=bool)
        for j,pj in enumerate(arr):
            if i==j or len(pj)==0: continue
            d=((xyz[:,None,:]-pj[None,:,:3])**2).sum(axis=2)**0.5
            nearest=d.min(axis=1); near=np.where(nearest<=radius)[0]
            if len(near):
                # A point is considered static only when the nearest match is
                # close in every coordinate; aggregate match count below.
                pass
        counts=np.zeros(len(pts),dtype=np.int32); motion=np.zeros(len(pts),dtype=np.float32)
        for j,pj in enumerate(arr):
            if i==j or len(pj)==0: continue
            d=((xyz[:,None,:]-pj[None,:,:3])**2).sum(axis=2)**0.5
            nearest=d.min(axis=1); counts += (nearest<=radius); motion += nearest.astype(np.float32)
        mean_motion=motion/np.maximum(counts,1)
        remove=(counts>=min_persistence)&(mean_motion<=max_motion)
        out.append(pts[~remove])
    return out

class ClutterWindowDataset(Dataset):
    def __init__(self, records,cfg,seq_len,level,seed,mode,stride,max_windows,filtered,radius,min_persistence,max_motion):
        self.cfg,self.seq_len,self.level,self.seed,self.mode=cfg,seq_len,level,seed,mode; self.filtered=filtered
        self.radius,self.min_persistence,self.max_motion=radius,min_persistence,max_motion
        self.transform=Transforms(cfg,augment=False,deterministic=True); self.items=[]; self.chunks=[]; self.keys=[]; self.profiles=[]
        for rec in records:
            c=np.load(rec.path,allow_pickle=True)
            if len(c)<seq_len: continue
            fi=len(self.chunks); self.chunks.append(c); self.keys.append(f"p{rec.subject}:{rec.label_name}:{rec.path.name}"); self.profiles.append(make_trial_profile(fi,seed,cfg))
            starts=list(range(0,len(c)-seq_len+1,stride))
            if max_windows>0 and len(starts)>max_windows: starts=sorted(np.random.default_rng(2026+fi).choice(starts,max_windows,replace=False).tolist())
            self.items += [(fi,s,rec.label) for s in starts]
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        fi,start,label=self.items[idx]; raw=[]
        for li,p in enumerate(self.chunks[fi][start:start+self.seq_len]):
            rng=np.random.default_rng(self.seed+fi*10007+(start+li)*9176)
            if self.mode=="structured_realistic_clutter": q=apply_realistic_clutter(p,self.level,rng,self.profiles[fi],start+li,self.cfg,"mixed")
            else: q=p
            if self.mode=="synthetic_clutter":
                from experiment2_degradation_robustness import make_clutter; q=make_clutter(np.asarray(p),self.level,rng,self.cfg)
            raw.append(q)
        if self.filtered: raw=temporal_persistence_filter(raw,self.radius,self.min_persistence,self.max_motion)
        x=np.stack([self.transform.normalize(self.transform.resampler.resample(p)) for p in raw])
        return torch.tensor(x,dtype=torch.float32),torch.tensor(label),torch.tensor(fi)

def load_model(method,ckpt,classes,seq_len,device):
    try: state=torch.load(ckpt,map_location=device,weights_only=False)
    except TypeError: state=torch.load(ckpt,map_location=device)
    m=build_deep_model(method,len(classes),seq_len).to(device); m.load_state_dict(state.get("model_state",state) if isinstance(state,dict) else state); m.eval(); return m

@torch.no_grad()
def evaluate(model,loader,device,class_names):
    yt=[]; yp=[]; fids=[]
    for x,y,f in loader:
        p=model(x.to(device)).argmax(1).cpu().numpy(); yt += y.numpy().tolist(); yp += p.tolist(); fids += f.numpy().tolist()
    rec=recall_score(yt,yp,labels=list(range(len(class_names))),average=None,zero_division=0)
    return {"sample_accuracy":accuracy_score(yt,yp),"sample_macro_f1":f1_score(yt,yp,average="macro",zero_division=0),"sit_recall":rec[class_names.index("sit")] if "sit" in class_names else np.nan,"stand_recall":rec[class_names.index("stand")] if "stand" in class_names else np.nan,"n":len(yt)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset_path",default="Processed_Dataset_NPY"); ap.add_argument("--checkpoint_root",default="experiment1_results"); ap.add_argument("--output_dir",default="missing_frontend_filter"); ap.add_argument("--methods",nargs="+",default=["radar_stnet","pct_gru"]); ap.add_argument("--test_subjects",nargs="+",type=int,default=[7]); ap.add_argument("--levels",nargs="+",default=["0","0.1","0.2","0.3"]); ap.add_argument("--seeds",nargs="+",default=["0","1","2"]); ap.add_argument("--clutter_type",choices=["synthetic_clutter","structured_realistic_clutter"],default="structured_realistic_clutter"); ap.add_argument("--seq_len",type=int,default=25); ap.add_argument("--n_points",type=int,default=128); ap.add_argument("--window_stride",type=int,default=1); ap.add_argument("--max_windows_per_file",type=int,default=20); ap.add_argument("--batch_size",type=int,default=64); ap.add_argument("--num_workers",type=int,default=0); ap.add_argument("--device",default="auto"); ap.add_argument("--radius",type=float,default=.035); ap.add_argument("--min_persistence",type=int,default=4); ap.add_argument("--max_motion",type=float,default=.015); args=ap.parse_args()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device if args.device!="auto" else "cpu"); root=Path(args.dataset_path); root=root if root.is_absolute() else PROJECT_ROOT/root; out=Path(args.output_dir); out=out if out.is_absolute() else PROJECT_ROOT/out; out.mkdir(parents=True,exist_ok=True); checkpoint_root=Path(args.checkpoint_root); checkpoint_root=checkpoint_root if checkpoint_root.is_absolute() else (PROJECT_ROOT/checkpoint_root if (PROJECT_ROOT/checkpoint_root).exists() else Path.cwd()/checkpoint_root); records,classes=collect_records(str(root)); records=[r for r in records if r.subject in args.test_subjects]; cfg=DatasetConfig(dataset_path=str(root),n_chunk_per_data=args.seq_len,n_sample_per_chunk=args.n_points)
    rows=[]
    for method in args.methods:
        cands=sorted(checkpoint_root.rglob(f"*{method}.pth"));
        if not cands: print(f"[skip] missing checkpoint: {method}"); continue
        ckpt=next((p for p in cands if "train_p1-p5_val_p6_test_p7" in p.name),cands[-1]); model=load_model(method,ckpt,classes,args.seq_len,device)
        for level in [float(v) for v in args.levels]:
            for seed in ([int(args.seeds[0])] if level==0 else [int(v) for v in args.seeds]):
                for filtered in [False,True]:
                    ds=ClutterWindowDataset(records,cfg,args.seq_len,level,seed,args.clutter_type,args.window_stride,args.max_windows_per_file,filtered,args.radius,args.min_persistence,args.max_motion); met=evaluate(model,DataLoader(ds,batch_size=args.batch_size,num_workers=args.num_workers),device,classes); rows.append({"method":method,"level":level,"seed":seed,"filtered":filtered,**met})
    df=pd.DataFrame(rows); df.to_csv(out/"frontend_persistence_raw.csv",index=False,encoding="utf-8-sig");
    if not df.empty: df.groupby(["method","level","filtered"],as_index=False).agg({"sample_accuracy":"mean","sample_macro_f1":"mean","sit_recall":"mean","stand_recall":"mean"}).to_csv(out/"frontend_persistence_summary.csv",index=False,encoding="utf-8-sig")
    (out/"frontend_persistence_config.json").write_text(json.dumps(vars(args),ensure_ascii=False,indent=2),encoding="utf-8"); print(f"saved: {out.resolve()}")
if __name__=="__main__": main()
