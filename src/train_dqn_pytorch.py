"""Canonical PyTorch DQN for contextual RF/FSO/THz channel selection."""
from pathlib import Path
import random, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'canonical'/'dataset_all_scenarios.csv'
OUT=ROOT/'results'/'dqn'; MODELS=ROOT/'models'/'dqn'
OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
ENV=['distance_m','rain_rate_mmph','humidity_pct','visibility_km','Cn2']
FULL=ENV+['SNR_RF_dB','SNR_FSO_dB','SNR_THz_dB','Eb_RF_J_per_bit','Eb_FSO_J_per_bit','Eb_THz_J_per_bit']
FEATURES={'environmental':ENV,'full':FULL}; SEEDS=[11,22,33,44,55]; LABELS=[0,1,2]
LR=1e-3; BATCH=64; BUFFER=50000; START=1000; TRAIN_FREQ=4; TOTAL=100000
EPS_START=1.; EPS_END=.05; EPS_FRAC=.20

class QNet(nn.Module):
    def __init__(self,n):
        super().__init__(); self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,x): return self.net(x)

def split(df,seed):
    rng=np.random.default_rng(seed); test=np.zeros(len(df),bool); sc=df.scenario.to_numpy(); rr=df.realization_id.to_numpy()
    for s in sorted(df.scenario.unique()):
        ids=np.array(sorted(df.loc[df.scenario==s,'realization_id'].unique())); c=rng.choice(ids,2,replace=False)
        test|=(sc==s)&np.isin(rr,c)
    return ~test,test

def reward_matrix(frame):
    ref=frame.best_channel.astype(int).to_numpy(); feas=np.column_stack([frame.feasible_RF,frame.feasible_FSO,frame.feasible_THz]).astype(bool)
    eb=np.column_stack([frame.Eb_RF_J_per_bit,frame.Eb_FSO_J_per_bit,frame.Eb_THz_J_per_bit]).astype(float)
    R=np.zeros((len(frame),3),np.float32); anyf=feas.any(1)
    for i in range(len(frame)):
        if anyf[i]:
            emin=max(float(np.min(eb[i,feas[i]])),1e-30)
            for a in range(3):
                R[i,a]=-1. if not feas[i,a] else .5*np.clip(emin/max(float(eb[i,a]),1e-30),0,1)+(1. if a==ref[i] else 0.)
        else:
            R[i,:]=-.5; R[i,ref[i]]=1.
    return R,feas,anyf,ref

def eps(step):
    d=int(TOTAL*EPS_FRAC); return EPS_END if step>=d else EPS_START+(EPS_END-EPS_START)*(step/d)

def metrics(y,p,R,feas,anyf):
    rew=R[np.arange(len(y)),p]; infeas=anyf & (~feas[np.arange(len(y)),p])
    return dict(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,labels=LABELS,average='macro',zero_division=0),
        macro_precision=precision_score(y,p,labels=LABELS,average='macro',zero_division=0),
        macro_recall=recall_score(y,p,labels=LABELS,average='macro',zero_division=0),mean_reward=float(rew.mean()),
        infeasible_action_rate=float(infeas.mean()),feasible_only_accuracy=accuracy_score(y[anyf],p[anyf]),
        fallback_accuracy=accuracy_score(y[~anyf],p[~anyf]) if (~anyf).any() else np.nan)

def main():
    df=pd.read_csv(DATA); rows=[]; cms={}
    for mode,feats in FEATURES.items():
      for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); tr,te=split(df,seed)
        train=df.loc[tr].reset_index(drop=True); test=df.loc[te].reset_index(drop=True)
        mu=train[feats].mean().to_numpy(float); sd=train[feats].std(ddof=0).replace(0,1).to_numpy(float)
        Xtr=((train[feats].to_numpy(float)-mu)/sd).astype(np.float32); Xte=((test[feats].to_numpy(float)-mu)/sd).astype(np.float32)
        Rtr,_,_,_=reward_matrix(train); Rte,fte,anyf,y=reward_matrix(test)
        q=QNet(len(feats)); opt=optim.Adam(q.parameters(),lr=LR); loss_fn=nn.SmoothL1Loss(); rng=np.random.default_rng(seed)
        bi=np.empty(BUFFER,np.int32); ba=np.empty(BUFFER,np.int64); br=np.empty(BUFFER,np.float32); bn=bp=0; t0=time.perf_counter()
        for step in range(TOTAL):
            idx=int(rng.integers(0,len(train)))
            if rng.random()<eps(step): a=int(rng.integers(0,3))
            else:
                with torch.no_grad(): a=int(q(torch.from_numpy(Xtr[idx:idx+1])).argmax(1).item())
            bi[bp]=idx; ba[bp]=a; br[bp]=Rtr[idx,a]; bp=(bp+1)%BUFFER; bn=min(bn+1,BUFFER)
            if step>=START and step%TRAIN_FREQ==0:
                ids=rng.integers(0,bn,size=BATCH); s=torch.from_numpy(Xtr[bi[ids]]); at=torch.from_numpy(ba[ids]); rt=torch.from_numpy(br[ids])
                loss=loss_fn(q(s).gather(1,at[:,None]).squeeze(1),rt); opt.zero_grad(); loss.backward(); opt.step()
        train_time=time.perf_counter()-t0
        with torch.no_grad(): pred=q(torch.from_numpy(Xte)).argmax(1).numpy()
        rows.append({'state_mode':mode,'seed':seed,'policy':'DQN',**metrics(y,pred,Rte,fte,anyf),'training_time_s':train_time})
        rngb=np.random.default_rng(seed+10000)
        for name,p in [('AnalyticalReference',y.copy()),('Random',rngb.integers(0,3,len(y))),('MostFrequentClass',np.full(len(y),int(train.best_channel.mode().iloc[0])) )]:
            rows.append({'state_mode':mode,'seed':seed,'policy':name,**metrics(y,p,Rte,fte,anyf),'training_time_s':0.})
        cms.setdefault(mode,np.zeros((3,3),int)); cms[mode]+=confusion_matrix(y,pred,labels=LABELS)
        torch.save({'state_dict':q.state_dict(),'features':feats,'mean':mu,'std':sd,'seed':seed},MODELS/f'dqn_{mode}_seed{seed}.pt')
        print(mode,seed,rows[-4]['accuracy'],rows[-4]['macro_f1'])
    per=pd.DataFrame(rows); per.to_csv(OUT/'dqn_pytorch_per_run.csv',index=False)
    summ=per.groupby(['state_mode','policy'],as_index=False).agg(accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),mean_reward_mean=('mean_reward','mean'),infeasible_action_rate_mean=('infeasible_action_rate','mean'),feasible_only_accuracy_mean=('feasible_only_accuracy','mean'),fallback_accuracy_mean=('fallback_accuracy','mean'),training_time_mean_s=('training_time_s','mean'))
    summ.to_csv(OUT/'dqn_pytorch_summary.csv',index=False)
    for mode,cm in cms.items(): pd.DataFrame(cm,index=['true_RF','true_FSO','true_THz'],columns=['pred_RF','pred_FSO','pred_THz']).to_csv(OUT/f'confusion_{mode}_DQN.csv')

if __name__=='__main__': main()
