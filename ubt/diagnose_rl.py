import argparse,copy,json,os,hashlib
from pathlib import Path
import numpy as np
import torch
from paths import DATA as SOURCE,ROOT,RUNS
from runner import load_actor,supervised_loss,batch,make_reward,save
from ubt_bapo_relations import policy_loss
from rl_clean import TaskReward,sample_group,integrated_loss,grpo_loss,balanced_rows

OUT=RUNS/'beauty_rl'
BASE=RUNS

def source_for(ds):
    from torch._subclasses.fake_tensor import FakeTensorMode
    options=[]
    for variant,path in [('full',BASE/ds/'stage1.pt')]:
        with FakeTensorMode():v=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        options.append((float(v['score']),variant,path))
    score,variant,path=max(options,key=lambda x:x[0])
    return path,variant,score

def grad_vector(actor,loss):
    params=[actor.model.gpt2.wte.weight,actor.model.main_mlp[0].weight,
            actor.model.main_mlp[2].weight,actor.model.mu_out.weight]
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,grads)])

def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2024)
    ds=a.dataset;folder=OUT/ds;folder.mkdir(parents=True,exist_ok=True)
    data=dict(np.load(SOURCE/ds/'assets.npz',allow_pickle=False));path,variant,score=source_for(ds)
    actor,_=load_actor(data,3,path,448);actor.eval();old=make_reward(data)
    exact=TaskReward(torch.as_tensor(data['codes'],device='cuda'))
    sid=TaskReward(torch.as_tensor(data['codes'],device='cuda'),.1)
    frozen=copy.deepcopy(actor).requires_grad_(False).eval()
    support=TaskReward(torch.as_tensor(data['codes'],device='cuda'),support_reference=frozen)
    rank=TaskReward(torch.as_tensor(data['codes'],device='cuda'),support_reference=frozen,support_mode='rank')
    rows=balanced_rows(data,2024);records=[]
    for i in range(4):
        b=batch(data,'train',rows[i*16:(i+1)*16])
        supervised,sd=supervised_loss(actor,b,align=variant!='noalign',null=True)
        sg=grad_vector(actor,supervised)
        rec={'batch':i,'supervised_norm':float(sg.norm()),'estimators':{}}
        old_loss,detail=policy_loss(actor,b,old,'integrated',2024+i,group=16,return_tensors=True)
        t=detail['tensors'];values=t['sampled'].sum(-1)+t['terminal'];hit=(t['actions']==(b['set_targets']-1)[:,None]).float().mean(-1)
        corr=np.corrcoef(values.detach().cpu().flatten(),hit.cpu().flatten())[0,1] if float(hit.std())>0 else None
        og=grad_vector(actor,old_loss)
        rec['old_reward']={'mean':float(values.mean()),'std':float(values.std()),'actual_hit':float(hit.mean()),'reward_hit_correlation':None if corr is None or not np.isfinite(corr) else float(corr),'weighted_RL_to_CE_norm':float(.1*og.norm()/sg.norm()),'cosine_to_CE':cosine(og,sg)}
        for kind,reward in [('integrated_hit',exact),('grpo_hit',exact),('grpo_sid',sid),('grpo_support',support),('grpo_rank',rank)]:
            if kind=='integrated_hit':loss,d=integrated_loss(actor,b,reward,2024+i,group=16)
            else:
                samples=sample_group(actor,b,reward,2024+i,group=16);loss,d=grpo_loss(actor,b,samples)
            g=grad_vector(actor,loss)
            rec['estimators'][kind]={'norm':float(g.norm()),'cosine_to_CE':cosine(g,sg),'suggested_weight':float((.25*sg.norm()/g.norm().clamp_min(1e-12)).clamp(.001,1000)),**d}
        records.append(rec);print(json.dumps(rec),flush=True)
    # Zero-gradient exact-hit groups contain no calibration information. Never
    # turn them into an enormous coefficient by dividing by epsilon.
    calibration={kind:float(np.median([r['estimators'][kind]['suggested_weight'] for r in records if r['estimators'][kind]['norm']>1e-8]))
        if any(r['estimators'][kind]['norm']>1e-8 for r in records) else 0.
        for kind in records[0]['estimators']}
    save(folder/'diagnostic.json',{'status':'complete','TRAIN_only':True,'VAL_only_source_selection':True,'TEST_reads':0,'source':str(path),'source_variant':variant,'source_val_SNDCG10':score,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'probe_rows':rows[:64].tolist(),'group':16,'calibration':'median over4 TRAIN batches; RL probe-group gradient norm targeted at .25 of supervised; bound .001..1000','weights':calibration,'batches':records})

if __name__=='__main__':main()
