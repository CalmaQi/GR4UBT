"""Validation-only pilots of compact RL objectives with matched CE/KL control."""
import argparse,copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from paths import DATA as SOURCE,ROOT,RUNS
from runner import load_actor,supervised_loss,reference_kl,snapshot,param_hash,batch,evaluate,save
from rl_clean import TaskReward,sample_group,integrated_loss,grpo_loss,balanced_rows

OUT=RUNS/'beauty_rl'

def mean_logs(logs):
    return {key:float(np.mean([v[key] for v in logs if key in v])) for key in set().union(*(v.keys() for v in logs))}

def train(args):
    ds=args.dataset;folder=OUT/ds/(args.tag or args.arm);folder.mkdir(parents=True,exist_ok=True)
    diag=json.loads((OUT/ds/'diagnostic.json').read_text());path=Path(diag['source'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=diag['source_sha256']:raise ValueError('source changed')
    data=dict(np.load(SOURCE/ds/'assets.npz',allow_pickle=False))
    actor,source=load_actor(data,3,path,448);actor.eval()
    reference=copy.deepcopy(actor).requires_grad_(False).eval();refhash=param_hash(reference)
    align=diag['source_variant']!='noalign'
    reward=TaskReward(torch.as_tensor(data['codes'],device='cuda'),.1 if args.arm=='grpo_sid' else 0.,
        support_reference=reference if args.arm in ('grpo_support','grpo_rank') else None,
        support_mode='rank' if args.arm=='grpo_rank' else 'logprob')
    calibration_key='grpo_hit_g128' if args.arm=='grpo_hit' and args.group==128 else args.arm
    coefficient=0. if args.arm=='control' else args.rl_scale*diag['weights'][calibration_key]
    groups=[{'params':[p for n,p in actor.named_parameters() if not n.startswith('model.mu_')],'lr':args.lr},
            {'params':[p for n,p in actor.named_parameters() if n.startswith('model.mu_')],'lr':10*args.lr}]
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    initial=evaluate(actor,data,'val',3);best=initial['SNDCG@10'];selected_step=0;selected_epoch=0
    snapshot(actor,folder/'selected.pt',0,best,'clean_rl')
    beta=.5;step=0;epoch_reports=[];validations=[{'step':0,'epoch':0,'metrics':initial}]
    config=vars(args)|{'source':str(path),'source_variant':diag['source_variant'],'source_val_score':source['score'],
        'training_seed':2024,'group':args.group,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'supervision':'same NTP + all-horizon token/item CE + inherited alignment + null auxiliary as Stage I',
        'CE_weight':1.,'RL_coefficient':coefficient,'calibration':'TRAIN-only probe gradients; initial ratio .25',
        'KL_direction':'P0||policy on independently sampled P0 prefixes; full catalog exact action sum',
        'KL_target':.01,'KL_beta_initial':beta,'KL_beta_bounds':[.5,20.],
        'no_diversity_reward':True,'no_redundancy_reward':True,'no_transition_count_reward':True,'no_cosine_reward':True,
        'replay':'fresh deterministic states and local probability summaries; no frozen hidden-state replay',
        'sampling':'on-policy original item temperature=.15; no top-k proposal',
        'rows':'up to2 windows/user/epoch drawn strictly from TRAIN; identical rows and optimizer steps across arms',
        'TEST_reads_during_selection':0,'success_requires':'nonzero selected update and comparison with CE+KL control; epoch0 fallback not RL success'}
    save(folder/'config.json',config)
    result={'status':'running','config':config,'initial_val':initial,'epochs':epoch_reports,'validations':validations}
    tick=time.time()
    for epoch in range(1,args.epochs+1):
        rows=balanced_rows(data,2024+epoch);epoch_tick=time.time();logs=[]
        # Evaluate four prespecified points per user-balanced pass, identically across arms.
        batches=math.ceil(len(rows)/args.batch);interval=max(1,math.ceil(batches/4))
        for bi,begin in enumerate(range(0,len(rows),args.batch)):
            b=batch(data,'train',rows[begin:begin+args.batch])
            samples=None
            if args.arm.startswith('grpo'):
                samples=sample_group(actor,b,reward,202400000+epoch*100000+bi,group=args.group)
            for inner in range(args.updates):
                actor.eval();optimizer.zero_grad(set_to_none=True)
                # Match the exact supervised recipe used by the selected source.
                torch.manual_seed(20240000+step)
                sup,detail=supervised_loss(actor,b,align=align,null=True)
                kl=reference_kl(actor,reference,b,20250000+step,group=2)
                if args.arm=='control':raw=sup*0;rd={}
                elif args.arm=='integrated_hit':raw,rd=integrated_loss(actor,b,reward,20260000+step,group=args.group)
                else:raw,rd=grpo_loss(actor,b,samples,clip=.1)
                # Short deterministic warmup prevents the first normalized group from jumping.
                warm=min((step+1)/32.,1.)
                loss=sup+coefficient*warm*raw+beta*kl
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite clean RL objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step();step+=1
                value=float(kl.detach())
                beta=float(np.clip(beta*math.exp(.05*np.clip(value/.01-1,-1,1)),.5,20.))
                logs.append(detail|{key:float(v) for key,v in rd.items()}|{'loss':float(loss.detach()),'raw_RL':float(raw.detach()),'KL':value,'beta':beta,'preclip_norm':float(norm),'step':step})
            if (bi+1)%interval==0 or bi+1==batches:
                val=evaluate(actor,data,'val',3);score=val['SNDCG@10']
                if score>best:
                    best=score;selected_step=step;selected_epoch=epoch;snapshot(actor,folder/'selected.pt',epoch,score,'clean_rl')
                validations.append({'step':step,'epoch':epoch,'fraction':(bi+1)/batches,'metrics':val,'selected_step':selected_step})
                result.update(optimizer_steps=step,selected_step=selected_step,selected_epoch=selected_epoch,best_val_SNDCG10=best)
                save(folder/'progress.json',result)
                print(json.dumps({'dataset':ds,'arm':args.arm,'epoch':epoch,'step':step,'VAL_SNDCG10':score,'best':best,'selected_step':selected_step,'mean':mean_logs(logs)}),flush=True)
        epoch_reports.append({'epoch':epoch,'rows':len(rows),'row_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),'seconds':time.time()-epoch_tick,'mean':mean_logs(logs)})
    assert param_hash(reference)==refhash
    snapshot(actor,folder/'endpoint.pt',args.epochs,validations[-1]['metrics']['SNDCG@10'],'clean_rl')
    selected=torch.load(folder/'selected.pt',weights_only=False,map_location='cuda');actor.load_state_dict(selected['state'])
    # No TEST access in the pilot; one promoted validation-selected recipe is evaluated later.
    result.update(status='complete',training_seconds=time.time()-tick,selected_step=selected_step,
        selected_epoch=selected_epoch,best_val_SNDCG10=best,VAL_gain=best-initial['SNDCG@10'],TEST_reads=0,reference_frozen=True)
    save(folder/'result.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('beauty','cellphones'),required=True)
    p.add_argument('--arm',choices=('control','integrated_hit','grpo_hit','grpo_sid','grpo_support','grpo_rank'),required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=32);p.add_argument('--updates',type=int,default=2);p.add_argument('--lr',type=float,default=1e-6);p.add_argument('--rl-scale',type=float,default=1.);p.add_argument('--tag');p.add_argument('--group',type=int,default=16);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('A CUDA-capable GPU is required')
    torch.set_num_threads(4);torch.manual_seed(2024);np.random.seed(2024);torch.backends.cuda.matmul.allow_tf32=True
    train(a)

if __name__=='__main__':main()
