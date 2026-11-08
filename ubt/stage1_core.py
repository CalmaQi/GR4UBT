"""Shared Stage-I training core for Last-FM and ML-1M."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_stage1 import (load_data, construct, SFTActor, set_stage_dropout,
    token_losses, snapshot, atomic_save)
from runner_eval import batch, evaluate
from paths import DATA, RUNS

OUT = RUNS / 'lastfm'
REFERENCE = DATA / 'lastfm/reference_val.json'
METRICS = ('MHR@5','MNDCG@5','MHR@10','MNDCG@10','SHR@5','SNDCG@5','SHR@10','SNDCG@10')

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)

def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def score(metrics, ndcg_guard=1.):
    reference=read(REFERENCE)
    if reference is None:
        raise FileNotFoundError(f'Missing validation reference asset: {REFERENCE.name}')
    ratios=[metrics[k]/reference[k]/(ndcg_guard if 'NDCG' in k else 1.) for k in METRICS]
    return (min(ratios),float(np.exp(np.mean(np.log(np.maximum(ratios,1e-12))))))

class DeepActor(SFTActor):
    def __init__(self,model,cfg):super().__init__(model);self.cfg=cfg
    def trajectory_from_hidden(self,h,actions0=None,K=None,generator=None,return_trace=False,greedy=False,condition_drop=0.):
        logs=[];chosen=[];actions=[];queries=[]
        mode=self.cfg['feedback_mode']
        if mode=='live_sid':table=self.model.item_feedback_table()
        elif mode=='live_input':table=self.model._input_embs(torch.arange(1,self.n_items+1,device=h.device)[None]).squeeze(0)
        else:table=self.feedback_table
        if table is None:table=self.model.item_feedback_table()
        for t in range(K):
            q=self.model.main_states(h[:,None]).squeeze(1)
            lp=F.log_softmax(self.model._policy_logits(self.model.score_catalog(q)),-1)
            action=actions0[:,t] if actions0 is not None else lp.argmax(-1) if greedy else torch.multinomial(lp.exp(),1,generator=generator).squeeze(1)
            logs.append(lp);actions.append(action);chosen.append(lp.gather(1,action[:,None]).squeeze(1))
            if return_trace:queries.append(q)
            if t+1<K:
                feedback_action=action
                if self.training and actions0 is not None and self.cfg['scheduled_prefix']:
                    draw=torch.rand(len(action),device=h.device)<self.cfg['scheduled_prefix']
                    feedback_action=torch.where(draw,lp.detach().argmax(-1),action)
                d,_,_=self.model.local_feedback(lp.exp(),feedback_action,table)
                if condition_drop:d=torch.where((torch.rand(len(d),device=h.device)<condition_drop)[:,None],self.model.mu_null[None],d)
                h=self.model._transition_hidden(h,d,t+2)
        result={'log_distributions':torch.stack(logs,1),'log_probs':torch.stack(chosen,1),'actions0':torch.stack(actions,1)}
        if return_trace:result['queries']=torch.stack(queries,1)
        return result

def make_actor(cfg,data,state=None):
    old=construct(cfg,data);model=old.model
    model.ubt_eta=cfg['eta'];model.ubt_policy_temperature=cfg['policy_tau'];model.item_ce_tau=cfg['policy_tau']
    model.mtp_tau=cfg['policy_tau'];model.ubt_kernel_temperature=cfg['kernel_tau'];model.temperature=cfg['token_tau']
    model.item_ce_ls=cfg['label_smoothing']
    if cfg.get('exact_semantic_kernel',False):
        # Same cosine kernel, without the fixed random dimensionality reduction.
        model.ubt_semantic_features=F.normalize(model.text_embs[1:].float(),dim=-1)
    actor=DeepActor(model,cfg).cuda()
    if state is not None:
        assert state['config'].get('alignment')==0 and not state['config'].get('extra_dense')
        if state.get('feedback') is not None:actor.feedback_table=state['feedback']
        actor.load_state_dict(state['state'])
    assert model.align_lambda==0 and cfg['alignment']==0 and not cfg['extra_dense'] and not cfg.get('repeat_dense')
    return actor

def training_mode(actor,cfg):
    # Training state also controls scheduled prefix sampling, independently of dropout.
    set_stage_dropout(actor,cfg['joint_drop']);actor.train()
    if cfg['freeze_backbone']:actor.model.gpt2.eval()

def loss_fn(actor,b,cfg):
    mask=b['set_targets']>0;targets=b['set_targets'].clamp_min(1);den=mask.sum().clamp_min(1)
    weights=b.get('query_weights',torch.ones(len(targets),device=targets.device))
    def weighted(values):return (values*weights).mean()
    def per_query(values):return (values*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    out=actor(b,actions0=targets-1,return_trace=True)
    tok=token_losses(actor.model,out['queries'],targets)
    lp=out['log_distributions'];item=-out['log_probs']
    if cfg['label_smoothing']:item=(1-cfg['label_smoothing'])*item-cfg['label_smoothing']*lp.mean(-1)
    nt=weighted(tok[:,0]);ni=weighted(item[:,0]);tt=weighted(per_query(tok));ti=weighted(per_query(item))
    total=cfg['token_scale']*(cfg['ntp_token_weight']*nt+cfg.get('trajectory_token_weight',cfg['trajectory_weight'])*tt)
    total=total+cfg['item_scale']*(cfg['ntp_item_weight']*ni+cfg.get('trajectory_item_weight',cfg['trajectory_weight'])*ti)
    if cfg['null_weight']:
        dropped=actor(b,actions0=targets-1,return_trace=True,condition_drop=cfg['feedback_drop'])
        aux=weighted(per_query(token_losses(actor.model,dropped['queries'],targets)))
        aux=cfg['token_scale']*aux+cfg['item_scale']*weighted(per_query(-dropped['log_probs']))
        total=total+cfg['null_weight']*aux
    ranking=total*0
    if cfg['rank_weight']:
        # Full-catalog top-5/top-10 boundary margin on TRAIN labels only.
        true=lp.gather(-1,(targets-1)[...,None]).squeeze(-1)
        neg=lp.masked_fill(F.one_hot(targets-1,actor.n_items).bool(),-torch.inf)
        hardest=neg.topk(10,-1).values
        ranking=weighted(per_query((F.softplus(hardest[...,4]-true+cfg['rank_margin'])+
                  F.softplus(hardest[...,9]-true+cfg['rank_margin']))*.5))
        total=total+cfg['rank_weight']*ranking
    return total,dict(token=float(tt.detach()),item=float(ti.detach()),ranking=float(ranking.detach()),loss=float(total.detach()))

def optimizer_for(actor,cfg):
    groups={};aliases={}
    for name,param in actor.named_parameters():
        kind='transition' if name.startswith('model.mu_') else 'backbone' if name.startswith(('model.gpt2.','model.text_','model.pool_')) else 'readout'
        if kind=='backbone' and cfg['freeze_backbone']:param.requires_grad_(False);continue
        rate=cfg['joint_lr']*(1. if kind=='transition' else cfg[kind+'_multiplier'])
        groups.setdefault((kind,rate),[]).append(param);aliases[name]=kind
    return torch.optim.AdamW([dict(params=params,lr=rate,name=kind) for (kind,rate),params in groups.items()],weight_decay=cfg['weight_decay'])

def initialize_feedback(actor):
    actor.eval()
    with torch.no_grad():actor.feedback_table=actor.model._input_embs(torch.arange(1,actor.n_items+1,device='cuda')[None]).squeeze(0).detach().clone()

def query_weights(data,cfg):
    counts=np.maximum(data['train_ends'].astype(np.int64)-(1 if cfg.get('tails') else cfg['K']),0)
    assert int(counts.sum())==len(data['train_h'])
    weights=np.ones(len(data['train_h']),dtype=np.float64)
    if cfg.get('user_balance',0):
        weights*=np.repeat(np.maximum(counts,1).astype(np.float64)**(-cfg['user_balance']),counts)
    if cfg.get('recent_tau',0):
        ages=np.concatenate([np.arange(int(n)-1,-1,-1) for n in counts if n])
        if cfg.get('tails'):ages=np.maximum(ages-(cfg['K']-1),0)
        floor=cfg.get('recent_floor',.25);assert 0<floor<=1
        weights*=floor+(1-floor)*np.exp(-ages/cfg['recent_tau'])
    weights/=weights.mean()
    assert np.isfinite(weights).all() and (weights>0).all()
    return weights.astype(np.float32)

def warmup(cfg):
    folder=OUT/'warmup'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    data=load_data(cfg);actor=make_actor(cfg,data);save(folder/'config.json',cfg)
    opt=torch.optim.AdamW(actor.parameters(),lr=cfg['dense_lr'],weight_decay=.0)
    total=math.ceil(len(data['dense_h'])/cfg['dense_batch'])*cfg['dense_epochs'];warm=max(1,int(.1*total))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:min((t+1)/warm,1.)*.5*(1+math.cos(math.pi*min(max(t-warm,0)/max(total-warm,1),1))))
    best=-math.inf;bad=0;logs=[];step=0;first=1
    if (folder/'recovery.pt').exists():
        r=torch.load(folder/'recovery.pt',map_location='cuda',weights_only=False);actor.load_state_dict(r['state']);opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler']);logs=r['history'];step=r['step'];first=r['epoch']+1;best=r['best'];bad=r['bad']
    for epoch in range(first,cfg['dense_epochs']+1):
        order=np.random.default_rng(2024+epoch).permutation(len(data['dense_h']));losses=[];tick=time.time()
        for start in range(0,len(order),cfg['dense_batch']):
            rr=order[start:start+cfg['dense_batch']];b=batch(data,'dense',rr);b.pop('set_targets')
            b['labels']=torch.as_tensor(data['dense_labels'][rr].astype(np.int64),device='cuda')
            set_stage_dropout(actor,cfg['dense_drop']);opt.zero_grad(set_to_none=True);torch.manual_seed(20240000+step)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=actor.model(b)
            loss=out.loss
            if not torch.isfinite(loss):raise FloatingPointError('dense nonfinite')
            assert not out.losses.get('align')
            loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step();step+=1;losses.append(float(loss.detach()))
        val=evaluate(actor,data,'val',5,bs=128)
        # Warmup has no learned trajectory. Use both next-item HR cutoffs and NDCG.
        criterion=float(np.exp(np.mean(np.log([val['position1_HR@5'],val['position1_HR@10'],val['position1_NDCG@5'],val['position1_NDCG@10']]))))
        if criterion>best:best=criterion;bad=0;snapshot(actor,folder/'dense.pt',cfg,epoch,best,'dense')
        else:bad+=1
        logs.append(dict(epoch=epoch,step=step,loss=float(np.mean(losses)),score=criterion,best=best,bad=bad,validation=val,seconds=time.time()-tick))
        save(folder/'progress.json',dict(status='running',history=logs));print(json.dumps({k:v for k,v in logs[-1].items() if k!='validation'}),flush=True)
        atomic_save(dict(state=actor.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=logs,step=step,epoch=epoch,best=best,bad=bad),folder/'recovery.pt')
        if bad>=cfg['dense_patience']:break
    save(folder/'result.json',dict(status='complete',config=cfg,best=best,history=logs,checkpoint=str(folder/'dense.pt'),alignment=0.,TEST_evaluations=0))

def train(cfg,phase):
    folder=OUT/phase/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    source=None;state=None
    if cfg.get('skip_dense_warmup'):
        assert cfg.get('reuse_dense') is None
    else:
        source=Path(cfg['reuse_dense']);state=torch.load(source,map_location='cuda',weights_only=False)
        assert state.get('stage')=='dense','Never use an older joint/RL checkpoint as a warmup'
        for key in ('asset','layers','history','token_scale','item_scale','dense_drop'):
            # A different token/item weighting is an explicit JOINT-only ablation.
            if key not in ('token_scale','item_scale'):assert state['config'][key]==cfg[key],(key,state['config'][key],cfg[key])
    data=load_data(cfg);actor=make_actor(cfg,data,state);initialize_feedback(actor);opt=optimizer_for(actor,cfg)
    weights=query_weights(data,cfg)
    maxsteps=384 if phase=='screen' else math.ceil(len(data['train_h'])/cfg['joint_batch'])*cfg.get('joint_epochs_override',6)
    ndcg_guard=.9 if phase=='full' else 1.
    patience=6 if phase=='screen' else cfg.get('joint_patience_override',10);interval=64
    scheduler=None
    if cfg.get('joint_lr_schedule')=='warmup_cosine':
        warm=cfg.get('joint_lr_warmup_steps',256)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min((step+1)/warm,1.)*.5*(1+math.cos(math.pi*min(max(step-warm,0)/max(maxsteps-warm,1),1))))
    config=cfg|dict(phase=phase,max_steps=maxsteps,validation_interval=interval,patience=patience,
        warmup_sha256=digest(source) if source else None,warmup_stage='dense' if source else 'none',alignment_loss=0.,extra_users=False,seed=2024,
        full_train_rows=len(data['train_h']),TEST_evaluations=0,VAL_NDCG_relative_target=ndcg_guard,
        query_weight_min=float(weights.min()),query_weight_max=float(weights.max()),query_weight_mean=float(weights.mean()))
    save(folder/'config.json',config)
    initial=evaluate(actor,data,'val',5,bs=128)
    best=(-math.inf,-math.inf);bad=0;logs=[];step=0
    if (folder/'recovery.pt').exists():
        r=torch.load(folder/'recovery.pt',map_location='cuda',weights_only=False)
        actor.feedback_table=r['feedback'];actor.load_state_dict(r['state']);opt.load_state_dict(r['optimizer']);logs=r['history'];step=r['step'];best=tuple(r['best']);bad=r['bad']
        if scheduler:scheduler.load_state_dict(r['scheduler'])
    tick=time.time();losses=[];startstep=step
    steps_epoch=math.ceil(len(data['train_h'])/cfg['joint_batch'])
    while step<maxsteps:
        epoch=step//steps_epoch;offset=step%steps_epoch
        order=np.random.default_rng(2025+epoch).permutation(len(data['train_h']))
        for pos in range(offset,steps_epoch):
            rr=order[pos*cfg['joint_batch']:(pos+1)*cfg['joint_batch']];b=batch(data,'train',rr)
            b['query_weights']=torch.as_tensor(weights[rr],device='cuda')
            training_mode(actor,cfg);opt.zero_grad(set_to_none=True);torch.manual_seed(20240000+step)
            loss,detail=loss_fn(actor,b,cfg)
            if not torch.isfinite(loss):raise FloatingPointError('joint nonfinite')
            loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1;losses.append(detail)
            if scheduler:scheduler.step()
            if step%interval==0 or step==maxsteps:
                val=evaluate(actor,data,'val',5,bs=128);objective=score(val,ndcg_guard)
                if objective>best:best=objective;bad=0;snapshot(actor,folder/'selected.pt',cfg,step,best[0],'stage1')
                else:bad+=1
                means={k:float(np.mean([row[k] for row in losses])) for k in losses[0]}
                row=dict(step=step,equivalent_epochs=step/steps_epoch,score=objective,best=best,bad=bad,losses=means,validation=val,seconds=time.time()-tick)
                logs.append(row);save(folder/'progress.json',dict(status='running',config=config,history=logs,best=best));print(json.dumps({k:v for k,v in row.items() if k!='validation'}),flush=True)
                atomic_save(dict(state=actor.state_dict(),feedback=actor.feedback_table,optimizer=opt.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,history=logs,step=step,best=best,bad=bad),folder/'recovery.pt');losses=[]
                if bad>=patience and (phase=='screen' or step>=steps_epoch):break
            if step>=maxsteps:break
        if bad>=patience and (phase=='screen' or step>=steps_epoch):break
    checkpoint=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
    selected=next(row for row in logs if row['step']==checkpoint['epoch'])
    save(folder/'result.json',dict(status='complete',config=config,initial_validation=initial,best=best,
         selected_step=checkpoint['epoch'],validation=selected['validation'],history=logs,
         checkpoint=str(folder/'selected.pt'),TEST_evaluations=0,alignment_loss=0.))

