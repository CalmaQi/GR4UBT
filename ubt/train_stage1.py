"""No-alignment SFT tuning workers. No RL and no TEST during search/refits."""
import argparse,hashlib,importlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from stage1_plan import OUT,ROOT,SOURCE,budgets
from paths import DATA
from runner_eval import batch,evaluate,save
from runner import Actor

def atomic_save(value,path):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial');torch.save(value,temp);temp.replace(path)

def load_data(cfg):
    with np.load(SOURCE/cfg['dataset']/'assets.npz',allow_pickle=False) as f:data=dict(f)
    asset=DATA/cfg['dataset']/(cfg['asset']+'.npz')
    with np.load(asset) as f:data.update(text=f['text'],codes=f['codes'])
    if cfg['history']==100 or cfg['tails']:
        name='history100' if cfg['history']==100 else 'tail50'
        with np.load(DATA/cfg['dataset']/(name+'.npz')) as f:data.update(dict(f))
    if cfg['extra_dense']:
        with np.load(asset.parent/'extra_dense.npz') as f:
            for key in ['dense_h','dense_labels']:data[key]=np.concatenate([data[key],f[key]],0)
    elif cfg.get('repeat_dense'):
        with np.load(asset.parent/'extra_dense.npz') as f:target=len(data['dense_h'])+len(f['dense_h'])
        indices=np.arange(target)%len(data['dense_h'])
        for key in ['dense_h','dense_labels']:data[key]=data[key][indices]
    data['dense_y']=np.ones((len(data['dense_h']),cfg['K']),np.int32)
    return data

class SFTActor(Actor):
    def encode_history(self,b):
        return self.model.encode_history({k:b[k] for k in ('input_ids','attention_mask','seq_lens')},deterministic=not self.training)

def construct(cfg,data):
    m=data['codes'].shape[1]
    # Each worker has one codebook count. These private process globals never
    # mutate shared source files or another experiment's Python process.
    for name in ['sidgen_mtp','sidgen_mtp_aux','sidgen_mu_joint','sidgen_trajectory']:
        module=importlib.import_module(name)
        if hasattr(module,'N_CODEBOOK'):module.N_CODEBOOK=m
    from sidgen_trajectory import SIDUBTTrajectory
    codes=torch.as_tensor(data['codes'].astype(np.int64));dc=F.pad(codes,(0,0,1,0),value=-1)
    tokens=dc+torch.arange(m)[None]*256+1;tokens[0]=0
    text=F.pad(torch.as_tensor(data['text'],dtype=torch.float32),(0,0,1,0))
    model=SIDUBTTrajectory(len(codes),tokens,dc,text,n_embd=448,n_layer=cfg['layers'],n_head=4,n_inner=1024,
        max_history=cfg['history'],embd_pdrop=cfg['dense_drop'],attn_pdrop=cfg['dense_drop'],resid_pdrop=0.,
        temperature=.03,item_ce_lambda=.5*cfg['item_scale'],item_ce_tau=.15,align_lambda=0.,mtp_depths=cfg['K'],
        mtp_lambda=0.,mtp_detach=0,detach_feedback=False,input_pooling='attn',token_ce_lambda=cfg['token_scale'],
        ce_chunk_size=cfg['ce_chunk'],ubt_eta=.75,ubt_policy_temperature=.15,ubt_kernel_temperature=.2,
        mu_cond_drop=cfg['feedback_drop'],ubt_null_aux_lambda=.1).cuda()
    if not cfg['text_input']:model.use_text=False
    assert model.align_lambda==0 and model.n_pred_head==m and len(model._digit_embeddings())==m
    return SFTActor(model).cuda()

def set_stage_dropout(actor,p):
    actor.model.gpt2.drop.p=p
    for block in actor.model.gpt2.h:block.attn.attn_dropout.p=p
    actor.train(p>0)

def token_losses(model,queries,targets):
    query=F.normalize(queries.float(),dim=-1);codes=model.item_dcodes[targets];tables=model._digit_embeddings()
    values=[]
    for d in range(model.n_pred_head):
        lp=F.log_softmax(query[...,d,:]@tables[d].float().T/model.temperature,-1)
        values.append(-lp.gather(-1,codes[...,d,None]).squeeze(-1))
    return torch.stack(values,-1).mean(-1)

def joint_loss(actor,b,cfg):
    mask=b['set_targets']>0;targets=b['set_targets'].clamp_min(1)
    out=actor(b,actions0=targets-1,return_trace=True)
    token=token_losses(actor.model,out['queries'],targets) if cfg['token_scale'] else out['log_probs']*0
    item=-out['log_probs'];den=mask.sum().clamp_min(1)
    nt=token[:,0].mean();ni=item[:,0].mean();tt=(token*mask).sum()/den;ti=(item*mask).sum()/den
    dropped=actor(b,actions0=targets-1,return_trace=True,condition_drop=cfg['feedback_drop'])
    at=(token_losses(actor.model,dropped['queries'],targets)*mask).sum()/den if cfg['token_scale'] else ni*0
    ai=(-dropped['log_probs']*mask).sum()/den
    loss=cfg['token_scale']*(nt+.5*tt+.1*at)+cfg['item_scale']*(.5*ni+.5*ti+.1*ai)
    return loss,{'NTP_token_CE':float(nt.detach()),'NTP_item_CE':float(ni.detach()),'trajectory_token_CE':float(tt.detach()),
                 'trajectory_item_CE':float(ti.detach()),'null_token_CE':float(at.detach()),'null_item_CE':float(ai.detach()),'alignment':0.}

def snapshot(actor,path,cfg,epoch,score,stage):
    atomic_save({'state':actor.state_dict(),'feedback':actor.feedback_table,'config':cfg,'epoch':epoch,'score':score,'stage':stage},path)

def load_tuned_checkpoint(path):
    """Reconstruct 8/16/32-code and 1/2-layer variants without legacy shape assumptions."""
    state=torch.load(path,map_location='cuda',weights_only=False);cfg=state['config'];data=load_data(cfg);actor=construct(cfg,data)
    actor.feedback_table=state['feedback'];actor.load_state_dict(state['state']);actor.eval()
    return actor,data,state

def averages(records):return {key:float(np.mean([r[key] for r in records if key in r])) for key in set().union(*(r.keys() for r in records))}

def train(cfg,phase):
    folder=OUT/cfg['dataset']/phase/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    data=load_data(cfg);actor=construct(cfg,data);schedule=budgets(cfg['dataset'],phase)
    config=cfg|schedule|{'phase':phase,'initialized_from_old_actor':False,'TEST_evaluations':0,
                       'parameters':sum(p.numel() for p in actor.parameters()),'codebooks':data['codes'].shape[1],
                       'unique_SIDs':len(np.unique(data['codes'],axis=0)),
                       'dense_rows':len(data['dense_h']),'dense_labels_per_epoch':int((data['dense_labels']>0).sum()),'joint_rows':len(data['train_h']),
                       'joint_dropout_is_actually_active':cfg['joint_drop']>0,
                       'asset_sha256':hashlib.sha256((DATA/cfg['dataset']/(cfg['asset']+'.npz')).read_bytes()).hexdigest()}
    save(folder/'config.json',config);records=[];global_step=0
    for stage in ['dense','joint']:
        completed=folder/(stage+'_complete.json')
        if completed.exists():
            state=torch.load(folder/(stage+'.pt'),map_location='cuda',weights_only=False)
            if state['feedback'] is not None:actor.feedback_table=state['feedback']
            actor.load_state_dict(state['state']);records=json.loads((folder/'progress.json').read_text())['history'];global_step=records[-1]['optimizer_steps_total'];continue
        if stage=='joint':
            actor.eval()
            with torch.no_grad():actor.feedback_table=actor.model._input_embs(torch.arange(1,actor.n_items+1,device='cuda')[None]).squeeze(0).detach().clone()
        split='dense' if stage=='dense' else 'train';num=len(data[split+'_h'])
        bs=cfg['dense_batch'] if stage=='dense' else cfg['joint_batch'];max_epochs=schedule[stage+'_epochs'];patience=schedule[stage+'_patience']
        optimizer=torch.optim.AdamW(actor.parameters(),lr=.003 if stage=='dense' else .0001,weight_decay=0.)
        nsteps=math.ceil(num/bs);warmup=max(1,int(.1*max_epochs*nsteps));total=max_epochs*nsteps
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*min(max(step-warmup,0)/max(total-warmup,1),1)))) if stage=='dense' else None
        best=-math.inf;bad=0;first=1;recovery=folder/(stage+'_recovery.pt')
        if recovery.exists():
            state=torch.load(recovery,map_location='cuda',weights_only=False)
            if state['feedback'] is not None:actor.feedback_table=state['feedback']
            actor.load_state_dict(state['state']);optimizer.load_state_dict(state['optimizer'])
            if scheduler:scheduler.load_state_dict(state['scheduler'])
            records=state['history'];first=state['epoch']+1;best=state['best'];bad=state['bad'];global_step=state['global_step']
            torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        for epoch in range(first,max_epochs+1):
            rows=np.random.default_rng(2024+epoch).permutation(num)
            if stage=='joint' and schedule['max_joint_rows']:rows=rows[:schedule['max_joint_rows']]
            logs=[];tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            for begin in range(0,len(rows),bs):
                rr=rows[begin:begin+bs];b=batch(data,split,rr);optimizer.zero_grad(set_to_none=True);torch.manual_seed(20240000+global_step)
                set_stage_dropout(actor,cfg['dense_drop'] if stage=='dense' else cfg['joint_drop'])
                if stage=='dense':
                    b['labels']=torch.as_tensor(data['dense_labels'][rr].astype(np.int64),device='cuda');b.pop('set_targets')
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=actor.model(b)
                    loss=out.loss;detail={key:float(value.detach()) for key,value in out.losses.items() if value is not None}
                    assert 'align' not in detail
                else:loss,detail=joint_loss(actor,b,cfg)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite supervised loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step();global_step+=1
                if scheduler:scheduler.step()
                logs.append(detail|{'loss':float(loss.detach())})
            val=evaluate(actor,data,'val',cfg['K'],bs=cfg['eval_batch']);score=val['position1_NDCG@10'] if stage=='dense' else val['SNDCG@10']
            if score>best:best=score;bad=0;snapshot(actor,folder/(stage+'.pt'),cfg,epoch,best,stage)
            else:bad+=1
            record={'stage':stage,'epoch':epoch,'rows':len(rows),'optimizer_steps_total':global_step,'seconds':time.monotonic()-tick,
                    'score':score,'best':best,'VAL_SNDCG10':val['SNDCG@10'],'patience':bad,'losses':averages(logs),
                    'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'validation':val}
            records.append(record);save(folder/'progress.json',{'status':'running','config':config,'history':records,
                                                               'best_val_SNDCG10':max((r['VAL_SNDCG10'] for r in records if r['stage']=='joint'),default=None)})
            print(json.dumps({k:v for k,v in record.items() if k!='validation'}),flush=True)
            atomic_save({'state':actor.state_dict(),'feedback':actor.feedback_table,'optimizer':optimizer.state_dict(),
                         'scheduler':scheduler.state_dict() if scheduler else None,'epoch':epoch,'best':best,'bad':bad,
                         'global_step':global_step,'history':records,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},recovery)
            if bad>=patience:break
        state=torch.load(folder/(stage+'.pt'),map_location='cuda',weights_only=False)
        if state['feedback'] is not None:actor.feedback_table=state['feedback']
        actor.load_state_dict(state['state']);save(completed,{'status':'complete','epoch':state['epoch'],'score':state['score']})
    snapshot(actor,folder/'stage1.pt',cfg,state['epoch'],state['score'],'stage1')
    save(folder/'result.json',{'status':'complete','config':config,'history':records,'best_val_SNDCG10':state['score'],
                             'selected_joint_epoch':state['epoch'],'alignment_loss':0.,'TEST_evaluations':0})

def test_selected(ds):
    folder=OUT/ds;decision=json.loads((folder/'selection.json').read_text());cfg=decision['config']
    policy_path=OUT/'report_policy.json';policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
    if policy.get('exclude_extra_users_from_final') and cfg.get('extra_dense'):
        raise ValueError('Extra-user pretraining is exploratory only and excluded from final TEST reports')
    actor,data,state=load_tuned_checkpoint(decision['checkpoint'])
    if policy.get('exclude_extra_users_from_final') and state['config'].get('extra_dense'):
        raise ValueError('Checkpoint lineage used extra users; cannot relabel it as a no-extra-user result')
    metrics=evaluate(actor,data,'test',cfg['K'],bs=cfg['eval_batch'],path=folder/'selected_test_predictions.npz')
    save(folder/'test_metrics.json',{'status':'complete','selection':decision,'metrics':metrics})

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--phase',choices=['screen','full'],default='screen');p.add_argument('--test-dataset',choices=['lastfm','cellphones']);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('A CUDA-capable GPU is required')
    torch.set_num_threads(4);torch.manual_seed(2024);np.random.seed(2024);torch.backends.cuda.matmul.allow_tf32=True
    if a.test_dataset:test_selected(a.test_dataset)
    else:train(json.loads(Path(a.config).read_text()),a.phase)

if __name__=='__main__':main()
