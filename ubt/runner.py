"""Full token/item CE + alignment, differentiable local recurrence, anchored RL."""
import argparse,copy,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sidgen_trajectory import SIDUBTTrajectory
from runner_eval import batch,evaluate,make_reward,save,metrics_from_ranks
from prepare_data import OUT,OLD
from paths import DATA
from ubt_bapo_relations import policy_loss
from ubt_context_relations import ContextRelationScorer,ContextRelationReward

class Actor(nn.Module):
    def __init__(self,model,table=None):
        super().__init__();self.model=model;self.n_items=model.n_items;self.horizon=model.mtp_depths
        self.register_buffer('feedback_table',table)
    def encode_history(self,b):
        return self.model.encode_history({k:b[k] for k in ('input_ids','attention_mask','seq_lens')})
    def trajectory_from_hidden(self,h,actions0=None,K=None,generator=None,return_trace=False,greedy=False,condition_drop=0.):
        logs=[];chosen=[];actions=[];queries=[]
        table=self.feedback_table if self.feedback_table is not None else self.model.item_feedback_table()
        for t in range(K):
            q=self.model.main_states(h[:,None]).squeeze(1)
            lp=F.log_softmax(self.model._policy_logits(self.model.score_catalog(q)),-1)
            a=actions0[:,t] if actions0 is not None else lp.argmax(-1) if greedy else torch.multinomial(lp.exp(),1,generator=generator).squeeze(1)
            logs.append(lp);actions.append(a);chosen.append(lp.gather(1,a[:,None]).squeeze(1))
            if return_trace:queries.append(q)
            if t+1<K:
                d,_,_=self.model.local_feedback(lp.exp(),a,table)
                if condition_drop:d=torch.where((torch.rand(len(d),device=d.device)<condition_drop)[:,None],self.model.mu_null[None],d)
                h=self.model._transition_hidden(h,d,t+2)
        out={'log_distributions':torch.stack(logs,1),'log_probs':torch.stack(chosen,1),'actions0':torch.stack(actions,1)}
        if return_trace:out['queries']=torch.stack(queries,1)
        return out
    def forward(self,b,actions0=None,K=None,**kwargs):
        return self.trajectory_from_hidden(self.encode_history(b),actions0=actions0,K=K or self.horizon,**kwargs)

def construct(data,k,dim=448):
    c=torch.as_tensor(data['codes'],dtype=torch.long);dc=F.pad(c,(0,0,1,0),value=-1)
    tokens=dc+torch.arange(16)[None]*256+1;tokens[0]=0
    text=F.pad(torch.as_tensor(data['text'],dtype=torch.float32),(0,0,1,0))
    return SIDUBTTrajectory(len(c),tokens,dc,text,n_embd=dim,n_layer=2,n_head=4,n_inner=1024,
        max_history=50,embd_pdrop=.5,attn_pdrop=.5,resid_pdrop=0.,temperature=.03,
        item_ce_lambda=.5,item_ce_tau=.15,item_ce_ls=0.,align_lambda=.1,mtp_depths=k,
        mtp_lambda=0.,mtp_detach=0,detach_feedback=False,input_pooling='attn',
        token_ce_lambda=1.,ce_chunk_size=256,ubt_eta=.75,ubt_policy_temperature=.15,
        ubt_kernel_temperature=.2,mu_cond_drop=.25,ubt_null_aux_lambda=.1).cuda()

def token_ce(model,queries,targets):
    q=F.normalize(queries.float(),dim=-1);tables=model._digit_embeddings();codes=model.item_dcodes[targets]
    values=[]
    for digit in range(16):
        lp=F.log_softmax(q[...,digit,:]@tables[digit].float().T/model.temperature,-1)
        values.append(-lp.gather(-1,codes[...,digit,None]).squeeze(-1))
    return torch.stack(values,-1).mean(-1)

def supervised_loss(actor,b,align=True,null=False):
    m=actor.model;out=actor(b,actions0=b['set_targets']-1,return_trace=True)
    token=token_ce(m,out['queries'],b['set_targets']);item=-out['log_probs']
    # Dense NTP anchor at every chronological query endpoint, plus every future horizon.
    ntp_token=token[:,0].mean();ntp_item=item[:,0].mean();traj_token=token.mean();traj_item=item.mean()
    al=m._align_loss(b) if align else ntp_token*0
    total=ntp_token+.5*ntp_item+.5*(traj_token+traj_item)+.1*al
    detail={'ntp_token_CE':float(ntp_token.detach()),'ntp_item_CE':float(ntp_item.detach()),'trajectory_token_CE':float(traj_token.detach()),'trajectory_item_CE':float(traj_item.detach()),'alignment':float(al.detach())}
    if null:
        dropped=actor(b,actions0=b['set_targets']-1,return_trace=True,condition_drop=.25)
        aux=(-dropped['log_probs']).mean()+token_ce(m,dropped['queries'],b['set_targets']).mean()
        total=total+.1*aux;detail['null_token_item_CE']=float(aux.detach())
    return total,detail

def reference_kl(actor,reference,b,seed,group=2):
    # KL(P0 || Ptheta) under P0 prefixes: exact catalog action sum, sampled reference prefixes.
    rng=torch.Generator(device='cuda').manual_seed(seed);k=actor.horizon
    with torch.no_grad():
        h0=reference.encode_history(b).repeat_interleave(group,0)
        old=reference.trajectory_from_hidden(h0,K=k,generator=rng)
    h=actor.encode_history(b).repeat_interleave(group,0)
    live=actor.trajectory_from_hidden(h,actions0=old['actions0'],K=k)
    log0=old['log_distributions'];logp=live['log_distributions']
    return (log0.exp()*(log0-logp)).sum(-1).mean()

def snapshot(actor,path,epoch,score,stage):
    torch.save({'state':actor.state_dict(),'feedback':actor.feedback_table,'epoch':epoch,'score':score,'eta':actor.model.ubt_eta,'stage':stage,'history':50},path)

def load_actor(data,k,path,dim):
    v=torch.load(path,weights_only=False,map_location='cuda');actor=Actor(construct(data,k,dim),v['feedback']).cuda();actor.load_state_dict(v['state']);actor.model.ubt_eta=v['eta'];return actor,v

def param_hash(actor):
    h=hashlib.sha256()
    for n,p in actor.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def base(args,data,folder):
    actor=Actor(construct(data,args.k,args.dim)).cuda();m=actor.model;logs=[]
    variant=getattr(args,'variant','full')
    if variant=='hard_tf':m.ubt_eta=0.
    if variant=='noalign':m.align_lambda=0.
    for stage,epochs,lr,patience in [('dense',120,.003,16),('joint',24,.0001,8)]:
        if stage=='dense' and variant=='hard_tf':
            source=OUT/args.dataset/'dense.pt';state=torch.load(source,weights_only=False,map_location='cuda')
            actor.load_state_dict(state['state']);m.ubt_eta=0.
            snapshot(actor,folder/'dense.pt',state['epoch'],state['score'],'dense')
            save(folder/'dense_reuse.json',{'checkpoint':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'same selected NTP warmup; no trajectory or RL weights reused; local feedback is inactive in dense NTP and selection uses position1 only'})
            print('REUSE_DENSE',str(source),flush=True)
            continue
        if stage=='joint':
            with torch.no_grad():actor.feedback_table=m._input_embs(torch.arange(1,actor.n_items+1,device='cuda')[None]).squeeze(0).detach().clone()
        opt=torch.optim.AdamW(actor.parameters(),lr=lr,weight_decay=0.)
        split='dense' if stage=='dense' else 'train';num=len(data[split+'_h']);steps=math.ceil(num/args.batch)
        total_steps=epochs*steps;warmup=max(1,int(.1*total_steps))
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda st:min((st+1)/warmup,1.)*.5*(1+math.cos(math.pi*min(max(st-warmup,0)/max(total_steps-warmup,1),1)))) if stage=='dense' else None
        best=-math.inf;bad=0
        for epoch in range(1,epochs+1):
            order=np.random.default_rng(2024+epoch).permutation(num);values=[];components=[];tick=time.time()
            for begin in range(0,num,args.batch):
                rr=order[begin:begin+args.batch];b=batch(data,split,rr);opt.zero_grad(set_to_none=True)
                if stage=='dense':
                    m.train();b['labels']=torch.as_tensor(data['dense_labels'][rr].astype(np.int64),device='cuda');b.pop('set_targets')
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=m(b)
                    loss=out.loss;detail={key:float(v.detach()) for key,v in out.losses.items() if v is not None}
                else:
                    actor.eval();loss,detail=supervised_loss(actor,b,align=variant!='noalign',null=variant!='hard_tf')
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite supervised loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step()
                if scheduler:scheduler.step()
                values.append(float(loss.detach()));components.append(detail)
            val=evaluate(actor,data,'val',args.k)
            # NTP checkpoint selection matches historical next1; joint selection uses all horizons.
            score=val['position1_NDCG@10'] if stage=='dense' else val['SNDCG@10']
            if score>best:best=score;bad=0;snapshot(actor,folder/(stage+'.pt'),epoch,score,stage)
            else:bad+=1
            means={key:float(np.mean([v[key] for v in components if key in v])) for key in set().union(*(v.keys() for v in components))}
            logs.append({'stage':stage,'epoch':epoch,'loss':float(np.mean(values)),'loss_components':means,'seconds':time.time()-tick,'lr':opt.param_groups[0]['lr'],'val':val})
            save(folder/'base_progress.json',logs);print(json.dumps({key:logs[-1][key] for key in ('stage','epoch','loss','loss_components','seconds','lr')}|{'position1_NDCG10':val['position1_NDCG@10'],'SNDCG10':val['SNDCG@10']}),flush=True)
            if bad>=patience:break
        state=torch.load(folder/(stage+'.pt'),weights_only=False,map_location='cuda');actor.load_state_dict(state['state'])
        save(folder/(stage+'_test_metrics.json'),evaluate(actor,data,'test',args.k,path=folder/(stage+'_test_predictions.npz')))
    snapshot(actor,folder/'stage1.pt',state['epoch'],state['score'],'stage1')
    metrics=evaluate(actor,data,'test',args.k,path=folder/'base_predictions.npz');save(folder/'base_metrics.json',metrics)
    save(folder/'stage1_complete.json',{'status':'complete','metrics':metrics,'variant':variant,'supervised_terms':['NTP token CE','NTP item CE','all-horizon token CE','all-horizon item CE'],'alignment_enabled':variant!='noalign','null_drop_auxiliary':variant!='hard_tf','frozen_backbone':False,'eta':m.ubt_eta})

def context_reward(data,ds,reward):
    payload=torch.load(OLD/ds/'scorer.pt',weights_only=False,map_location='cuda')
    features=F.normalize(F.normalize(torch.as_tensor(data['text'],device='cuda'),dim=-1)@payload['projection'],dim=-1)
    scorer=ContextRelationScorer(features).cuda();scorer.load_state_dict(payload['state']);scorer.set_stage('frozen').eval()
    return ContextRelationReward(reward,scorer)

def post(args,data,folder):
    actor,source=load_actor(data,args.k,folder/'stage1.pt',args.dim);actor.eval();actor.model.mu_null.requires_grad_(True)
    reference=copy.deepcopy(actor).requires_grad_(False).eval();refhash=param_hash(reference)
    reward=make_reward(data)
    if args.mode=='context':reward=context_reward(data,args.dataset,reward)
    arm=folder/args.mode;arm.mkdir(exist_ok=False)
    opt=torch.optim.AdamW([{'params':[p for n,p in actor.named_parameters() if not n.startswith('model.mu_')],'lr':1e-5},{'params':[p for n,p in actor.named_parameters() if n.startswith('model.mu_')],'lr':1e-4}],weight_decay=0.)
    initial=evaluate(actor,data,'val',args.k);best=initial['SNDCG@10'];best_epoch=0
    # Epoch0 remains a legitimate validation-selected checkpoint; endpoint6 is always reported.
    snapshot(actor,arm/'selected.pt',0,best,'stage2');logs=[]
    for epoch in range(1,7):
        order=np.random.default_rng(2024+epoch).permutation(len(data['train_h']));tick=time.time();values=[];step_logs=[]
        for begin in range(0,len(order),args.rl_batch):
            b=batch(data,'train',order[begin:begin+args.rl_batch]);opt.zero_grad(set_to_none=True)
            supervised,detail=supervised_loss(actor,b,null=False)
            raw,rd=policy_loss(actor,b,reward,'integrated',20240000+epoch*10000000+begin,group=8)
            kl=reference_kl(actor,reference,b,20250000+epoch*10000000+begin,group=2)
            loss=supervised+args.rl_weight*raw+args.kl_weight*kl
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite anchored RL loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step()
            values.append(float(loss.detach()))
            step_logs.append(detail|{'RL_raw':float(raw.detach()),'KL':float(kl.detach()),'sample_reward':rd['sample_reward']})
        val=evaluate(actor,data,'val',args.k);score=val['SNDCG@10']
        if score>best:best=score;best_epoch=epoch;snapshot(actor,arm/'selected.pt',epoch,score,'stage2')
        logs.append({'epoch':epoch,'loss':float(np.mean(values)),'seconds':time.time()-tick,'components':{key:float(np.mean([v[key] for v in step_logs])) for key in step_logs[0]},'val':val,'best_epoch':best_epoch,'RL_weight':args.rl_weight,'KL_weight':args.kl_weight,'supervised_weight':1.})
        save(arm/'progress.json',logs);print(json.dumps({key:logs[-1][key] for key in ('epoch','loss','components','seconds','best_epoch')}|{'SNDCG10':score}),flush=True)
    assert param_hash(reference)==refhash
    snapshot(actor,arm/'endpoint6.pt',6,val['SNDCG@10'],'stage2')
    endpoint=evaluate(actor,data,'test',args.k,path=arm/'endpoint6_predictions.npz');save(arm/'endpoint6_metrics.json',endpoint)
    selected=torch.load(arm/'selected.pt',weights_only=False,map_location='cuda');actor.load_state_dict(selected['state'])
    metrics=evaluate(actor,data,'test',args.k,path=arm/'selected_predictions.npz')
    save(arm/'result.json',{'status':'complete','selected_epoch':best_epoch,'selected_is_unmodified_stage1':best_epoch==0,'training_seed':2024,'RL_weight':args.rl_weight,'KL_weight':args.kl_weight,'supervised_weight':1.,'all_token_item_CE':True,'alignment':True,'reference_frozen':True,'metrics':metrics,'endpoint6_metrics':endpoint})

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--k',type=int,required=True);p.add_argument('--mode',choices=('base','rl','context'),required=True);p.add_argument('--batch',type=int,default=256);p.add_argument('--rl-batch',type=int,default=64);p.add_argument('--dim',type=int,default=448);p.add_argument('--rl-weight',type=float,default=.1);p.add_argument('--kl-weight',type=float,default=.1);p.add_argument('--variant',choices=('full','hard_tf','noalign'),default='full');a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('A CUDA-capable GPU is required')
    assert a.k in (3,5)
    torch.set_num_threads(4);torch.manual_seed(2024);np.random.seed(2024);torch.backends.cuda.matmul.allow_tf32=True
    folder=OUT/a.dataset;data=dict(np.load(DATA/a.dataset/'assets.npz',allow_pickle=False))
    if a.variant!='full':
        assert a.mode=='base','these user-requested variants are Stage I ablations only'
        folder=folder/'ablations'/a.variant;folder.mkdir(parents=True,exist_ok=True)
    save(folder/(a.mode+'_config.json'),vars(a)|{'history':50,'train_stride':1,'seed':2024,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (base if a.mode=='base' else post)(a,data,folder)

if __name__=='__main__':main()
