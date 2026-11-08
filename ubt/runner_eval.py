"""Single-seed paper-aligned UBT evaluation. Fresh supervised fitting; CE-free RL.

Input asset.npz: text [N,D], codes [N,16], train/val/test padded histories,
targets and lengths. No held-out label enters free-running ranking.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sidgen_trajectory import SIDUBTTrajectory
from ubt_bapo_relations import RelationReward, policy_loss
from ubt_context_relations import ContextRelationScorer, ContextRelationReward

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs'

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)

class Actor(nn.Module):
    def __init__(self, model, feedback=None):
        super().__init__();self.model=model;self.n_items=model.n_items
        self.register_buffer('feedback_table', feedback)
    def encode_history(self,batch):
        return self.model.encode_history({k:batch[k] for k in ('input_ids','attention_mask','seq_lens')})
    def trajectory_from_hidden(self,h,actions0=None,K=None,generator=None,return_trace=False,greedy=False,condition_drop=0.):
        logs=[];selected=[];actions=[]
        table=self.feedback_table if self.feedback_table is not None else self.model.item_feedback_table()
        for t in range(K):
            lp=F.log_softmax(self.model._policy_logits(self.model.main_logits(h)),dim=-1)
            a=actions0[:,t] if actions0 is not None else lp.argmax(-1) if greedy else torch.multinomial(lp.exp(),1,generator=generator).squeeze(1)
            logs.append(lp);selected.append(lp.gather(1,a[:,None]).squeeze(1));actions.append(a)
            if t+1<K:
                d,_,_=self.model.local_feedback(lp.exp(),a,table)
                if condition_drop:
                    d=torch.where((torch.rand(len(d),device=d.device)<condition_drop)[:,None],self.model.mu_null[None],d)
                h=self.model._transition_hidden(h,d,t+2)
        return {'log_distributions':torch.stack(logs,1),'log_probs':torch.stack(selected,1),'actions0':torch.stack(actions,1)}
    def forward(self,batch,actions0=None,K=None,**kwargs):
        return self.trajectory_from_hidden(self.encode_history(batch),actions0=actions0,K=K or batch['set_targets'].shape[1],**kwargs)

def batch(data,split,rows,device='cuda'):
    h=torch.as_tensor(data[split+'_h'][rows].astype(np.int64),device=device)
    return {'input_ids':h,'attention_mask':(h>0).long(),
            'seq_lens':(h>0).sum(1),
            'set_targets':torch.as_tensor(data[split+'_y'][rows].astype(np.int64),device=device)}

def metrics_from_ranks(ranks,nll):
    item_nll=float(nll.astype(np.float64).mean());seq_nll=float(nll.astype(np.float64).sum(1).mean())
    out={'users':len(ranks),'K':ranks.shape[1],'SeqNLL':float(nll.sum(1).mean()),
         'item_NLL':item_nll,'PPL':math.exp(item_nll) if item_nll<709 else None,
         'sequence_PPL':math.exp(seq_nll) if seq_nll<709 else None,
         'log_sequence_PPL':seq_nll}
    for cutoff in (5,10,20,50,100):
        hit=(ranks<=cutoff).astype(float);dcg=hit/np.log2(ranks+1)
        hr=hit.mean(0);nd=dcg.mean(0)
        out.update({f'MHR@{cutoff}':float(hr.mean()),f'MNDCG@{cutoff}':float(nd.mean()),
                    f'SHR@{cutoff}':float(np.prod(hr)**(1/ranks.shape[1])),
                    f'SNDCG@{cutoff}':float(np.prod(nd)**(1/ranks.shape[1])),
                    f'SeqMatch@{cutoff}':float(hit.all(1).mean())})
        for t in range(ranks.shape[1]):
            out[f'position{t+1}_HR@{cutoff}']=float(hr[t]);out[f'position{t+1}_NDCG@{cutoff}']=float(nd[t])
    return out

@torch.no_grad()
def evaluate(actor,data,split,k,bs=64,path=None):
    actor.eval();ranks=[];nll=[];pred=[];top=[]
    for begin in range(0,len(data[split+'_h']),bs):
        b=batch(data,split,np.arange(begin,min(begin+bs,len(data[split+'_h']))))
        # Ranking prefixes use model predictions only. Likelihood is separately teacher-forced.
        free=actor(b,K=k,greedy=True)
        law=free['log_distributions'];target=b['set_targets']-1
        true=law.gather(2,target[:,:,None]);ids=torch.arange(actor.n_items,device=law.device)[None,None]
        rank=1+(law>true).sum(-1)+((law==true)&(ids<target[:,:,None])).sum(-1)
        teacher=actor(b,actions0=target,K=k)
        ranks.append(rank.cpu().numpy());nll.append(-teacher['log_probs'].cpu().numpy())
        pred.append(free['actions0'].cpu().numpy()+1);top.append(law.topk(min(100,actor.n_items),dim=-1).indices.cpu().numpy()+1)
    r=np.concatenate(ranks);n=np.concatenate(nll)
    if path:np.savez_compressed(path,ranks=r,teacher_nll=n,predictions=np.concatenate(pred),top100=np.concatenate(top),targets=data[split+'_y'])
    return metrics_from_ranks(r,n)

def construct(data,k,dim):
    codes=torch.as_tensor(data['codes'],dtype=torch.long)
    dc=F.pad(codes,(0,0,1,0),value=-1);tokens=dc+torch.arange(16)[None]*256+1;tokens[0]=0
    text=F.pad(torch.as_tensor(data['text'],dtype=torch.float32),(0,0,1,0))
    return SIDUBTTrajectory(len(codes),tokens,dc,text,n_embd=dim,n_layer=2,n_head=4,n_inner=1024,
        embd_pdrop=.5,attn_pdrop=.5,resid_pdrop=0.,temperature=.03,item_ce_lambda=.5,
        item_ce_tau=.15,item_ce_ls=0.,align_lambda=.1,mtp_depths=k,mtp_lambda=0.,
        mtp_detach=0,detach_feedback=False,input_pooling='attn',token_ce_lambda=1.,ce_chunk_size=256,
        ubt_eta=0.,ubt_policy_temperature=.15,ubt_kernel_temperature=.2,mu_cond_drop=.25,ubt_null_aux_lambda=.1).cuda()

def supervised(args,data,folder):
    model=construct(data,args.k,args.dim);actor=Actor(model).cuda()
    logs=[]
    for stage,epochs in [('dense',120),('trajectory',24),('continuation',16)]:
        if stage=='continuation':
            with torch.no_grad():actor.feedback_table=model._input_embs(torch.arange(1,actor.n_items+1,device='cuda')[None]).squeeze(0).detach().clone()
        for name,p in model.named_parameters():p.requires_grad_(stage!='continuation' or name.startswith('mu_'))
        lr=.003 if stage=='dense' else .0001
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0.)
        split='dense' if stage=='dense' else 'train';best=-1;bad=0
        for epoch in range(1,epochs+1):
            actor.eval();model.train(stage=='dense')
            order=np.random.default_rng(2024+epoch).permutation(len(data[split+'_h']));losses=[];start=time.time()
            for begin in range(0,len(order),args.batch):
                rr=order[begin:begin+args.batch];b=batch(data,split,rr);optimizer.zero_grad(set_to_none=True)
                if stage=='dense':
                    b['labels']=torch.as_tensor(data['dense_labels'][rr].astype(np.int64),device='cuda');b.pop('set_targets')
                    # Dense next-item initialization only: stochastic supervised encoder.
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(b).loss
                else:
                    loss=-actor(b,actions0=b['set_targets']-1,K=args.k)['log_probs'].mean()
                    if stage=='trajectory':
                        # Retain dense next-item and alignment supervision while fitting the trajectory.
                        di=np.random.default_rng(2024000+epoch*10000+begin).choice(len(data['dense_h']),len(rr))
                        dense=batch(data,'dense',di);dense['labels']=torch.as_tensor(data['dense_labels'][di].astype(np.int64),device='cuda');dense.pop('set_targets')
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss+model(dense).loss
                        loss=loss+.1*(-actor(b,actions0=b['set_targets']-1,K=args.k,condition_drop=.25)['log_probs'].mean())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite supervised loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            # Whole-trajectory validation selection, fixed before TEST is read.
            val=evaluate(actor,data,'val',args.k);score=val['SNDCG@10']
            improved=score>best
            if improved:
                best=score;bad=0;torch.save({'state':actor.state_dict(),'feedback':actor.feedback_table,'stage':stage,'epoch':epoch},folder/(stage+'.pt'))
            else:bad+=1
            logs.append({'stage':stage,'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-start,'val':val})
            save(folder/'supervised_progress.json',logs)
            print(json.dumps({'stage':stage,'epoch':epoch,'loss':np.mean(losses),'SNDCG10':score,'seconds':time.time()-start}),flush=True)
            if bad>=8:break
        saved=torch.load(folder/(stage+'.pt'),weights_only=False);actor.load_state_dict(saved['state'])
    for p in actor.parameters():p.requires_grad_(True)
    actor.model.mu_null.requires_grad_(False)
    base=evaluate(actor,data,'test',args.k,path=folder/'base_predictions.npz')
    save(folder/'base_metrics.json',base)
    # Same configured local-feedback policy initializes both pure-RL variants.
    actor.model.ubt_eta=.75
    save(folder/'rl_initial_metrics.json',evaluate(actor,data,'test',args.k,path=folder/'rl_initial_predictions.npz'))
    torch.save({'state':actor.state_dict(),'feedback':actor.feedback_table,'eta':.75},folder/'stage1.pt')
    save(folder/'stage1_complete.json',{'status':'complete','base_metrics':base})

def make_reward(data):
    n=len(data['text']);counts=np.zeros((n,n),np.float32)
    np.add.at(counts,(data['edges'][:,0]-1,data['edges'][:,1]-1),1.)
    out=counts.sum(1,keepdims=True);pop=(counts.sum(0)+1)/(counts.sum()+n)
    lift=out/(out+10)*np.tanh(np.log((counts+10*pop)/(out+10)/pop)/3)
    return RelationReward(torch.as_tensor(data['text'],device='cuda'),torch.as_tensor(lift,device='cuda'))

def scorer_fit(args,data,folder):
    # Independent scorer parameters; all examples strictly before validation boundary.
    raw=torch.as_tensor(data['text'],device='cuda');g=torch.Generator().manual_seed(1729)
    projection=torch.randn(raw.shape[1],64,generator=g).cuda()/8
    scorer=ContextRelationScorer(F.normalize(F.normalize(raw,dim=-1)@projection,dim=-1)).cuda()
    n=len(data['scorer_a']);fit=np.flatnonzero(data['scorer_uid']%5!=0);hold=np.flatnonzero(data['scorer_uid']%5==0)
    report=[]
    for stage in ('base','relation'):
        scorer.set_stage(stage);opt=torch.optim.AdamW([p for p in scorer.parameters() if p.requires_grad],lr=.003,weight_decay=.001)
        best=math.inf;bad=0
        for epoch in range(1,21):
            losses=[]
            for rr in np.array_split(np.random.default_rng(2024+epoch).permutation(fit),max(1,math.ceil(len(fit)/256))):
                h=torch.as_tensor(data['scorer_h'][rr].astype(np.int64),device='cuda');a=torch.as_tensor(data['scorer_a'][rr].astype(np.int64),device='cuda');y=torch.as_tensor(data['scorer_b'][rr].astype(np.int64),device='cuda')
                full,base=scorer.distributions(h,a,'base' if stage=='base' else 'full');loss=F.nll_loss(full,y-1)
                if stage=='relation':loss=loss+.01*(full-base).square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(scorer.parameters(),5.);opt.step();losses.append(float(loss.detach()))
            values=[]
            with torch.no_grad():
                for rr in np.array_split(hold,max(1,math.ceil(len(hold)/256))):
                    h=torch.as_tensor(data['scorer_h'][rr].astype(np.int64),device='cuda');a=torch.as_tensor(data['scorer_a'][rr].astype(np.int64),device='cuda');y=torch.as_tensor(data['scorer_b'][rr].astype(np.int64),device='cuda')
                    lp,_=scorer.distributions(h,a,'base' if stage=='base' else 'full');values.extend((-lp.gather(1,(y-1)[:,None])).cpu().flatten().tolist())
            score=float(np.mean(values));report.append({'stage':stage,'epoch':epoch,'fit_loss':float(np.mean(losses)),'hold_nll':score})
            print(json.dumps(report[-1]),flush=True);save(folder/'scorer_progress.json',report)
            if score<best:best=score;bad=0;torch.save(scorer.state_dict(),folder/(stage+'_scorer.pt'))
            else:bad+=1
            if bad>=4:break
        scorer.load_state_dict(torch.load(folder/(stage+'_scorer.pt'),weights_only=False))
    torch.save({'state':scorer.state_dict(),'projection':projection.cpu()},folder/'scorer.pt')

def rl(args,data,folder):
    saved=torch.load(folder/'stage1.pt',weights_only=False,map_location='cuda');actor=Actor(construct(data,args.k,args.dim),saved['feedback']).cuda();actor.load_state_dict(saved['state']);actor.model.ubt_eta=saved['eta'];actor.eval();actor.model.mu_null.requires_grad_(False)
    reward=make_reward(data)
    if args.mode=='context':
        payload=torch.load(folder/'scorer.pt',weights_only=False,map_location='cuda');features=F.normalize(F.normalize(torch.as_tensor(data['text'],device='cuda'),dim=-1)@payload['projection'],dim=-1)
        scorer=ContextRelationScorer(features).cuda();scorer.load_state_dict(payload['state']);scorer.set_stage('frozen').eval();reward=ContextRelationReward(reward,scorer)
    groups=[{'params':[p for n,p in actor.named_parameters() if p.requires_grad and not n.startswith('model.mu_')],'lr':1e-5},
            {'params':[p for n,p in actor.named_parameters() if p.requires_grad and n.startswith('model.mu_')],'lr':1e-4}]
    opt=torch.optim.AdamW(groups,weight_decay=0.);logs=[];best=-1
    arm=folder/args.mode;arm.mkdir(exist_ok=False)
    for epoch in range(1,7):
        order=np.random.default_rng(2024+epoch).permutation(len(data['train_h']));losses=[];start=time.time()
        for begin in range(0,len(order),args.rl_batch):
            b=batch(data,'train',order[begin:begin+args.rl_batch]);opt.zero_grad(set_to_none=True)
            # Pure on-policy integrated RL: no logged CE, alignment, dropout auxiliary or KL loss.
            loss,detail=policy_loss(actor,b,reward,'integrated',20240000+epoch*100000+begin,group=8)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        val=evaluate(actor,data,'val',args.k);score=val['SNDCG@10']
        if score>best:
            best=score;torch.save({'state':actor.state_dict(),'feedback':actor.feedback_table,'epoch':epoch,'eta':.75},arm/'selected.pt')
        logs.append({'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-start,'val':val,'CE_coefficient':0,'KL_coefficient':0,'group':8})
        save(arm/'progress.json',logs);print(json.dumps({'mode':args.mode,'epoch':epoch,'SNDCG10':score,'seconds':time.time()-start}),flush=True)
    save(arm/'endpoint6_metrics.json',evaluate(actor,data,'test',args.k,path=arm/'endpoint6_predictions.npz'))
    chosen=torch.load(arm/'selected.pt',weights_only=False);actor.load_state_dict(chosen['state'])
    result={'status':'complete','selected_epoch':chosen['epoch'],'seed':2024,'CE_coefficient':0,'KL_coefficient':0,'metrics':evaluate(actor,data,'test',args.k,path=arm/'selected_predictions.npz')}
    save(arm/'result.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--k',type=int,required=True);p.add_argument('--mode',choices=('base','rl','scorer','context'),required=True);p.add_argument('--dim',type=int,default=448);p.add_argument('--batch',type=int,default=128);p.add_argument('--rl-batch',type=int,default=32);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('A CUDA-capable GPU is required')
    torch.set_num_threads(4);torch.manual_seed(2024);np.random.seed(2024);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    from paths import DATA
    folder=OUT/a.dataset;folder.mkdir(parents=True,exist_ok=True);data=dict(np.load(DATA/a.dataset/'assets.npz',allow_pickle=False))
    assert a.k in (3,5) and data['train_y'].shape[1]==a.k
    if a.mode=='base':supervised(a,data,folder)
    elif a.mode=='scorer':scorer_fit(a,data,folder)
    else:rl(a,data,folder)

if __name__=='__main__':main()
