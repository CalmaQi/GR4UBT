"""Task-only action rewards, fresh on-policy integration and item-level GRPO.

No diversity, redundancy, co-occurrence or external semantic scorer is used.
All catalog actions are complete items; deterministic local feedback stays live.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from ubt_bapo_relations import credit_surrogate


class TaskReward:
    def __init__(self,codes,sid_weight=0.,support_reference=None,support_mode='logprob'):
        self.codes=torch.as_tensor(codes,dtype=torch.long)
        self.n_items=len(codes);self.sid_weight=float(sid_weight);self.support_reference=support_reference;self.support_mode=support_mode
    @torch.no_grad()
    def catalog(self,actions,targets1,last0,t):
        b,g,k=actions.shape
        target=targets1[:,t]-1
        hit=F.one_hot(target,self.n_items).float()
        if self.sid_weight:
            sid=(self.codes[None]==self.codes[target,None]).float().mean(-1)
            hit=hit+self.sid_weight*sid
        return hit[:,None].expand(b,g,-1)/k
    @torch.no_grad()
    def steps(self,actions,targets1):
        y=(targets1-1)[:,None]
        exact=(actions==y).float()
        sid=(self.codes[actions]==self.codes[y]).float().mean(-1)
        return exact+self.sid_weight*sid,exact,sid
    @torch.no_grad()
    def global_reward(self,actions,targets1):
        return torch.zeros(actions.shape[:2],device=actions.device)


@torch.no_grad()
def sample_group(actor,batch,reward,seed,group=16):
    actor.eval();k=batch['set_targets'].shape[1];b=len(batch['input_ids'])
    rng=torch.Generator(device=batch['input_ids'].device).manual_seed(seed)
    h=actor.encode_history(batch).repeat_interleave(group,0)
    out=actor.trajectory_from_hidden(h,K=k,generator=rng)
    actions=out['actions0'].reshape(b,group,k)
    steps,hit,sid=reward.steps(actions,batch['set_targets'])
    returns=steps.mean(-1);support=torch.zeros_like(returns)
    if reward.support_reference is not None:
        ref=reward.support_reference
        h0=ref.encode_history(batch).repeat_interleave(group,0)
        scored=ref.trajectory_from_hidden(h0,actions0=actions.reshape(b*group,k),K=k)
        targets=(batch['set_targets']-1).repeat_interleave(group,0)
        # Ground truth enters only AFTER the frozen reference has scored the
        # independently generated prefixes. It never conditions generation.
        logged=scored['log_distributions'].gather(-1,targets[:,:,None]).squeeze(-1).reshape(b,group,k)
        if reward.support_mode=='rank':
            laws=scored['log_distributions'];true=laws.gather(-1,targets[:,:,None])
            ids=torch.arange(actor.n_items,device=targets.device)[None,None]
            ranks=1+(laws>true).sum(-1)+((laws==true)&(ids<targets[:,:,None])).sum(-1)
            support=(1/torch.log2(ranks.float()+1)).reshape(b,group,k)[:,:,1:].mean(-1)
        else:support=logged[:,:,1:].clamp(-20,0).mean(-1)/20.
        returns=returns+.2*support
    std=returns.std(1,unbiased=False,keepdim=True)
    active=std>1e-6
    advantages=torch.where(active,(returns-returns.mean(1,keepdim=True))/std.clamp_min(1e-6),torch.zeros_like(returns))
    return {'actions':actions,'old_logp':out['log_probs'].reshape(b,group,k).detach(),
        'advantages':advantages.detach(),'returns':returns,'hits':hit,'sid':sid,
        'active':active.squeeze(1),'reward_std':std.squeeze(1),'support':support}


def clipped_surrogate(logp,old_logp,advantages,clip=.1):
    """One trajectory advantage, one likelihood ratio per complete-item decision."""
    delta=logp-old_logp
    ratio=delta.exp()
    advantage=advantages[...,None]
    objective=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    loss=-objective.mean()
    with torch.no_grad():
        stats={'clip_fraction':float(((ratio<1-clip)|(ratio>1+clip)).float().mean()),
            'mean_ratio':float(ratio.mean()),'old_to_current_approx_KL':float((delta.exp()-delta-1).mean())}
    return loss,stats


def grpo_loss(actor,batch,samples,clip=.1):
    b,g,k=samples['actions'].shape
    h=actor.encode_history(batch).repeat_interleave(g,0)
    replay=actor.trajectory_from_hidden(h,actions0=samples['actions'].reshape(b*g,k),K=k)
    logp=replay['log_probs'].reshape(b,g,k)
    loss,stats=clipped_surrogate(logp,samples['old_logp'],samples['advantages'],clip)
    stats.update(active_group_fraction=float(samples['active'].float().mean()),
        sample_hit=float(samples['hits'].mean()),sample_sid_match=float(samples['sid'].mean()),
        reward=float(samples['returns'].mean()),group_reward_std=float(samples['reward_std'].mean()),
        frozen_future_support=float(samples['support'].mean()))
    return loss,stats


def integrated_loss(actor,batch,reward,seed,group=16):
    """Original unbiased full-catalog action integration, now with exact task utility."""
    samples=sample_group(actor,batch,reward,seed,group)
    b,g,k=samples['actions'].shape
    h=actor.encode_history(batch).repeat_interleave(g,0)
    out=actor.trajectory_from_hidden(h,actions0=samples['actions'].reshape(b*g,k),K=k)
    logp=out['log_probs'].reshape(b,g,k)
    if not torch.equal(logp.detach(),samples['old_logp']):raise ValueError('sample/replay mismatch')
    law=out['log_distributions'].reshape(b,g,k,actor.n_items)
    expected=[];sampled=[]
    last=batch['input_ids'].gather(1,(batch['seq_lens']-1)[:,None]).squeeze(1)-1
    for t in range(k):
        table=reward.catalog(samples['actions'],batch['set_targets'],last,t)
        expected.append((law[:,:,t].exp()*table).sum(-1))
        sampled.append(table.gather(-1,samples['actions'][:,:,t,None]).squeeze(-1))
    expected=torch.stack(expected,-1);sampled=torch.stack(sampled,-1)
    terminal=torch.zeros((b,g),device=logp.device)
    loss=credit_surrogate(logp,sampled,expected,terminal,'integrated').mean()
    return loss,{'sample_hit':float(samples['hits'].mean()),'expected_hit':float(expected.detach().sum(-1).mean()),
        'active_group_fraction':float(samples['active'].float().mean()),'sample_replay_equal':True}


def balanced_rows(data,seed,per_user=2):
    """Match user-averaged evaluation: do not let the longest users dominate RL."""
    import numpy as np
    counts=np.maximum(data['train_ends']-data['train_y'].shape[1],0)
    offsets=np.r_[0,np.cumsum(counts)]
    if int(offsets[-1])!=len(data['train_y']):raise ValueError('query/user boundary mismatch')
    rng=np.random.default_rng(seed);rows=[]
    for begin,count in zip(offsets[:-1],counts):
        if count:rows.extend((begin+rng.choice(int(count),min(per_user,int(count)),replace=False)).tolist())
    return rng.permutation(np.asarray(rows,dtype=np.int64))
