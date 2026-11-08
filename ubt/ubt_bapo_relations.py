"""Behavior-relation rewards and item-law credit with live SID feedback."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


def build_relations(metadata, features, shrinkage=10.):
    n = len(features)
    edges = {}
    for uid, start, end, history, length, target, times in zip(
            metadata['user_hashes'], metadata['target_start'], metadata['train_end_exclusive'],
            metadata['histories1'], metadata['lengths'], metadata['targets1'], metadata['boundary_timestamps']):
        start, end, length = int(start), int(end), int(length)
        if length < 1 or start + len(target) > end or not np.all(np.diff(times) > 0):
            raise ValueError('strict TRAIN boundaries required')
        path = [int(history[length-1])] + list(map(int, target))
        for t, (left, right) in enumerate(zip(path, path[1:])):
            key = (int(uid), start - 1 + t)
            if not 1 <= min(left, right) <= max(left, right) <= n:
                raise ValueError('invalid catalogue ID')
            if key in edges and edges[key] != (left-1, right-1):
                raise ValueError('inconsistent TRAIN transition')
            edges[key] = (left-1, right-1)
    pairs = np.asarray(list(edges.values()), dtype=np.int64)
    counts = np.zeros((n, n), dtype=np.float32)
    np.add.at(counts, (pairs[:, 0], pairs[:, 1]), 1.)
    outgoing = counts.sum(1, keepdims=True)
    popularity = (counts.sum(0) + 1.) / (len(pairs) + n)
    conditional = (counts + shrinkage * popularity[None]) / (outgoing + shrinkage)
    lift = np.log(conditional / popularity[None])
    lift = (outgoing / (outgoing + shrinkage)) * np.tanh(lift / 3.)
    device = features.device
    return RelationReward(features, torch.from_numpy(lift).to(device), {
        'source': 'strict TRAIN boundary edges only, deduplicated UID/position',
        'unique_edges': len(edges), 'shrinkage': shrinkage, 'lift_scale': 3.,
        'edge_sha256': hashlib.sha256(pairs.tobytes()).hexdigest(),
        'transition_sha256': hashlib.sha256(lift.tobytes()).hexdigest(),
        'weights': {'relevance': 1., 'transition': .25, 'redundancy': .10, 'global': .25}})


class RelationReward:
    def __init__(self, features, transition, metadata=None):
        self.features = F.normalize(features.detach().float(), dim=-1)
        self.transition = transition.detach().float()
        self.metadata = metadata or {}
        self.n_items = len(features)
        if transition.shape != (self.n_items, self.n_items):
            raise ValueError('full directed transition table required')

    @torch.no_grad()
    def catalog(self, actions, targets1, last0, t):
        b, g, k = actions.shape
        relevance = self.features[targets1[:, t]-1] @ self.features.T
        relevance = ((relevance + 1.) / 2.)[:, None].expand(b, g, -1)
        previous = last0[:, None].expand(b, g) if t == 0 else actions[:, :, t-1]
        transition = self.transition[previous]
        redundant = torch.zeros_like(transition)
        for j in range(t):
            cosine = self.features[actions[:, :, j]] @ self.features.T
            redundant = torch.maximum(redundant, ((cosine - .9) / .1).clamp(0., 1.))
        return (relevance + .25 * transition - .10 * redundant) / k

    @torch.no_grad()
    def global_reward(self, actions, targets1):
        x = self.features[actions]
        y = self.features[targets1-1][:, None]
        distance = (x-y).square().sum(-1).mean(-1)
        exact = (actions == (targets1-1)[:, None]).all(-1).float()
        return .25 * (torch.exp(-distance / 2.) + .05 * exact)


def credit_surrogate(logp, sampled, expected, terminal, estimator, baseline=True):
    if estimator not in ('sampled', 'integrated'):
        raise ValueError('unknown estimator')
    k = logp.shape[-1]
    values = sampled if estimator == 'sampled' else expected
    cumulative = values.flip(-1).cumsum(-1).flip(-1)
    returns = cumulative if estimator == 'sampled' else cumulative - values
    returns = returns + terminal[..., None]
    if baseline:
        g = logp.shape[1]
        if g < 2:
            raise ValueError('independent trajectory LOO needs two draws')
        total = values.sum(-1) + terminal
        other = (total.sum(1, keepdim=True)-total) / (g-1)
        returns = returns - other[..., None]
    per_path = -(returns.detach() * logp).sum(-1)
    if estimator == 'integrated':
        per_path = per_path - expected.sum(-1)
    return per_path


def policy_loss(actor, batch, reward, estimator, seed, group=8, return_tensors=False):
    if getattr(actor, 'generation_order', (1, 2, 3)) != (1, 2, 3):
        raise ValueError('chronological actor required')
    history = {key: batch[key] for key in ('input_ids', 'attention_mask', 'seq_lens')}
    if hasattr(reward, 'bind_history'):
        reward = reward.bind_history(history)
    b = len(history['input_ids'])
    targets = batch['set_targets']
    k = targets.shape[1]
    rng = torch.Generator(device=targets.device).manual_seed(seed)
    with torch.no_grad():
        sampled_h = actor.encode_history(history).repeat_interleave(group, 0)
        draw = actor.trajectory_from_hidden(sampled_h, K=k, generator=rng, return_trace=False)
        actions = draw['actions0'].reshape(b, group, k)
    hidden = actor.encode_history(history).repeat_interleave(group, 0)
    replay = actor.trajectory_from_hidden(hidden, actions0=actions.reshape(b*group, k), K=k, return_trace=False)
    if not torch.equal(draw['log_probs'], replay['log_probs'].detach()):
        raise ValueError('sample/replay laws differ')
    distributions = replay['log_distributions'].reshape(b, group, k, actor.n_items)
    logp = replay['log_probs'].reshape(b, group, k)
    last = history['input_ids'].gather(1, (history['seq_lens']-1)[:, None]).squeeze(1)-1
    sampled_values, expected_values = [], []
    for t in range(k):
        table = reward.catalog(actions, targets, last, t)
        dtype = torch.float64 if t == 0 else torch.float32
        probability = distributions[:, :, t].to(dtype).exp()
        probability = probability / probability.sum(-1, keepdim=True)
        sampled_values.append(table.gather(-1, actions[:, :, t, None]).squeeze(-1))
        expected_values.append((probability * table).sum(-1))
    sampled_values = torch.stack(sampled_values, -1)
    expected_values = torch.stack(expected_values, -1)
    terminal = reward.global_reward(actions, targets)
    loss = credit_surrogate(logp, sampled_values, expected_values, terminal, estimator).mean()
    detail = {'estimator': estimator, 'surrogate': float(loss.detach()),
        'sample_reward': float((sampled_values.sum(-1)+terminal).mean()),
        'integrated_reward': float((expected_values.sum(-1)+terminal).mean()),
        'per_step_sample': sampled_values.mean((0, 1)).tolist(),
        'per_step_expected': expected_values.detach().mean((0, 1)).tolist(),
        'global_reward': float(terminal.mean()), 'group': group,
        'sample_replay_equal': True, 'own_prefix_full_graph': True}
    if return_tensors:
        detail['tensors'] = {'actions': actions, 'logp': logp, 'expected': expected_values,
            'sampled': sampled_values, 'terminal': terminal, 'replay': replay}
    return loss, detail
