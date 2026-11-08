"""Same-horizon conditional log-lift for frozen trajectory rewards."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def relation_examples(metadata, max_history=20):
    records = {}
    for uid, start, end, history, length, targets, times in zip(
            metadata['user_hashes'], metadata['target_start'], metadata['train_end_exclusive'],
            metadata['histories1'], metadata['lengths'], metadata['targets1'],
            metadata['boundary_timestamps']):
        start, end, length = int(start), int(end), int(length)
        if length < 1 or start + len(targets) > end or not np.all(np.diff(times) > 0):
            raise ValueError('strict TRAIN windows required')
        prefix = list(map(int, history[:length]))
        for t, target in enumerate(targets):
            a, context = prefix[-1], prefix[:-1][-max_history:]
            key = (int(uid), start + t)
            value = (context, a, int(target))
            if key not in records or len(context) > len(records[key][0]):
                records[key] = value
            elif records[key][1:] != value[1:]:
                raise ValueError('inconsistent repeated event')
            prefix.append(int(target))
    histories = np.zeros((len(records), max_history), dtype=np.int64)
    users, actions, targets = [], [], []
    for row, ((uid, _), (context, a, b)) in enumerate(sorted(records.items())):
        histories[row, :len(context)] = context
        users.append(uid); actions.append(a); targets.append(b)
    return {'history': histories, 'a': np.asarray(actions), 'b': np.asarray(targets),
            'uid': np.asarray(users, dtype=np.uint64)}

class ContextRelationScorer(nn.Module):
    def __init__(self, features, hidden=64):
        super().__init__()
        features = F.normalize(features.detach().float(), dim=-1)
        self.register_buffer('features', F.pad(features, (0, 0, 1, 0)))
        d = features.shape[-1]
        self.context = nn.GRU(d, hidden, batch_first=True)
        self.base_query = nn.Linear(hidden, d)
        self.base_bias = nn.Parameter(torch.zeros(len(features)))
        self.action = nn.Linear(d, hidden, bias=False)
        self.gate = nn.Linear(hidden, hidden)
        self.relation_out = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.relation_out.weight)

    def context_state(self, histories):
        mask = histories > 0
        states, _ = self.context(self.features[histories])
        lengths = mask.sum(-1)
        last = states[torch.arange(len(histories), device=histories.device),
                      (lengths - 1).clamp_min(0)]
        return last * (lengths > 0)[:, None]

    def distributions(self, histories, a, mode='full'):
        h = self.context_state(histories)
        base = self.base_query(h) @ self.features[1:].T + self.base_bias
        if mode == 'base':
            return F.log_softmax(base, -1), F.log_softmax(base, -1)
        if mode == 'no_context':
            h = torch.zeros_like(h)
        update = self.action(self.features[a]) * torch.sigmoid(self.gate(h))
        residual = self.relation_out(update) @ self.features[1:].T
        return F.log_softmax(base + residual, -1), F.log_softmax(base, -1)

    def set_stage(self, stage):
        if stage not in ('base', 'relation', 'frozen'):
            raise ValueError('unknown scorer stage')
        self.requires_grad_(False)
        modules = (self.context, self.base_query) if stage == 'base' else (
            self.action, self.gate, self.relation_out) if stage == 'relation' else ()
        for module in modules:
            module.requires_grad_(True)
        self.base_bias.requires_grad_(stage == 'base')
        return self

    @torch.no_grad()
    def lift(self, histories, a, mode='full'):
        full, base = self.distributions(histories, a, mode)
        return torch.tanh((full - base) / 3.)

class ContextRelationReward:
    def __init__(self, original, scorer, weight=.25, mode='full'):
        self.original, self.scorer = original, scorer
        self.weight, self.mode = weight, mode
        self.n_items = original.n_items
        self.features = original.features
        if any(p.requires_grad for p in scorer.parameters()):
            raise ValueError('reward scorer must be frozen')

    def bind_history(self, history):
        return BoundContextReward(self, history)


class BoundContextReward:
    def __init__(self, parent, history):
        self.parent = parent
        self.history = history

    @torch.no_grad()
    def catalog(self, actions, targets1, last0, t):
        p = self.parent
        b, g, k = actions.shape
        prefixes = []
        for row in range(b):
            length = int(self.history['seq_lens'][row])
            observed = self.history['input_ids'][row, :length]
            for draw in range(g):
                prefix = torch.cat((observed, actions[row, draw, :t] + 1))
                prefixes.append(prefix)
        histories = torch.zeros(b*g, 20, dtype=torch.long, device=actions.device)
        previous = torch.empty(b*g, dtype=torch.long, device=actions.device)
        for row, prefix in enumerate(prefixes):
            context = prefix[:-1][-20:]
            histories[row, :len(context)] = context
            previous[row] = prefix[-1]
        lift = p.scorer.lift(histories, previous, p.mode).reshape(b, g, -1)
        fixed = p.original.catalog(actions, targets1, last0, t)
        previous0 = last0[:, None].expand(b, g) if t == 0 else actions[:, :, t-1]
        return fixed - .25 * p.original.transition[previous0] / k + p.weight * lift / k

    def global_reward(self, actions, targets1):
        return self.parent.original.global_reward(actions, targets1)
