"""Action-conditioned UBT trajectories with one shared SID catalogue readout.

Item IDs are zero based in rollout APIs (``actions0``/``prefix0``) and one
based in dataset targets. ``logits`` always means logits of the *actual*
sampling law, including its single temperature and optional no-repeat mask.
No independent item classifier, candidate reranking, or future labels enter
generation. A frozen random projection of the supplied Sentence-T5 text asset
defines local semantic neighbourhoods; trainable SID means provide feedback.
"""
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

from sidgen_mtp import N_CODEBOOK
from sidgen_mu_joint import SIDMuJoint


class SIDUBTTrajectory(SIDMuJoint):
    def __init__(self, *args, ubt_eta=0.75, ubt_semantic_dim=64,
                 ubt_semantic_seed=1729, ubt_semantic_chunk=1024,
                 ubt_kernel_temperature=0.2, ubt_policy_temperature=None,
                 ubt_no_repeat=False, ubt_null_aux_lambda=0.1,
                 mu_cond_drop=0.25, **kwargs):
        if not 0 <= ubt_eta < 1:
            raise ValueError("ubt_eta must lie in [0, 1); retain a real-item residual")
        if ubt_semantic_dim < 1 or ubt_semantic_chunk < 1:
            raise ValueError("semantic projection and chunk sizes must be positive")
        if ubt_kernel_temperature <= 0 or ubt_null_aux_lambda < 0:
            raise ValueError("kernel temperature positive; null auxiliary weight nonnegative")
        if kwargs.get("mtp_detach", 0) or kwargs.get("detach_feedback", False):
            raise ValueError("UBT requires intact history and probability-to-feedback gradients")
        kwargs["mtp_detach"] = 0
        kwargs["detach_feedback"] = False
        kwargs["clone_future"] = False
        super().__init__(*args, mu_cond_drop=mu_cond_drop, **kwargs)
        self.ubt_eta = float(ubt_eta)
        self.ubt_kernel_temperature = float(ubt_kernel_temperature)
        self.ubt_policy_temperature = float(
            self.item_ce_tau if ubt_policy_temperature is None else ubt_policy_temperature)
        if self.ubt_policy_temperature <= 0:
            raise ValueError("policy temperature must be positive")
        # All full-item CE terms use the same law as actual sampling.
        self.item_ce_tau = self.ubt_policy_temperature
        self.mtp_tau = self.ubt_policy_temperature
        self.ubt_no_repeat = bool(ubt_no_repeat)
        self.ubt_null_aux_lambda = float(ubt_null_aux_lambda)
        self.ubt_semantic_seed = int(ubt_semantic_seed)
        self._tie_readout()
        if not self.use_text:
            raise ValueError("UBT needs the complete frozen Sentence-T5 item text asset")
        if self.text_embs.shape[0] != self.n_items + 1:
            raise ValueError("text asset must contain PAD plus every catalogue item")
        # Construct N x d once, without N x N or N x L x D allocations.
        # The fixed projection uses neither labels nor a fitted recommendation model.
        generator = torch.Generator(device="cpu").manual_seed(self.ubt_semantic_seed)
        projection = torch.randn(self.text_dim, int(ubt_semantic_dim),
                                 generator=generator, dtype=torch.float32)
        projection = projection / float(ubt_semantic_dim) ** 0.5
        projection = projection.to(self.text_embs.device)
        features = []
        with torch.no_grad(), torch.autocast(device_type=self.text_embs.device.type, enabled=False):
            for start in range(1, self.n_items + 1, int(ubt_semantic_chunk)):
                text = self.text_embs[start:start + int(ubt_semantic_chunk)].float()
                if not bool(torch.isfinite(text).all()) or bool((text.norm(dim=-1) == 0).any()):
                    raise ValueError("real-item semantic features must be finite and nonzero")
                features.append(F.normalize(F.normalize(text, dim=-1) @ projection, dim=-1))
        self.register_buffer("ubt_semantic_features", torch.cat(features, 0), persistent=False)

    def _tie_readout(self):
        # Aliases preserve compatibility with old diagnostics, but contain no
        # additional trainable parameters. future_step explicitly uses main_logits.
        self.mtp1_blocks = self.main_blocks
        self.mtp1_ln = self.main_ln
        self.mtp1_mlp = self.main_mlp

    @contextmanager
    def policy_mode(self):
        """Disable all stochastic layers without disabling gradients or losing modes."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            yield self
        finally:
            for module, training in modes:
                module.training = training

    def load_base_state(self, state, load_transition=False):
        """Strictly transfer the entire old next-item model, never its future head.

        A checkpoint wrapper containing ``state_dict`` is also accepted. Asset
        hashes are the runner's responsibility. By default the new transition
        keeps its zero output initialization; explicit transition transfer needs
        matching shapes (including horizon). Unrelated/missing base keys fail.
        """
        if "state_dict" in state:
            state = state["state_dict"]
        excluded = ("mtp1_", "mu_", "ubt_", "mtp_blocks.", "mtp_ln.",
                    "mtp_mlp.", "mtp_slot.", "cond_proj.", "cond_null.")
        current = self.state_dict()
        expected = {name for name in current if not name.startswith(excluded)}
        supplied = {name for name in state if not name.startswith(excluded)}
        missing, unexpected = sorted(expected - supplied), sorted(supplied - expected)
        if missing or unexpected:
            raise ValueError(f"base state mismatch: missing={missing}, unexpected={unexpected}")
        transfer = {name: state[name] for name in expected}
        if load_transition:
            transition = {name for name in current if name.startswith("mu_")}
            if not transition.issubset(state):
                raise ValueError("base checkpoint lacks the complete mu transition")
            transfer.update({name: state[name] for name in transition})
        for name, value in transfer.items():
            if value.shape != current[name].shape:
                raise ValueError(f"base tensor shape mismatch for {name}: {value.shape} vs {current[name].shape}")
        self.load_state_dict(transfer, strict=False)
        self._tie_readout()
        return {"loaded": sorted(transfer), "skipped": sorted(set(state) - set(transfer)),
                "transition_transferred": bool(load_transition), "shared_main_readout": True}

    def encode_history(self, batch, deterministic=True):
        """Encode only observed history; deterministic mode still builds a graph."""
        if deterministic:
            with self.policy_mode():
                return super().encode_history(batch)
        return super().encode_history(batch)

    def _transition_hidden(self, hidden, condition, depth, anchor=None):
        if not 2 <= int(depth) <= self.mtp_depths:
            raise ValueError("depth must lie in [2, mtp_depths]")
        if condition.shape != hidden.shape:
            raise ValueError("condition and hidden must have shape [batch, hidden_dim]")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            horizon = self.mu_horizon.weight[int(depth) - 2].expand(len(hidden), -1)
            joined = torch.cat((self.mu_ln_h(hidden.float()),
                                self.mu_ln_mu(condition.float()), horizon.float()), -1)
            next_hidden = hidden.float() + self.mu_out(F.gelu(self.mu_in(joined)))
        return next_hidden

    def future_step(self, hidden, condition, depth, anchor=None):
        next_hidden = self._transition_hidden(hidden, condition, depth, anchor=anchor)
        return next_hidden, self._score_hidden_at_depth(next_hidden, depth)

    def _readout_states_at_depth(self, hidden, depth):
        """Default: the same main SID readout at every item position.

        A private-future-head baseline may override this one routing point so
        teacher item/token CE and sampled/replayed trajectories stay identical.
        """
        if not 1 <= int(depth) <= self.mtp_depths:
            raise ValueError("readout depth must lie in [1, mtp_depths]")
        return self.main_states(hidden.unsqueeze(1)).squeeze(1)

    def _score_hidden_at_depth(self, hidden, depth):
        return self.score_catalog(self._readout_states_at_depth(hidden, depth))

    def local_feedback(self, probabilities, actions0, table=None, rho=None):
        """Compute pi-weighted semantic local mean and concrete-action feedback.

        ``rho`` is an explicit diagnostic override for the hard residual weight
        (default 1-eta). rho=1 is hard feedback, rho=0 is the soft-only ablation;
        the configured production model always has rho>0.
        """
        if table is None:
            table = self.item_feedback_table()
        hard_weight = 1.0 - self.ubt_eta if rho is None else float(rho)
        if not 0 <= hard_weight <= 1:
            raise ValueError("rho must lie in [0, 1]")
        actions0 = actions0.long()
        if actions0.shape != (len(probabilities),) or bool(
                ((actions0 < 0) | (actions0 >= self.n_items)).any()):
            raise ValueError("feedback actions must be zero-based real catalogue items")
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            semantic = self.ubt_semantic_features.float()
            cosine = semantic.index_select(0, actions0) @ semantic.T
            # Strictly positive kernel; clamp prevents float underflow for tiny tau.
            kernel = ((cosine.clamp(-1, 1) - 1.0) /
                      self.ubt_kernel_temperature).clamp_min(-60.0).exp()
            local = probabilities.float() * kernel
            local = local / local.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            mean = local @ table.float()
            hard = table.index_select(0, actions0).float()
            condition = hard_weight * hard + (1.0 - hard_weight) * mean
        return condition, mean, local

    def _policy_logits(self, scores, used=None):
        logits = scores.float() / self.ubt_policy_temperature
        return logits if used is None else logits.masked_fill(used, -torch.inf)

    def trajectory_from_hidden(self, hidden, actions0=None, K=None, generator=None,
                               prefix0=None, deterministic=True, return_trace=True,
                               condition_drop=0.0, rho=None):
        """Replay actions or sample actual items after an optional fixed prefix.

        Returns logits [B,K,N], selected log_probs/entropy/actions0 [B,K],
        and (when requested) graph-connected hidden/probabilities/conditions.
        At position t the distribution is computed before inspecting action t;
        only action t and pi_t determine the transition to position t+1.
        ``actions0`` and ``prefix0`` are mutually exclusive. A supplied full
        path is replayed, with no graph truncation of its deterministic state.
        """
        if actions0 is not None and prefix0 is not None:
            raise ValueError("supply a complete action path OR a fixed prefix")
        if K is None:
            K = actions0.shape[-1] if actions0 is not None else self.mtp_depths
        K = int(K)
        if not 1 <= K <= self.mtp_depths:
            raise ValueError("trajectory K must lie in [1, mtp_depths]")
        if self.ubt_no_repeat and K > self.n_items:
            raise ValueError("no-repeat trajectory exceeds catalogue size")
        if not 0 <= condition_drop < 1:
            raise ValueError("condition_drop must lie in [0, 1)")
        if deterministic and condition_drop:
            raise ValueError("policy replay cannot apply a stochastic condition mask")
        fixed = actions0 if actions0 is not None else prefix0
        if fixed is not None:
            if fixed.ndim != 2 or len(fixed) != len(hidden) or fixed.shape[1] > K:
                raise ValueError("fixed actions must have shape [batch, prefix_length<=K]")
            if actions0 is not None and fixed.shape[1] != K:
                raise ValueError("actions0 is a complete replay path and must have K positions")
            if bool(((fixed < 0) | (fixed >= self.n_items)).any()):
                raise ValueError("rollout actions must be zero-based real item IDs")
            fixed = fixed.to(device=hidden.device, dtype=torch.long)
        if deterministic:
            with self.policy_mode():
                return self._trajectory(hidden, fixed, K, generator, return_trace, 0.0, rho)
        return self._trajectory(hidden, fixed, K, generator, return_trace, condition_drop, rho)

    def _trajectory(self, hidden, fixed, K, generator, return_trace, condition_drop, rho):
        anchor = hidden
        table = self.item_feedback_table() if K > 1 else None
        scores = self.main_logits(hidden)
        used = torch.zeros_like(scores, dtype=torch.bool) if self.ubt_no_repeat else None
        logits_all, logps, entropies, chosen = [], [], [], []
        states, probabilities, means, conditions = [hidden], [], [], []
        for t in range(K):
            logits = self._policy_logits(scores, used)
            logp = F.log_softmax(logits, -1)
            p = logp.exp()
            action = (fixed[:, t] if fixed is not None and t < fixed.shape[1] else
                      torch.multinomial(p, 1, generator=generator).squeeze(1))
            selected = logp.gather(1, action[:, None]).squeeze(1)
            if bool((~torch.isfinite(selected)).any()):
                raise ValueError("fixed prefix contains an action outside policy support")
            logits_all.append(logits)
            logps.append(selected)
            entropies.append(-(p * torch.where(torch.isfinite(logp), logp,
                                               torch.zeros_like(logp))).sum(-1))
            chosen.append(action)
            if return_trace:
                probabilities.append(p)
            if t + 1 < K:
                condition, mean, _ = self.local_feedback(p, action, table, rho=rho)
                if condition_drop:
                    drop = torch.rand(len(hidden), device=hidden.device,
                                      generator=generator) < condition_drop
                    condition = torch.where(drop[:, None], self.mu_null[None, :], condition)
                hidden, scores = self.future_step(hidden, condition, t + 2, anchor=anchor)
                if return_trace:
                    conditions.append(condition)
                    means.append(mean)
                    states.append(hidden)
                if used is not None:
                    # Clone: an in-place update must never invalidate a mask
                    # saved for the differentiable masked_fill of an earlier pi.
                    used = used.scatter(1, action[:, None], True)
        result = {"actions0": torch.stack(chosen, 1), "logits": torch.stack(logits_all, 1),
                  "log_probs": torch.stack(logps, 1), "entropy": torch.stack(entropies, 1)}
        if return_trace:
            result.update(hidden=states, probabilities=probabilities, mu=means,
                          conditions=conditions, feedback_table=table)
        return result

    @torch.no_grad()
    def sample_trajectories(self, batch, k, group, generator=None, prefix0=None,
                            return_trace=False):
        """IID full trajectory draws [batch,group,k]; never reads future labels."""
        if int(group) < 1:
            raise ValueError("group must be positive")
        group = int(group)
        hidden = self.encode_history(batch, deterministic=True)
        batch_size = len(hidden)
        fixed = None
        if prefix0 is not None:
            if prefix0.ndim == 2 and prefix0.shape[0] == batch_size:
                fixed = prefix0.repeat_interleave(group, 0)
            elif prefix0.ndim == 3 and prefix0.shape[:2] == (batch_size, group):
                fixed = prefix0.reshape(batch_size * group, prefix0.shape[-1])
            else:
                raise ValueError("prefix0 must have shape [B,T] or [B,group,T]")
        result = self.trajectory_from_hidden(hidden.repeat_interleave(group, 0), K=k,
                    prefix0=fixed, generator=generator, deterministic=True, return_trace=return_trace)
        for key in ("actions0", "log_probs", "entropy", "logits"):
            value = result[key]
            result[key] = value.reshape(batch_size, group, *value.shape[1:])
        return result

    def next_log_probs(self, batch, prefix0=None):
        """Graph-connected next policy after a zero-based prefix, without sampling."""
        # Keep both encoder and readout deterministic, including the no-prefix
        # fast path. encode_history alone restores the outer training modes.
        with self.policy_mode():
            return self._next_log_probs_in_policy_mode(batch, prefix0)

    def _next_log_probs_in_policy_mode(self, batch, prefix0):
        hidden = self.encode_history(batch, deterministic=True)
        if prefix0 is None:
            return F.log_softmax(self._policy_logits(self.main_logits(hidden)), -1)
        if prefix0.ndim != 2 or prefix0.shape[0] != len(hidden):
            raise ValueError("prefix0 must have shape [B,T]")
        length = prefix0.shape[1]
        if length >= self.mtp_depths:
            raise ValueError("next prediction exceeds configured horizon")
        # A placeholder current action is harmless: its distribution is scored
        # first and no transition follows it. For no-repeat choose a legal one.
        if self.ubt_no_repeat:
            used = torch.zeros(len(hidden), self.n_items, device=hidden.device, dtype=torch.bool)
            used = used.scatter(1, prefix0.to(hidden.device).long(), True)
            placeholder = (~used).long().argmax(-1, keepdim=True)
        else:
            placeholder = torch.zeros(len(hidden), 1, device=hidden.device, dtype=torch.long)
        fixed = torch.cat((prefix0.to(hidden.device), placeholder), -1)
        trace = self.trajectory_from_hidden(hidden, actions0=fixed, K=length + 1,
                                            deterministic=True, return_trace=False)
        return F.log_softmax(trace["logits"][:, -1], -1)

    def teacher_force_trajectory_loss(self, batch_or_hidden, targets1=None,
                                      start_depth=1, deterministic=True,
                                      condition_drop=0.0, token_ce_lambda=0.0):
        """Full-catalogue ordered CE on complete or right-padded partial paths.

        Each prediction sees only preceding valid targets. Missing targets end
        that row; gaps followed by positives are rejected. ``by_depth`` uses
        one-based prediction positions. No sampled wrong prefix is paired with
        an unrelated original suffix. The auxiliary token CE is optional.
        """
        if isinstance(batch_or_hidden, dict):
            if targets1 is None:
                targets1 = batch_or_hidden["set_targets"]
            hidden = self.encode_history(batch_or_hidden, deterministic=deterministic)
        else:
            hidden = batch_or_hidden
        if targets1 is None or targets1.ndim != 2 or len(targets1) != len(hidden):
            raise ValueError("targets1 must have shape [batch,horizon]")
        if not 1 <= targets1.shape[1] <= self.mtp_depths or not 1 <= start_depth <= self.mtp_depths:
            raise ValueError("supervised trajectory lies outside configured horizon")
        valid = targets1 > 0
        if bool((targets1 > self.n_items).any()):
            raise ValueError("target outside catalogue")
        if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("partial trajectories must be contiguous prefixes, not target gaps")
        if not 0 <= condition_drop < 1:
            raise ValueError("condition_drop must lie in [0, 1)")
        if deterministic and condition_drop:
            raise ValueError("deterministic CE cannot use stochastic condition masks")
        if deterministic:
            with self.policy_mode():
                return self._teacher_loss(hidden, targets1, start_depth, 0.0, token_ce_lambda)
        return self._teacher_loss(hidden, targets1, start_depth, condition_drop, token_ce_lambda)

    def _teacher_loss(self, hidden, targets1, start_depth, condition_drop, token_weight):
        zero = hidden.sum() * 0.0
        item_sum, token_sum, count = zero, zero, 0
        by_depth, token_by_depth, counts = {}, {}, {}
        active = torch.nonzero(targets1[:, 0] > 0, as_tuple=False).flatten()
        h = hidden.index_select(0, active)
        table = self.item_feedback_table()
        used = (torch.zeros(len(active), self.n_items, dtype=torch.bool, device=hidden.device)
                if self.ubt_no_repeat else None)
        for t in range(targets1.shape[1]):
            if len(active) == 0:
                break
            if token_weight:
                states = self._readout_states_at_depth(h, t + 1)
                scores = self.score_catalog(states)
            else:
                scores = self._score_hidden_at_depth(h, t + 1)
            logits = self._policy_logits(scores, used)
            yy = targets1[active, t].long()
            if t + 1 >= start_depth:
                loss = F.cross_entropy(logits, yy - 1, reduction="sum")
                by_depth[t + 1], counts[t + 1] = loss / len(active), len(active)
                item_sum, count = item_sum + loss, count + len(active)
                if token_weight:
                    normalized = F.normalize(states.float(), dim=-1)
                    tables = self._digit_embeddings()
                    with torch.autocast(device_type=h.device.type, enabled=False):
                        token_loss = torch.stack([F.cross_entropy(
                            normalized[:, digit] @ tables[digit].float().T / self.temperature,
                            self.item_dcodes[yy, digit], reduction="sum")
                            for digit in range(N_CODEBOOK)]).mean()
                    token_by_depth[t + 1] = token_loss / len(active)
                    token_sum = token_sum + token_loss
            if t + 1 == targets1.shape[1]:
                break
            keep = targets1[active, t + 1] > 0
            if not bool(keep.any()):
                break
            p = F.softmax(logits[keep], -1)
            action = yy[keep] - 1
            condition, _, _ = self.local_feedback(p, action, table)
            if condition_drop:
                dropped = torch.rand(len(condition), device=h.device) < condition_drop
                condition = torch.where(dropped[:, None], self.mu_null[None], condition)
            # The next loop computes its SID readout once, including optional
            # token auxiliary logits. Avoid a second discarded catalogue pass.
            h = self._transition_hidden(h[keep], condition, t + 2,
                                        anchor=hidden.index_select(0, active[keep]))
            active = active[keep]
            if used is not None:
                used = used[keep].scatter(1, action[:, None], True)
        item_loss, token_loss = item_sum / max(count, 1), token_sum / max(count, 1)
        return {"loss": item_loss + token_weight * token_loss, "item_loss": item_loss,
                "token_loss": token_loss, "by_depth": by_depth,
                "token_by_depth": token_by_depth, "counts": counts, "count": count}

    def forward(self, batch, return_loss=True):
        out = self._history_outputs(batch)
        hidden = out.last_hidden_for_agg
        if not return_loss:
            out.final_states = self.main_states(hidden)
            return out
        token, item = self._main_supervised_losses(hidden, batch["labels"])
        losses = {"token": token, "item": item,
                  "align": (self._align_loss(batch) if self.use_text and self.align_lambda > 0 else None)}
        total = self.token_ce_lambda * token + self.item_ce_lambda * item
        if losses["align"] is not None:
            total = total + self.align_lambda * losses["align"]
        targets = batch.get("set_targets")
        if targets is not None:
            h_last = self._h_last(out, batch)
            ordered = self.teacher_force_trajectory_loss(h_last, targets, start_depth=2,
                                                         deterministic=False)
            for depth, loss in ordered["by_depth"].items():
                losses[f"mtp{depth}"] = loss
                total = total + self.mtp_lambda * self.mtp_decay ** (depth - 2) * loss
            losses["trajectory"] = ordered["item_loss"]
            if self.training and self.mu_cond_drop and self.ubt_null_aux_lambda:
                auxiliary = self.teacher_force_trajectory_loss(h_last, targets, start_depth=2,
                    deterministic=False, condition_drop=self.mu_cond_drop)
                losses["null_aux"] = auxiliary["item_loss"]
                total = total + self.ubt_null_aux_lambda * auxiliary["item_loss"]
        out.final_states = None
        out.loss, out.losses = total, losses
        return out

    @torch.no_grad()
    def generate(self, batch, n_return_sequences=20, **kwargs):
        scores = self.main_logits(self.encode_history(batch, deterministic=True))
        return scores.topk(min(n_return_sequences, self.n_items), dim=-1).indices + 1, None

    def depth_head_param_counts(self):
        return {"depth_heads": 1, "shared_with_first": True,
                "main_readout": sum(p.numel() for m in
                    (self.main_blocks, self.main_ln, self.main_mlp) for p in m.parameters()),
                "transition": sum(p.numel() for n, p in self.named_parameters() if n.startswith("mu_"))}
