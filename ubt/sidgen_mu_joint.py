"""One future readout with differentiable catalogue-mean conditioning.

The main scorer retains the SIDMTP input/trunk/readout.  Every item objective
is full-catalogue CE.  Future positions share one residual transition and one
readout, and receive the previous *predicted* distribution's weighted SID mean.
Targets are read only after prediction.  The public transition API also accepts
a sampled item's embedding for matched policy-gradient and condition-response
experiments; neither mode introduces additional depth-specific heads.
"""
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from sidgen_mtp import N_CODEBOOK, CODEBOOK_SIZE, _readout_forward
from sidgen_mtp_aux import SIDMTP1


class SIDMuJoint(SIDMTP1):
    def __init__(self, *args, feedback_temperature=1.0, detach_feedback=False,
                 feedback_mode="soft", token_ce_lambda=1.0, ce_chunk_size=256,
                 adapter_width=128, mu_cond_drop=0.0, clone_future=True,
                 soft_fraction=0.75, action_temperature=0.15,
                 **kwargs):
        kwargs.setdefault("mtp_detach", 0)
        kwargs["mtp1_mode"] = "traj"
        super().__init__(*args, **kwargs)
        if feedback_mode not in ("soft", "hard", "mixed", "innovation"):
            raise ValueError("feedback_mode must be soft, hard, mixed or innovation")
        if feedback_temperature <= 0 or ce_chunk_size <= 0:
            raise ValueError("positive temperature and CE chunk size required")
        if not 0 <= mu_cond_drop < 1:
            raise ValueError("mu_cond_drop must be in [0, 1)")
        if not 0 <= soft_fraction <= 1:
            raise ValueError("soft_fraction must be in [0, 1]")
        if action_temperature <= 0:
            raise ValueError("action_temperature must be positive")
        self.feedback_temperature = float(feedback_temperature)
        self.detach_feedback = bool(detach_feedback)
        self.feedback_mode = feedback_mode
        self.soft_fraction = float(soft_fraction)
        self.action_temperature = float(action_temperature)
        self.token_ce_lambda = float(token_ce_lambda)
        self.ce_chunk_size = int(ce_chunk_size)
        self.mu_cond_drop = float(mu_cond_drop)
        dim = self.gpt2.config.n_embd
        # Remove inherited conditioning modules: this transition is the only
        # conditioning path and each future depth uses precisely this one stack.
        del self.cond_proj, self.cond_null, self.mtp_slot
        self.mu_ln_h = nn.LayerNorm(dim)
        self.mu_ln_mu = nn.LayerNorm(dim)
        self.mu_horizon = nn.Embedding(self.mtp_depths - 1, 16)
        self.mu_in = nn.Linear(2 * dim + 16, adapter_width)
        self.mu_out = nn.Linear(adapter_width, dim)
        self.mu_null = nn.Parameter(torch.zeros(dim))
        nn.init.zeros_(self.mu_out.weight)
        nn.init.zeros_(self.mu_out.bias)
        nn.init.normal_(self.mu_horizon.weight, std=0.02)
        if clone_future:
            self.mtp1_blocks = deepcopy(self.main_blocks)
            self.mtp1_ln = deepcopy(self.main_ln)
            self.mtp1_mlp = deepcopy(self.main_mlp)

    def _history_outputs(self, batch):
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[1] > self.gpt2.config.n_positions:
            raise ValueError("history must be a padded sequence within the trunk budget")
        lens = batch["seq_lens"]
        if bool(((lens < 1) | (lens > ids.shape[1])).any()):
            raise ValueError("each history requires at least one real item")
        out = self.gpt2(inputs_embeds=self._input_embs(ids),
                        attention_mask=batch["attention_mask"])
        out.last_hidden_for_agg = out.last_hidden_state
        return out

    def encode_history(self, batch):
        """Encode history only; labels and set_targets are never inspected."""
        out = self._history_outputs(batch)
        return self._h_last(out, batch)

    def main_logits(self, hidden):
        return self.score_catalog(self.main_states(hidden.unsqueeze(1)).squeeze(1))

    def score_catalog(self, states):
        """Full real-item catalogue, FP32 mathematics; column j is item j+1."""
        with torch.autocast(device_type=states.device.type, enabled=False):
            states = F.normalize(states.float(), dim=-1)
            tables = self._digit_embeddings()
            codes = self.item_dcodes[1:]
            scores = states.new_zeros((len(states), self.n_items))
            for digit in range(N_CODEBOOK):
                logits = states[:, digit] @ tables[digit].float().T
                logp = F.log_softmax(logits / self.temperature, dim=-1)
                scores = scores + logp[:, codes[:, digit]]
            return scores / N_CODEBOOK

    def item_feedback_table(self):
        """Raw SID-token means, as in the archived soft rollout, without PAD.

        Summing digit-wise avoids materializing N x 16 x D token embeddings.
        The table stays differentiable; OpenAI text enters history encoding.
        """
        tokens = self.item_id2tokens[1:]
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            weight = self.gpt2.wte.weight.float()
            table = weight.index_select(0, tokens[:, 0])
            for digit in range(1, N_CODEBOOK):
                table = table + weight.index_select(0, tokens[:, digit])
            return table / N_CODEBOOK

    def future_step(self, hidden, mu, depth):
        """One shared transition; depth is the actual prediction position 2..K."""
        if not 2 <= int(depth) <= self.mtp_depths:
            raise ValueError("depth must lie in [2, mtp_depths]")
        if mu.shape != hidden.shape:
            raise ValueError("mu and hidden must both have shape [batch, hidden_dim]")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            horizon = self.mu_horizon.weight[int(depth) - 2].expand(len(hidden), -1)
            joined = torch.cat((self.mu_ln_h(hidden.float()),
                                self.mu_ln_mu(mu.float()), horizon.float()), -1)
            residual = self.mu_out(F.gelu(self.mu_in(joined)))
            next_hidden = hidden.float() + residual
        states = _readout_forward(self.mtp1_blocks, self.mtp1_ln,
                                  self.mtp1_mlp, next_hidden.unsqueeze(1)).squeeze(1)
        return next_hidden, self.score_catalog(states)

    def _rollout_from_hidden(self, hidden, K, return_trace=True,
                             detach_feedback=None, feedback_mode=None,
                             soft_fraction=None, action_temperature=None):
        if not 1 <= int(K) <= self.mtp_depths:
            raise ValueError("rollout length must lie in [1, mtp_depths]")
        detached = self.detach_feedback if detach_feedback is None else bool(detach_feedback)
        mode = self.feedback_mode if feedback_mode is None else feedback_mode
        if mode not in ("soft", "hard", "mixed", "innovation"):
            raise ValueError("unknown feedback mode")
        fraction = self.soft_fraction if soft_fraction is None else float(soft_fraction)
        if not 0 <= fraction <= 1:
            raise ValueError("soft_fraction must be in [0, 1]")
        action_temp = (self.action_temperature if action_temperature is None
                       else float(action_temperature))
        if action_temp <= 0:
            raise ValueError("action_temperature must be positive")
        if self.mtp_detach:
            hidden = hidden.detach()
        table = self.item_feedback_table()
        if self.mtp_detach:
            table = table.detach()
        logits, hidden_trace, mus, probabilities = [], [hidden], [], []
        scores = self.main_logits(hidden)
        logits.append(scores)
        # The mask selects the hard branch's action only.  The soft mean always
        # uses the unmasked catalogue, matching analytic_joint_policy_loss.
        used = (torch.zeros_like(scores, dtype=torch.bool)
                if mode in ("hard", "mixed", "innovation") else None)
        chosen_trace = []
        action_probabilities = []
        for depth in range(2, int(K) + 1):
            p = F.softmax(scores.float() / self.feedback_temperature, dim=-1)
            probabilities.append(p)
            feedback_p = p.detach() if detached else p
            if mode == "soft":
                mu = feedback_p @ table
            else:
                chosen = scores.masked_fill(used, -torch.inf).argmax(-1)
                chosen_trace.append(chosen)
                hard_mu = table.index_select(0, chosen)
                if mode == "innovation":
                    action_p = F.softmax(
                        scores.float().masked_fill(used.clone(), -torch.inf) / action_temp, -1)
                    action_probabilities.append(action_p)
                    feedback_action_p = action_p.detach() if detached else action_p
                    # The policy's masked/action-temperature mean differs from
                    # the raw/feedback-temperature mean.  Centering on the
                    # former makes the sampled residual exactly zero-mean:
                    # E_{a~pi}[mu_base + rho*(E[a]-E_pi)] = mu_base.
                    mu_base = feedback_p @ table
                    mu_action = feedback_action_p @ table
                    mu = mu_base + (1.0-fraction) * (hard_mu - mu_action)
                else:
                    mu = (hard_mu if mode == "hard" else
                          fraction * (feedback_p @ table) + (1.0-fraction) * hard_mu)
                used.scatter_(1, chosen[:, None], True)
            if self.training and self.mu_cond_drop:
                drop = torch.rand(len(mu), device=mu.device) < self.mu_cond_drop
                mu = torch.where(drop[:, None], self.mu_null[None, :], mu)
            hidden, scores = self.future_step(hidden, mu, depth)
            mus.append(mu)
            hidden_trace.append(hidden)
            logits.append(scores)
        probabilities.append(F.softmax(scores.float() / self.feedback_temperature, -1))
        result = {"logits": torch.stack(logits, dim=1)}
        if return_trace:
            result.update(hidden=hidden_trace, mu=mus, probabilities=probabilities,
                          feedback_table=table)
            if used is not None:
                chosen_trace.append(scores.masked_fill(used, -torch.inf).argmax(-1))
                result["actions"] = torch.stack(chosen_trace, dim=1)
            if mode == "innovation":
                action_probabilities.append(F.softmax(
                    scores.float().masked_fill(used, -torch.inf) / action_temp, -1))
                result["action_probabilities"] = action_probabilities
        return result

    def soft_rollout(self, batch, K=None, return_trace=True, detach_feedback=None,
                     feedback_mode=None, soft_fraction=None, action_temperature=None):
        """Teacher-free prediction; K defaults to the configured training horizon."""
        return self._rollout_from_hidden(
            self.encode_history(batch), self.mtp_depths if K is None else K,
            return_trace, detach_feedback, feedback_mode, soft_fraction, action_temperature)

    def _main_supervised_losses(self, hidden, labels):
        if labels.ndim != 2 or labels.shape != hidden.shape[:2]:
            raise ValueError("training labels must match padded history positions")
        mask = labels > 0
        if bool(((labels != -100) & ~mask).any()):
            raise ValueError("labels must be one-based item IDs or -100")
        selected = hidden[mask]
        targets = labels[mask]
        zero = hidden.sum() * 0.0
        if len(targets) == 0:
            return zero, zero
        if bool((targets > self.n_items).any()):
            raise ValueError("target item outside catalogue")
        loss_item, loss_token = zero, zero
        for start in range(0, len(targets), self.ce_chunk_size):
            stop = start + self.ce_chunk_size
            yy = targets[start:stop]
            states = self.main_states(selected[start:stop].unsqueeze(1)).squeeze(1)
            scores = self.score_catalog(states)
            loss_item = loss_item + F.cross_entropy(
                scores / self.item_ce_tau, yy - 1, reduction="sum",
                label_smoothing=self.item_ce_ls)
            if self.token_ce_lambda:
                states = F.normalize(states.float(), dim=-1)
                digit_losses = []
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    tables = self._digit_embeddings()
                    for digit in range(N_CODEBOOK):
                        token_logits = states[:, digit] @ tables[digit].float().T
                        digit_losses.append(F.cross_entropy(
                            token_logits / self.temperature,
                            self.item_dcodes[yy, digit], reduction="sum"))
                loss_token = loss_token + torch.stack(digit_losses).mean()
        return loss_token / len(targets), loss_item / len(targets)

    def forward(self, batch, return_loss=True):
        out = self._history_outputs(batch)
        hidden = out.last_hidden_for_agg
        if not return_loss:
            # Compatibility with legacy representation diagnostics; ordinary
            # generation uses encode_history and avoids this full-position stack.
            out.final_states = self.main_states(hidden)
            return out
        token, item = self._main_supervised_losses(hidden, batch["labels"])
        losses = {"token": token, "item": item,
                  "align": (self._align_loss(batch) if self.use_text and self.align_lambda > 0
                            else None)}
        st = batch.get("set_targets")
        if st is not None:
            if st.ndim != 2 or st.shape[0] != len(hidden):
                raise ValueError("set_targets must have shape [batch, horizon]")
            length = min(st.shape[1], self.mtp_depths)
            trace = self._rollout_from_hidden(self._h_last(out, batch), length,
                                              return_trace=False)
            for depth in range(2, length + 1):
                yy = st[:, depth - 1]
                valid = yy > 0
                if bool((yy[valid] > self.n_items).any()):
                    raise ValueError("future target outside catalogue")
                losses[f"mtp{depth}"] = (F.cross_entropy(
                    trace["logits"][valid, depth - 1] / self.mtp_tau,
                    yy[valid] - 1) if bool(valid.any()) else None)
        total = self.token_ce_lambda * token + self.item_ce_lambda * item
        if losses["align"] is not None:
            total = total + self.align_lambda * losses["align"]
        for depth in range(2, self.mtp_depths + 1):
            loss = losses.get(f"mtp{depth}")
            if loss is not None:
                total = total + self.mtp_lambda * self.mtp_decay ** (depth - 2) * loss
        out.final_states = None  # Full-position states are unnecessary during training.
        out.loss, out.losses = total, losses
        return out

    def _depth_scores(self, k, h_last, cond_ids, null_mask=None):
        table = self.item_feedback_table()
        if null_mask is None and bool(((cond_ids < 1) | (cond_ids > self.n_items)).any()):
            raise ValueError("hard feedback IDs must be one-based real item IDs")
        mu = (self.mu_null[None, :].expand_as(h_last) if null_mask is not None
              else table.index_select(0, cond_ids.long() - 1))
        return self.future_step(h_last, mu, k)[1]

    @torch.no_grad()
    def generate(self, batch, n_return_sequences=20, **kwargs):
        scores = self.main_logits(self.encode_history(batch))
        return scores.topk(min(n_return_sequences, self.n_items), dim=-1).indices + 1, None

    @torch.no_grad()
    def decode_set(self, batch, K, mode="soft", beam=4, soft_fraction=None,
                   action_temperature=None):
        if not 1 <= K <= self.n_items or (K > self.mtp_depths and mode != "main"):
            raise ValueError("decode K outside future horizon/catalogue")
        hidden = self.encode_history(batch)
        first_scores = self.main_logits(hidden)
        if mode == "main":
            top = first_scores.topk(K, dim=-1)
            return top.indices + 1, top.values.mean(-1)
        if mode == "union":
            mode = "soft_mix"
        if mode in ("soft", "soft_mix"):
            all_scores = self._rollout_from_hidden(
                hidden, K, return_trace=False, feedback_mode="soft")["logits"]
            if mode == "soft_mix":
                mix = torch.logsumexp(F.log_softmax(all_scores, -1), dim=1)
                top = mix.topk(K, dim=-1)
                return top.indices + 1, top.values.mean(-1)
            used = torch.zeros_like(first_scores, dtype=torch.bool)
            chosen, values = [], []
            for depth in range(K):
                scores = all_scores[:, depth].masked_fill(used, -torch.inf)
                top = scores.max(-1)
                chosen.append(top.indices)
                values.append(top.values)
                used.scatter_(1, top.indices[:, None], True)
            return torch.stack(chosen, 1) + 1, torch.stack(values, 1).mean(1)
        if mode not in ("spec", "hard", "mixed", "innovation"):
            raise ValueError("supported modes: main, soft, soft_mix/union, spec/hard, mixed, innovation")
        fraction = self.soft_fraction if soft_fraction is None else float(soft_fraction)
        if not 0 <= fraction <= 1:
            raise ValueError("soft_fraction must be in [0, 1]")
        action_temp = (self.action_temperature if action_temperature is None
                       else float(action_temperature))
        if action_temp <= 0:
            raise ValueError("action_temperature must be positive")
        used = torch.zeros_like(first_scores, dtype=torch.bool)
        table = self.item_feedback_table()
        scores, chosen, values = first_scores, [], []
        for depth in range(1, K + 1):
            action_scores = scores.masked_fill(used, -torch.inf)
            top = action_scores.max(-1)
            chosen.append(top.indices)
            values.append(top.values)
            used.scatter_(1, top.indices[:, None], True)
            if depth < K:
                hard_mu = table[top.indices]
                mu = hard_mu
                if mode == "mixed":
                    probability = F.softmax(scores.float() / self.feedback_temperature, -1)
                    mu = fraction * (probability @ table) + (1.0-fraction) * hard_mu
                elif mode == "innovation":
                    probability = F.softmax(scores.float() / self.feedback_temperature, -1)
                    action_probability = F.softmax(action_scores.float() / action_temp, -1)
                    mu = probability @ table + (1.0-fraction) * (
                        hard_mu - action_probability @ table)
                hidden, scores = self.future_step(hidden, mu, depth + 1)
        return torch.stack(chosen, 1) + 1, torch.stack(values, 1).mean(1)

    def depth_head_param_counts(self):
        names = ("mtp1_blocks", "mtp1_ln", "mtp1_mlp", "mu_ln_h", "mu_ln_mu",
                 "mu_horizon", "mu_in", "mu_out")
        return {"depth_heads": 1,
                "shared_stack_and_transition": sum(
                    p.numel() for name in names for p in getattr(self, name).parameters()),
                "mu_null": self.mu_null.numel()}
