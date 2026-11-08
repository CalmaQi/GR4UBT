"""SIDMTP: strictly MTP-auxiliary generative Semantic-ID recommender.

DeepSeek-V3-style Multi-Token Prediction transplanted to next-item recommendation:
  - ONE shared GPT-2 trunk over the user's SID history (identical input recipe
    to RecSet-Base: 16 digit embeddings + text-conditioned attention pooling).
  - Main head (depth 1): the RecSet-Base readout (16 ResBlocks + LN + MLP),
    trained with token CE + item CE + align loss.  Task A inference unchanged
    (one-pass full-catalog composed scoring).
  - MTP heads (depth k = 2..K): each is its own readout stack, conditioned on
    the GROUND-TRUTH item at lag k-1 (teacher-forced chain, exactly DeepSeek's
    module k seeing the true embedding of position i+k-1).  Trained with a
    SHARP single-positive InfoNCE on the lag-k item -- no multi-positive
    logsumexp anywhere, so the P1 entropy floor cannot arise by construction.
  - Conditioning dropout (learned null embedding) makes every depth head a
    valid standalone scorer, unlocking parallel "union" decoding and damping
    exposure bias (training covers both draft-conditioned and free modes).

Inference (Task B, three modes):
  spec      : greedy speculative set decode (draft b1 from main head, then each
              depth-k head scores the FULL catalog conditioned on the previous
              pick -- conditioning is readout-level, the trunk runs once).
  spec_beam : beam over top-B drafts with chain-score verification.
  union     : parallel direct set generation (free depth scores, max-aggregated).

Everything is ONE model trained end-to-end from scratch.  No fusion, no
second model, no ensemble.  NEW CODE (mtp track); existing files untouched.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

N_CODEBOOK = 16
CODEBOOK_SIZE = 256


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


def _make_readout(n_embd):
    """Per-digit query stack: 16 ResBlocks + LayerNorm + MLP refine (RecSet-Base shape)."""
    blocks = nn.ModuleList([ResBlock(n_embd) for _ in range(N_CODEBOOK)])
    ln = nn.LayerNorm(n_embd)
    mlp = nn.Sequential(
        nn.Linear(n_embd * N_CODEBOOK, n_embd * 2),
        nn.GELU(),
        nn.Linear(n_embd * 2, n_embd * N_CODEBOOK),
    )
    return blocks, ln, mlp


def _readout_forward(blocks, ln, mlp, h):
    fs = torch.stack([blocks[i](h) for i in range(N_CODEBOOK)], dim=-2)
    B, L, H, D = fs.shape
    fs = fs + ln(mlp(fs.reshape(B * L, -1)).reshape(B, L, H, D))
    return fs


class SIDMTP(nn.Module):
    def __init__(self, n_items, item_codes, item_dcodes, text_embs=None,
                 n_embd=448, n_layer=2, n_head=4, n_inner=1024,
                 resid_pdrop=0.0, embd_pdrop=0.5, attn_pdrop=0.5,
                 layer_norm_epsilon=1e-12, initializer_range=0.02,
                 temperature=0.03,
                 item_ce_lambda=0.5, item_ce_neg=127, item_ce_tau=1.0,
                 item_ce_max_pos=4096, item_ce_inbatch=0, item_ce_ls=0.0,
                 align_lambda=0.1, align_temp=0.07,
                 mtp_depths=3, mtp_lambda=0.25, mtp_decay=1.0,
                 mtp_neg=255, mtp_tau=1.0, mtp_max_pos=4096, mtp_inbatch=0,
                 mtp_cond="single", mtp_cond_drop=0.25, mtp_detach=1,
                 input_pooling="attn", text_encoder="mlp", max_history=20):
        super().__init__()
        self.n_items = n_items
        self.register_buffer("item_id2tokens", item_codes.clone(), persistent=False)
        self.register_buffer("item_dcodes", item_dcodes.clone(), persistent=False)
        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        gpt2config = GPT2Config(
            vocab_size=N_CODEBOOK * CODEBOOK_SIZE + 2, n_positions=int(max_history),
            n_embd=n_embd, n_layer=n_layer, n_head=n_head, n_inner=n_inner,
            activation_function="gelu_new",
            resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon, initializer_range=initializer_range,
            eos_token_id=N_CODEBOOK * CODEBOOK_SIZE + 1,
        )
        self.gpt2 = GPT2Model(gpt2config)
        self.n_pred_head = N_CODEBOOK

        # ---- main head (depth 1): RecSet-Base readout ----
        self.main_blocks, self.main_ln, self.main_mlp = _make_readout(n_embd)

        # ---- MTP heads (depths 2..K) ----
        self.mtp_depths = int(mtp_depths)
        assert self.mtp_depths >= 2, "MTP track needs at least 2 depths"
        self.mtp_lambda = float(mtp_lambda)
        self.mtp_decay = float(mtp_decay)
        self.mtp_neg = int(mtp_neg)
        self.mtp_tau = float(mtp_tau)
        self.mtp_max_pos = int(mtp_max_pos)
        self.mtp_inbatch = int(mtp_inbatch)
        self.mtp_cond = mtp_cond
        self.mtp_cond_drop = float(mtp_cond_drop)
        # Detach the trunk for the MTP path: MTP heads train ONLY their own
        # readout stacks on frozen context features.  Rationale (empirical,
        # this track): MTP->trunk gradients consistently interfere with the
        # item/ranking objective (see EXPERIMENT_LOG 36.x); v2's stage-2
        # "frozen features + light readout" is the per-depth generalization.
        self.mtp_detach = bool(mtp_detach)
        self.mtp_blocks = nn.ModuleList()
        self.mtp_ln = nn.ModuleList()
        self.mtp_mlp = nn.ModuleList()
        self.cond_proj = nn.ModuleList()
        self.cond_null = nn.ParameterList()
        for _ in range(self.mtp_depths - 1):
            blocks, ln, mlp = _make_readout(n_embd)
            self.mtp_blocks.append(blocks)
            self.mtp_ln.append(ln)
            self.mtp_mlp.append(mlp)
            proj = nn.Linear(n_embd, n_embd)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.cond_proj.append(proj)
            self.cond_null.append(nn.Parameter(torch.zeros(n_embd)))

        # ---- text grounding (same recipe as RecSet-Base) ----
        self.use_text = text_embs is not None
        self.text_dim = int(text_embs.shape[1]) if self.use_text else 0
        self.text_encoder = text_encoder
        if self.use_text:
            self.register_buffer("text_embs", text_embs, persistent=False)
            if text_encoder == "deep":
                self.text_proj = nn.Sequential(
                    nn.Linear(self.text_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(),
                    nn.Linear(1024, n_embd))
                self.text_skip = nn.Linear(self.text_dim, n_embd)
                nn.init.zeros_(self.text_skip.weight)
                nn.init.zeros_(self.text_skip.bias)
            else:
                self.text_proj = nn.Sequential(
                    nn.Linear(self.text_dim, n_embd * 2), nn.GELU(),
                    nn.Linear(n_embd * 2, n_embd))
        self.input_pooling = input_pooling
        if input_pooling == "attn" and self.use_text:
            self.pool_query = nn.Linear(n_embd, n_embd)
            self.pool_key = nn.Linear(n_embd, n_embd)
            self.pool_scale = float(1.0 / (n_embd ** 0.5))

        self.item_ce_lambda = float(item_ce_lambda)
        self.item_ce_neg = int(item_ce_neg)
        self.item_ce_tau = float(item_ce_tau)
        self.item_ce_max_pos = int(item_ce_max_pos)
        self.item_ce_inbatch = int(item_ce_inbatch)
        self.item_ce_ls = float(item_ce_ls)
        self.align_lambda = float(align_lambda)
        self.align_temp = float(align_temp)

    # ------------------------------------------------------------ input (RecSet-Base recipe)
    def _text_proj_emb(self, ids):
        t = self.text_embs[ids]
        out = self.text_proj(t)
        if self.text_encoder == "deep":
            out = out + self.text_skip(t)
        return out

    def _input_embs(self, ids):
        tokens = self.item_id2tokens[ids]
        digit_embs = self.gpt2.wte(tokens)
        if not self.use_text:
            return digit_embs.mean(dim=-2)
        tproj = self._text_proj_emb(ids)
        if self.input_pooling == "mean":
            return digit_embs.mean(dim=-2) + tproj
        q = self.pool_query(tproj).unsqueeze(2)
        k = self.pool_key(digit_embs)
        att = (q * k).sum(-1) * self.pool_scale
        w = att.softmax(dim=-1)
        return (w.unsqueeze(-1) * digit_embs).sum(dim=2) + tproj

    def _digit_embeddings(self):
        w = F.normalize(self.gpt2.wte.weight[1:-1], dim=-1)
        return torch.chunk(w, self.n_pred_head, dim=0)

    def _item_rep(self, ids):
        """Standalone item representation for MTP conditioning: same pipeline
        as trunk inputs.  ids: (B,) -> (B, D)."""
        return self._input_embs(ids.unsqueeze(1)).squeeze(1)

    def _sample_negs(self, n):
        return torch.randint(1, self.n_items + 1, (n,),
                             device=self.item_dcodes.device)

    def _compose_scores(self, states, codes):
        """states: (P, 16, D) normalized; codes: (P, ..., 16).  -> composed log-probs."""
        token_embs = self._digit_embeddings()
        lsm = [F.log_softmax(torch.matmul(states[:, i, :], token_embs[i].T)
                             / self.temperature, dim=-1) for i in range(N_CODEBOOK)]
        P = codes.shape[0]
        rows = torch.arange(P, device=codes.device)
        tail = codes.shape[1:-1]
        if len(tail) > 0:
            rows = rows.view(-1, *([1] * len(tail))).expand(P, *tail)
        s = 0
        for d in range(N_CODEBOOK):
            s = s + lsm[d][rows, codes[..., d]]
        return s / N_CODEBOOK

    # ------------------------------------------------------------ heads
    def main_states(self, h):
        return _readout_forward(self.main_blocks, self.main_ln, self.main_mlp, h)

    def mtp_states(self, k, h_last, cond_emb):
        """k = 0..(depths-2) indexes depth k+2.  h_last: (B, D); cond_emb: (B, D)."""
        idx = min(k, len(self.mtp_blocks) - 1)
        x = h_last + self.cond_proj[idx](cond_emb)
        x = x.unsqueeze(1)                                   # (B, 1, D)
        fs = _readout_forward(self.mtp_blocks[idx], self.mtp_ln[idx],
                              self.mtp_mlp[idx], x)          # (B, 1, 16, D)
        return fs.squeeze(1)

    # ------------------------------------------------------------ forward
    def forward(self, batch, return_loss=True):
        input_embs = self._input_embs(batch["input_ids"])
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=batch["attention_mask"])
        h = outputs.last_hidden_state
        outputs.final_states = self.main_states(h)
        outputs.last_hidden_for_agg = h
        if not return_loss:
            return outputs

        B, L, D = h.shape
        losses = {}

        # ---- token CE + item CE on the main head (RecSet-Base, unchanged) ----
        labels = batch["labels"]
        label_mask = labels.view(-1) != -100
        flat_labels = labels.view(-1)[label_mask]
        sel = F.normalize(outputs.final_states.view(-1, N_CODEBOOK, D)[label_mask], dim=-1)
        token_embs = self._digit_embeddings()
        digit_logits = [torch.matmul(sel[:, i, :], token_embs[i].T) / self.temperature
                        for i in range(N_CODEBOOK)]
        token_labels = self.item_id2tokens[flat_labels]
        losses["token"] = torch.mean(torch.stack([
            self.loss_fct(digit_logits[i], token_labels[:, i] - i * CODEBOOK_SIZE - 1)
            for i in range(N_CODEBOOK)]))
        losses["item"] = self._item_ce(digit_logits, flat_labels,
                                       batch.get("all_label_ids", flat_labels))

        if self.use_text and self.align_lambda > 0:
            losses["align"] = self._align_loss(batch)
        else:
            losses["align"] = None

        # ---- MTP losses (teacher-forced chain, sharp InfoNCE per depth) ----
        set_t = batch.get("set_targets")
        mtp_losses = []
        if set_t is not None:
            lh_idx = (batch["seq_lens"] - 1).view(-1, 1, 1).expand(-1, 1, D)
            h_last = h.gather(dim=1, index=lh_idx).squeeze(1)         # (B, D)
            hl = h_last.detach() if self.mtp_detach else h_last
            for k in range(2, self.mtp_depths + 1):
                tgt = set_t[:, k - 1]
                valid = tgt > 0
                if not valid.any():
                    mtp_losses.append(None)
                    continue
                if self.mtp_cond == "chain":
                    cond_ids = set_t[:, : k - 1].clamp(min=0)
                    cond_emb = self._input_embs(cond_ids).mean(dim=1)
                    zero_rows = ~valid
                    cond_emb = cond_emb.masked_fill(zero_rows.unsqueeze(-1), 0.0)
                else:                                                  # "single" (DeepSeek exact)
                    cond_ids = set_t[:, k - 2].clamp(min=0)
                    cond_emb = self._item_rep(cond_ids)
                if self.mtp_detach:
                    cond_emb = cond_emb.detach()
                if self.mtp_cond_drop > 0 and self.training:
                    drop = torch.rand(B, device=h.device) < self.mtp_cond_drop
                    idx_k = min(k - 2, len(self.cond_null) - 1)
                    cond_emb = torch.where(drop.unsqueeze(-1),
                                           self.cond_null[idx_k].unsqueeze(0).expand_as(cond_emb),
                                           cond_emb)
                states = F.normalize(
                    self.mtp_states(k - 2, hl, cond_emb), dim=-1)      # (B, 16, D)
                states = states[valid]
                tgt_v = tgt[valid]
                losses[f"mtp{k}"] = self._mtp_ce(states, tgt_v, all_ids=tgt_v)
                mtp_losses.append(losses[f"mtp{k}"])
        outputs.losses = losses
        outputs.loss = self._combine(losses)
        return outputs

    def _combine(self, losses):
        total = losses["token"]
        if losses.get("item") is not None:
            total = total + self.item_ce_lambda * losses["item"]
        if losses.get("align") is not None:
            total = total + self.align_lambda * losses["align"]
        for k in range(2, self.mtp_depths + 1):
            v = losses.get(f"mtp{k}")
            if v is not None:
                total = total + self.mtp_lambda * (self.mtp_decay ** (k - 2)) * v
        return total

    # ------------------------------------------------------------- losses
    def _item_ce(self, digit_logits, label_ids, all_label_ids):
        """InfoNCE on the composed score (uniform negatives + optional in-batch)."""
        P = label_ids.shape[0]
        pos_codes = self.item_dcodes[label_ids]
        neg_codes = [self.item_dcodes[self._sample_negs(P * self.item_ce_neg)]
                     .view(P, self.item_ce_neg, N_CODEBOOK)]
        if self.item_ce_inbatch > 0 and all_label_ids is not None and P > 1:
            pool = all_label_ids
            k = min(self.item_ce_inbatch, pool.shape[0] - 1)
            rand = torch.randint(0, pool.shape[0], (P, k), device=pool.device)
            ar = torch.arange(P, device=pool.device).unsqueeze(1)
            rand = rand + (rand == ar).long()
            neg_codes.append(self.item_dcodes[pool[rand.clamp(max=pool.shape[0] - 1)]])
        neg_codes = torch.cat(neg_codes, dim=1)
        token_embs = self._digit_embeddings()
        lsm = [F.log_softmax(digit_logits[i], dim=-1) for i in range(N_CODEBOOK)]
        rows = torch.arange(P, device=label_ids.device).view(-1, 1)
        s_neg = sum(lsm[d][rows.expand(P, neg_codes.shape[1]), neg_codes[..., d]]
                    for d in range(N_CODEBOOK)) / N_CODEBOOK
        s_pos = sum(lsm[d][rows.squeeze(1), pos_codes[:, d]]
                    for d in range(N_CODEBOOK)) / N_CODEBOOK
        logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1) / self.item_ce_tau
        target = torch.zeros(P, dtype=torch.long, device=label_ids.device)
        return F.cross_entropy(logits, target, label_smoothing=self.item_ce_ls)

    def _mtp_ce(self, states, target_ids, all_ids=None):
        """Sharp single-positive InfoNCE at one MTP depth (same form as item CE)."""
        if states.shape[0] > self.mtp_max_pos:
            idx = torch.randperm(states.shape[0], device=states.device)[: self.mtp_max_pos]
            states = states[idx]
            target_ids = target_ids[idx]
            all_ids = all_ids[idx] if all_ids is not None else None
        P = target_ids.shape[0]
        pos_codes = self.item_dcodes[target_ids]
        neg_codes = [self.item_dcodes[self._sample_negs(P * self.mtp_neg)]
                     .view(P, self.mtp_neg, N_CODEBOOK)]
        if self.mtp_inbatch > 0 and all_ids is not None and P > 1:
            pool = all_ids
            k = min(self.mtp_inbatch, pool.shape[0] - 1)
            rand = torch.randint(0, pool.shape[0], (P, k), device=pool.device)
            ar = torch.arange(P, device=pool.device).unsqueeze(1)
            rand = rand + (rand == ar).long()
            neg_codes.append(self.item_dcodes[pool[rand.clamp(max=pool.shape[0] - 1)]])
        neg_codes = torch.cat(neg_codes, dim=1)
        token_embs = [t.detach() for t in self._digit_embeddings()]  # full isolation
        lsm = [F.log_softmax(torch.matmul(states[:, i, :], token_embs[i].T)
                             / self.temperature, dim=-1) for i in range(N_CODEBOOK)]
        rows = torch.arange(P, device=states.device).view(-1, 1)
        s_neg = sum(lsm[d][rows.expand(P, neg_codes.shape[1]), neg_codes[..., d]]
                    for d in range(N_CODEBOOK)) / N_CODEBOOK
        s_pos = sum(lsm[d][rows.squeeze(1), pos_codes[:, d]]
                    for d in range(N_CODEBOOK)) / N_CODEBOOK
        logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1) / self.mtp_tau
        target = torch.zeros(P, dtype=torch.long, device=states.device)
        return F.cross_entropy(logits, target)

    def _align_loss(self, batch, max_pos=2048):
        ids = batch["input_ids"]
        tokens = self.item_id2tokens[ids]
        digit_mean = self.gpt2.wte(tokens).mean(dim=-2)
        mask = batch["attention_mask"].bool()
        tproj = self._text_proj_emb(ids)
        dm = digit_mean[mask]
        tp = tproj[mask]
        if dm.shape[0] > max_pos:
            idx = torch.randperm(dm.shape[0], device=dm.device)[:max_pos]
            dm, tp = dm[idx], tp[idx]
        d = F.normalize(dm, dim=-1)
        t = F.normalize(tp, dim=-1)
        logits = d @ t.T / self.align_temp
        target = torch.arange(d.shape[0], device=d.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    # ------------------------------------------------------------ scoring / decode
    def score_catalog(self, states):
        """states: (B, 16, D) (will be normalized here).  -> composed scores (B, N)."""
        states = F.normalize(states, dim=-1)
        token_embs = self._digit_embeddings()
        lsm = [F.log_softmax(torch.matmul(states[:, i, :], token_embs[i].T)
                             / self.temperature, dim=-1) for i in range(N_CODEBOOK)]
        codes_t = self.item_dcodes[1:].t()                            # (16, N)
        out = 0
        for d in range(N_CODEBOOK):
            out = out + lsm[d][:, codes_t[d]]
        return out / N_CODEBOOK

    @torch.no_grad()
    def generate(self, batch, n_return_sequences=20, **kw):
        """Task A: full-catalog composed scoring from the main head (RecSet-Base path)."""
        outputs = self.forward(batch, return_loss=False)
        B, L, D = outputs.last_hidden_for_agg.shape
        lh_idx = (batch["seq_lens"] - 1).view(-1, 1, 1).expand(-1, 1, D)
        h_last = outputs.last_hidden_for_agg.gather(dim=1, index=lh_idx).squeeze(1)
        s = self.score_catalog(self.main_states(h_last.unsqueeze(1)).squeeze(1))
        top = torch.topk(s, k=n_return_sequences, dim=-1).indices + 1
        return top, None

    def _h_last(self, outputs, batch):
        D = outputs.last_hidden_for_agg.shape[-1]
        lh_idx = (batch["seq_lens"] - 1).view(-1, 1, 1).expand(-1, 1, D)
        return outputs.last_hidden_for_agg.gather(dim=1, index=lh_idx).squeeze(1)

    def _depth_scores(self, k, h_last, cond_ids, null_mask=None):
        """Full-catalog scores from MTP depth k (k = actual depth, >=2), conditioned
        on cond_ids (B,) or null where null_mask.  -> (B, N) scores."""
        idx_k = min(k - 2, len(self.cond_proj) - 1)
        if null_mask is None:
            cond_emb = self._item_rep(cond_ids)
        else:
            cond_emb = self.cond_null[idx_k].unsqueeze(0).expand(h_last.shape[0], -1)
        states = self.mtp_states(k - 2, h_last, cond_emb)
        s = self.score_catalog(states)
        return s

    @torch.no_grad()
    def decode_set(self, batch, K, mode="spec", beam=4):
        """Task B set decoding.  Returns (ids LongTensor (B, K), scores (B,))."""
        outputs = self.forward(batch, return_loss=False)
        h_last = self._h_last(outputs, batch)
        s1 = self.score_catalog(self.main_states(h_last.unsqueeze(1)).squeeze(1))
        B = h_last.shape[0]
        device = h_last.device

        if mode == "main":
            # RecSet-Base fallback: top-K of the main head alone (no MTP heads)
            u = s1.clone()
            chosen, scores = [], []
            for _ in range(K):
                top = u.max(dim=-1)
                chosen.append(top.indices)
                scores.append(top.values)
                u.scatter_(1, top.indices.unsqueeze(1), float("-inf"))
            return torch.stack(chosen, dim=1) + 1, torch.stack(scores, dim=1).mean(dim=1)

        if mode == "union":
            stack = [s1]
            for k in range(2, min(K, self.mtp_depths) + 1):
                stack.append(self._depth_scores(k, h_last, None, null_mask=True))
            u = torch.stack(stack, dim=0).max(dim=0).values           # (B, N)
            chosen, scores = [], []
            u_masked = u.clone()
            for _ in range(K):
                top = u_masked.max(dim=-1)
                chosen.append(top.indices)
                scores.append(top.values)
                u_masked.scatter_(1, top.indices.unsqueeze(1), float("-inf"))
            return torch.stack(chosen, dim=1) + 1, torch.stack(scores, dim=1).mean(dim=1)

        if mode in ("spec", "spec_beam"):
            b1_cnt = 1 if mode == "spec" else beam
            topb = torch.topk(s1, k=b1_cnt, dim=-1)                   # (B, b1_cnt)
            best_sets, best_scores = None, None
            for j in range(b1_cnt):
                b1 = topb.indices[:, j]
                chain = topb.values[:, j]
                chosen = [b1]
                taken = torch.zeros((B, self.n_items), device=device, dtype=torch.bool)
                taken.scatter_(1, b1.unsqueeze(1), True)
                cur = b1
                for k in range(2, min(K, self.mtp_depths) + 1):
                    sk = self._depth_scores(k, h_last, cur)
                    sk = sk.masked_fill(taken, float("-inf"))
                    topk = sk.max(dim=-1)
                    cur = topk.indices
                    chain = chain + topk.values
                    chosen.append(cur)
                    taken.scatter_(1, cur.unsqueeze(1), True)
                if best_scores is None:
                    best_sets, best_scores = torch.stack(chosen, dim=1), chain
                else:
                    better = chain > best_scores
                    best_scores = torch.where(better, chain, best_scores)
                    best_sets = torch.where(better.unsqueeze(1),
                                            torch.stack(chosen, dim=1), best_sets)
            return best_sets + 1, best_scores
        raise ValueError(f"unknown decode mode {mode}")
