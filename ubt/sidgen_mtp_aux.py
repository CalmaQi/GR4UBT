"""SIDMTP1: single shared MTP readout stack (one head) instead of K-1 per-depth heads.

Motivation (open question this file exists to answer): with day-resolution
timestamps the *order* inside a next-K window is often not observable (Beauty:
43.9% of adjacent events share a day, ties broken by ASIN; 64.2% of users have
at least one tied pair among the last 3 events).  So "next-K" can be posed as

    traj  : joint distribution of the ORDERED trajectory
            p(y1) p(y2|y1) p(y3|y1,y2)                     -- what SIDMTP does
    setar : joint distribution of the unordered set, via a random-order
            autoregressive factorization p(S) = E_pi prod_k p(y_pi(k) | y_pi(<k))
    indep : product of marginals prod_k p(yk | x)            -- "Direct"

SIDMTP1 holds the architecture fixed across all three: ONE readout stack shared
by every depth (so the only per-depth capacity is a learned slot embedding),
which is the minimal change from SIDMTP's K-1 independent stacks.  The three
factors above then differ only in the *target/conditioning* fed to that head,
so the comparison isolates "does order carry signal" from "does capacity".

Everything else (trunk, main head, token/item/align losses, detach semantics,
decode modes) is inherited unchanged from SIDMTP.  NEW CODE; existing files
untouched.
"""
import torch
import torch.nn as nn

from sidgen_mtp import SIDMTP, _make_readout, _readout_forward

MODES = ("traj", "setar", "indep")


class SIDMTP1(SIDMTP):
    """SIDMTP with the K-1 depth heads collapsed into one shared depth head."""

    def __init__(self, *args, mtp1_mode="traj", **kwargs):
        super().__init__(*args, **kwargs)
        if mtp1_mode not in MODES:
            raise ValueError(f"unknown mtp1_mode {mtp1_mode}; expected one of {MODES}")
        self.mtp1_mode = mtp1_mode
        # "indep" drops the chain condition entirely: every depth is scored from
        # the context (+ its slot embedding) only.
        self.mtp1_cond_on = mtp1_mode != "indep"
        n_embd = self.gpt2.config.n_embd

        # replace the K-1 per-depth readouts with ONE shared stack
        del self.mtp_blocks, self.mtp_ln, self.mtp_mlp
        blocks, ln, mlp = _make_readout(n_embd)
        self.mtp1_blocks, self.mtp1_ln, self.mtp1_mlp = blocks, ln, mlp

        # per-slot identity is the only depth-specific parameter left
        self.mtp_slot = nn.Embedding(self.mtp_depths, n_embd)
        nn.init.zeros_(self.mtp_slot.weight)

        # single shared conditioning projector (parent indexes cond_proj[0] /
        # cond_null[0] once the list length is 1)
        del self.cond_proj, self.cond_null
        proj = nn.Linear(n_embd, n_embd)
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)
        self.cond_proj = nn.ModuleList([proj])
        self.cond_null = nn.ParameterList([nn.Parameter(torch.zeros(n_embd))])

    # ------------------------------------------------ setar: random-order targets
    def forward(self, batch, return_loss=True):
        if self.training and self.mtp1_mode == "setar" \
                and batch.get("set_targets") is not None:
            batch = dict(batch)
            batch["set_targets"] = self.permute_targets(batch["set_targets"])
        return super().forward(batch, return_loss=return_loss)

    @staticmethod
    def permute_targets(st, generator=None):
        """Shuffle the valid entries of every row (padding stays at -1).

        Applying a fresh random order per batch turns the depth chain into a
        random-order autoregressive factorization of the SET distribution,
        p(S) = E_pi prod_k p(y_pi(k) | y_pi(<k)), instead of the trajectory
        factorization p(y1)p(y2|y1)p(y3|y1,y2)."""
        B, K = st.shape
        r = torch.rand(B, K, device=st.device, generator=generator)
        r = r.masked_fill(st <= 0, 2.0)          # padding sinks to the back
        order = r.argsort(dim=1, stable=True)
        out = st.gather(1, order)
        return torch.where(st > 0, out, st)

    # idx = depth - 2 (parent passes the 0-based depth index positionally)
    def mtp_states(self, idx, h_last, cond_emb):
        slot = torch.full((h_last.shape[0],), int(idx) + 1,
                          device=h_last.device, dtype=torch.long)
        x = h_last + self.mtp_slot(slot)
        if self.mtp1_cond_on:
            x = x + self.cond_proj[0](cond_emb)
        fs = _readout_forward(self.mtp1_blocks, self.mtp1_ln, self.mtp1_mlp,
                              x.unsqueeze(1))
        return fs.squeeze(1)

    # ------------------------------------------------ diagnostics
    def depth_head_param_counts(self):
        """One shared stack + slot table, vs SIDMTP's (depths-1) stacks."""
        shared = sum(p.numel() for p in self.mtp1_blocks.parameters())
        shared += sum(p.numel() for m in (self.mtp1_ln, self.mtp1_mlp)
                      for p in m.parameters())
        shared += sum(p.numel() for p in self.mtp_slot.parameters())
        shared += sum(p.numel() for p in self.cond_proj.parameters())
        return {"shared_stack": shared, "depth_heads": 1,
                "equivalent_per_depth_stacks": self.mtp_depths - 1}
