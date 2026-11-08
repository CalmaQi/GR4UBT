"""Relation-aware Stage-II policy optimization for every packaged dataset.

The four registered arms form the latest transition-reward x continuation-credit
factorial. All relation statistics and training windows come from strict TRAIN.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paths import DATA, ROOT, RUNS, save
from runner_eval import batch, evaluate
from ubt_bapo_relations import credit_surrogate


DATASETS = ("beauty", "cellphones", "ml1m", "lastfm")
STAGE2 = RUNS / "stage2"
ARMS = ((False, False), (False, True), (True, False), (True, True))


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def beauty_supervised_loss(runner, actor, batch_data):
    """Stage-I CE anchor with masks for the partial tail targets used in Stage II."""
    targets = batch_data["set_targets"].clamp_min(1)
    mask = batch_data["set_targets"] > 0

    def per_query(values):
        return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    out = actor(batch_data, actions0=targets - 1, return_trace=True)
    token = runner.token_ce(actor.model, out["queries"], targets)
    item = -out["log_probs"]
    total = token[:, 0].mean() + 0.5 * item[:, 0].mean()
    total = total + 0.5 * (per_query(token).mean() + per_query(item).mean())

    dropped = actor(batch_data, actions0=targets - 1, return_trace=True,
                    condition_drop=0.25)
    null_token = runner.token_ce(actor.model, dropped["queries"], targets)
    null_item = -dropped["log_probs"]
    total = total + 0.1 * (per_query(null_token).mean() + per_query(null_item).mean())
    return total


def load_stage1(dataset):
    """Recreate the matching Stage-I actor from its selected checkpoint."""
    if dataset == "beauty":
        import runner

        cfg = read_json(ROOT / "configs/beauty.json")
        data = dict(np.load(DATA / "beauty/assets.npz", allow_pickle=False))
        with np.load(DATA / "beauty/sentence_t5_u16.npz", allow_pickle=False) as z:
            data.update(text=z["text"], codes=z["codes"])
        path = RUNS / "beauty/stage1.pt"
        actor, state = runner.load_actor(data, cfg["stage1"]["K"], path,
                                         cfg["stage1"]["dim"])
        supervised = lambda a, b: beauty_supervised_loss(runner, a, b)
        return actor, data, state, cfg["stage1"]["K"], path, supervised

    if dataset == "cellphones":
        import train_stage1

        cfg = read_json(ROOT / "configs/cellphones.json")
        path = RUNS / "cellphones/full" / cfg["name"] / "stage1.pt"
        actor, data, state = train_stage1.load_tuned_checkpoint(path)
        recipe = state["config"]
        supervised = lambda a, b: train_stage1.joint_loss(a, b, recipe)[0]
        return actor, data, state, recipe["K"], path, supervised

    if dataset == "ml1m":
        import train_ml1m

        cfg = read_json(ROOT / "configs/ml1m.json")["full"]
        data = train_ml1m.load_data(cfg)
        path = RUNS / "ml1m/full" / cfg["name"] / "selected.pt"
        state = torch.load(path, map_location="cuda", weights_only=False)
        actor = train_ml1m.core.make_actor(cfg, data, state)
        supervised = lambda a, b: train_ml1m.core.loss_fn(a, b, cfg)[0]
        return actor, data, state, cfg["K"], path, supervised

    if dataset == "lastfm":
        import train_lastfm

        cfg = read_json(ROOT / "configs/lastfm.json")
        data = train_lastfm.core.load_data(cfg)
        path = RUNS / "lastfm/full" / cfg["name"] / "selected.pt"
        state = torch.load(path, map_location="cuda", weights_only=False)
        actor = train_lastfm.core.make_actor(cfg, data, state)
        supervised = lambda a, b: train_lastfm.core.loss_fn(a, b, cfg)[0]
        return actor, data, state, cfg["K"], path, supervised

    raise ValueError(f"unknown dataset: {dataset}")


def full_train_windows(data, k, history=50):
    """Make every next-item prefix and right-padded K-step target from TRAIN."""
    n_items = len(data["codes"])
    train_ends = data["train_ends"].astype(np.int64)
    uid = data["scorer_uid"].astype(np.int64)
    left = data["scorer_a"].astype(np.int64)
    right = data["scorer_b"].astype(np.int64)
    expected = int(np.maximum(train_ends - 1, 0).sum())
    if len(left) != expected or len(right) != expected or len(uid) != expected:
        raise ValueError("relation edges must be exactly the strict TRAIN transitions")
    bounds = np.r_[0, np.flatnonzero(uid[1:] != uid[:-1]) + 1, len(uid)]
    if len(bounds) != len(train_ends) + 1:
        raise ValueError("TRAIN edge users do not match the registered user cohort")

    rows_h = np.zeros((expected, history), dtype=np.int32)
    rows_y = np.zeros((expected, k), dtype=np.int32)
    offset = 0
    for user, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        if begin == end:
            raise ValueError("a registered user has no strict TRAIN transition")
        sequence = np.r_[left[begin], right[begin:end]].astype(np.int64)
        if len(sequence) != int(train_ends[user]):
            raise ValueError(f"TRAIN endpoint mismatch at user row {user}")
        if np.any(sequence < 1) or np.any(sequence > n_items):
            raise ValueError("TRAIN edge contains an invalid item id")
        for pos in range(1, len(sequence)):
            history_values = sequence[max(0, pos - history):pos]
            rows_h[offset, :len(history_values)] = history_values
            target_values = sequence[pos:min(pos + k, len(sequence))]
            rows_y[offset, :len(target_values)] = target_values
            offset += 1

    data = dict(data)
    data["train_h"] = rows_h
    data["train_y"] = rows_y
    if offset != expected:
        raise AssertionError("the generated training windows do not cover strict TRAIN edges")
    if data["val_y"].shape[1] != k or data["test_y"].shape[1] != k:
        raise ValueError("registered validation/test horizons do not match Stage I")
    return data


def build_train_transition(data, shrinkage=10.0):
    """Directed popularity-corrected next-item lift, estimated on TRAIN only."""
    n = len(data["codes"])
    left = data["scorer_a"].astype(np.int64) - 1
    right = data["scorer_b"].astype(np.int64) - 1
    if len(left) != int(np.maximum(data["train_ends"] - 1, 0).sum()):
        raise ValueError("transition reward source is not exactly strict TRAIN edges")
    counts = np.zeros((n, n), dtype=np.float32)
    if len(left):
        np.add.at(counts, (left, right), 1.0)
    outgoing = counts.sum(axis=1, keepdims=True)
    popularity = (counts.sum(axis=0) + 1.0) / (len(left) + n)
    counts += shrinkage * popularity[None, :]
    counts /= outgoing + shrinkage
    counts /= popularity[None, :]
    np.log(counts, out=counts)
    counts *= 1.0 / 3.0
    np.tanh(counts, out=counts)
    counts *= outgoing / (outgoing + shrinkage)
    metadata = {
        "source": "K-specific strict TRAIN scorer_a/scorer_b edges only",
        "edges": int(len(left)), "items": int(n), "shrinkage": float(shrinkage),
        "lift_scale": 3.0,
        "edge_sha256": hashlib.sha256(np.stack([left, right], axis=1).tobytes()).hexdigest(),
        "transition_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "formula": "out/(out+10) * tanh(log(smoothed P(next|prev)/pop(next))/3)",
    }
    return counts, metadata


def balanced_tail_rows(data, seed, per_user=2):
    counts = np.maximum(data["train_ends"].astype(np.int64) - 1, 0)
    offsets = np.r_[0, np.cumsum(counts)]
    if int(offsets[-1]) != len(data["train_y"]):
        raise ValueError("tail-window/user boundaries disagree")
    rng = np.random.default_rng(seed)
    rows = []
    for start, count in zip(offsets[:-1], counts):
        if count:
            chosen = rng.choice(int(count), min(per_user, int(count)), replace=False)
            rows.extend((int(start) + chosen).tolist())
    return rng.permutation(np.asarray(rows, dtype=np.int64))


def reward_laws(actions, log_law, targets, last, transition, use_transition,
                transition_weight=0.25):
    batch_size, group, horizon = actions.shape
    n_items = log_law.shape[-1]
    sampled, expected = [], []
    for t in range(horizon):
        valid = (targets[:, t] > 0).to(log_law.dtype)
        target0 = (targets[:, t] - 1).clamp_min(0)
        hit = F.one_hot(target0, num_classes=n_items).to(log_law.dtype)
        table = hit[:, None, :].expand(batch_size, group, n_items)
        if use_transition:
            previous = last[:, None].expand(batch_size, group) if t == 0 else actions[:, :, t - 1]
            table = table + float(transition_weight) * transition[previous]
        table = table * (valid[:, None, None] / float(horizon))
        law = log_law[:, :, t].exp()
        expected.append((law * table).sum(-1))
        sampled.append(table.gather(-1, actions[:, :, t, None]).squeeze(-1))
    return torch.stack(sampled, -1), torch.stack(expected, -1)


def sampled_replay(actor, batch_data, group, seed, horizon):
    actor.eval()
    batch_size = len(batch_data["input_ids"])
    generator = torch.Generator(device=batch_data["input_ids"].device).manual_seed(int(seed))
    with torch.no_grad():
        hidden = actor.encode_history(batch_data).repeat_interleave(group, dim=0)
        draw = actor.trajectory_from_hidden(hidden, K=horizon, generator=generator)
        actions = draw["actions0"].reshape(batch_size, group, horizon)
        old_logp = draw["log_probs"].reshape(batch_size, group, horizon)
    hidden = actor.encode_history(batch_data).repeat_interleave(group, dim=0)
    replay = actor.trajectory_from_hidden(hidden, actions0=actions.reshape(-1, horizon),
                                          K=horizon)
    log_law = replay["log_distributions"].reshape(batch_size, group, horizon,
                                                  actor.n_items)
    logp = replay["log_probs"].reshape(batch_size, group, horizon)
    if not torch.equal(old_logp, logp.detach()):
        raise ValueError("sample/replay policy mismatch")
    last_idx = (batch_data["seq_lens"] - 1).clamp_min(0)
    last = batch_data["input_ids"].gather(1, last_idx[:, None]).squeeze(1) - 1
    return actions, logp, log_law, last


def stage2_policy_loss(actor, batch_data, transition, use_transition,
                       continuation_credit, seed, group, horizon):
    actions, logp, log_law, last = sampled_replay(actor, batch_data, group, seed, horizon)
    sampled, expected = reward_laws(actions, log_law, batch_data["set_targets"], last,
                                    transition, use_transition)
    if continuation_credit:
        terminal = torch.zeros(logp.shape[:2], device=logp.device)
        loss = credit_surrogate(logp, sampled, expected, terminal, "integrated").mean()
    else:
        loss = -expected.sum(-1).mean()
    detail = {
        "policy_loss": float(loss.detach()),
        "sampled_step_reward": sampled.detach().mean((0, 1)).tolist(),
        "expected_step_reward": expected.detach().mean((0, 1)).tolist(),
        "sampled_total_reward": float(sampled.detach().sum(-1).mean()),
        "expected_total_reward": float(expected.detach().sum(-1).mean()),
        "sample_replay_equal": True,
        "group": int(group),
    }
    return loss, detail


def reference_kl(actor, reference, batch_data, seed, group, horizon):
    generator = torch.Generator(device=batch_data["input_ids"].device).manual_seed(int(seed))
    with torch.no_grad():
        hidden = reference.encode_history(batch_data).repeat_interleave(group, dim=0)
        old = reference.trajectory_from_hidden(hidden, K=horizon, generator=generator)
    actions = old["actions0"]
    hidden = actor.encode_history(batch_data).repeat_interleave(group, dim=0)
    live = actor.trajectory_from_hidden(hidden, actions0=actions, K=horizon)
    old_law, current_law = old["log_distributions"], live["log_distributions"]
    return (old_law.exp() * (old_law - current_law)).sum(-1).mean()


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
    total = sum(float(g.detach().float().square().sum()) for g in grads if g is not None)
    return math.sqrt(total)


def calibrate_scale(actor, data, transition, supervised, device, group, horizon):
    params = [p for p in actor.parameters() if p.requires_grad]
    rows = np.random.default_rng(20241105).choice(
        len(data["train_h"]), min(len(data["train_h"]), 4 * 16), replace=False)
    ratios, probes = [], []
    actor.eval()
    for i, begin in enumerate(range(0, len(rows), 16)):
        selected = rows[begin:begin + 16]
        b = batch(data, "train", selected, device=device)
        b["query_weights"] = torch.ones(len(selected), device=device)
        sup = supervised(actor, b)
        sup_norm = grad_norm(sup, params)
        policy, detail = stage2_policy_loss(actor, b, transition, False, True,
                                            2024110500 + i, group, horizon)
        policy_norm = grad_norm(policy, params)
        if not math.isfinite(policy_norm) or policy_norm <= 1e-12:
            raise FloatingPointError("TRAIN-only relation-aware RL calibration has zero gradient")
        ratios.append(0.25 * sup_norm / policy_norm)
        probes.append({"rows": len(selected), "supervised_gradient_norm": sup_norm,
                       "task_reward_gradient_norm": policy_norm,
                       "target_rl_to_ce_gradient_ratio": 0.25,
                       "suggested_rl_coefficient": ratios[-1], "probe_reward": detail})
    coefficient = float(np.median(ratios))
    return coefficient, {"source": "fixed TRAIN-only calibration probes; no VAL/TEST used",
                         "target_gradient_ratio": 0.25, "aggregation": "median",
                         "coefficient": coefficient, "batches": probes}


def stage2_folder(dataset, horizon, use_transition, continuation_credit):
    return (STAGE2 / dataset / f"k{horizon}" /
            f"transition_{int(use_transition)}_continuation_{int(continuation_credit)}")


def run_arms(dataset, actor, data, state, horizon, source_path, supervised,
             device="cuda", epochs=3, batch_size=32, updates=2, group=8,
             policy_lr=1e-6, transition_lr=1e-5):
    data = full_train_windows(data, horizon)
    transition_np, transition_meta = build_train_transition(data)
    transition = torch.as_tensor(transition_np, device=device)
    calibration_path = STAGE2 / dataset / f"k{horizon}" / "rl_scale_calibration.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    if calibration_path.exists():
        calibration = read_json(calibration_path)
        coefficient = float(calibration["coefficient"])
    else:
        coefficient, calibration = calibrate_scale(actor, data, transition, supervised,
                                                   device, group, horizon)
        save(calibration_path, calibration)

    results = []
    asset_file = DATA / dataset / (
        "sentence_t5_bge_sid.npz" if dataset == "lastfm" else "sentence_t5_u16.npz")
    for use_transition, continuation in ARMS:
        folder = stage2_folder(dataset, horizon, use_transition, continuation)
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists() and read_json(result_path).get("status") == "complete":
            results.append(read_json(result_path))
            continue

        torch.manual_seed(2024)
        actor, data, state, horizon, source_path, supervised = load_stage1(dataset)
        data = full_train_windows(data, horizon)
        reference = copy.deepcopy(actor).requires_grad_(False).eval()
        actor.eval()
        transition_params = [p for n, p in actor.named_parameters() if n.startswith("model.mu_")]
        policy_params = [p for n, p in actor.named_parameters() if not n.startswith("model.mu_")]
        optimizer = torch.optim.AdamW([
            {"params": policy_params, "lr": policy_lr, "name": "policy"},
            {"params": transition_params, "lr": transition_lr, "name": "transition"},
        ], weight_decay=0.0)

        initial = evaluate(actor, data, "val", horizon, bs=64)
        best = float(initial["SNDCG@10"])
        selected_epoch = 0
        validations = [{"epoch": 0, "SNDCG@10": best, "validation": initial}]
        torch.save({"state": actor.state_dict(), "feedback": actor.feedback_table,
                    "epoch": 0, "score": best, "stage": "relation_aware_initial"},
                   folder / "selected.pt")
        config = {
            "dataset": dataset, "K": horizon, "seed": 2024,
            "source_stage1_checkpoint": str(source_path.relative_to(ROOT)),
            "source_stage1_epoch": int(state.get("epoch", 0)),
            "text_embedding": "Sentence-T5 base, 768 dimensions",
            "feature_asset": str(asset_file.relative_to(ROOT)),
            "feature_asset_sha256": digest(asset_file),
            "split_asset_sha256": digest(DATA / dataset / "assets.npz"),
            "relation_aware_rl": True,
            "target_reward": "exact position-aligned catalog hit, divided by K",
            "collaborative_transition_reward": bool(use_transition),
            "transition_reward": transition_meta,
            "transition_reward_weight": 0.25,
            "continuation_credit": bool(continuation),
            "continuation_definition": "later-position expected rewards credited to earlier actions; immediate expected reward retained",
            "reward_estimator": "exact full-catalog expectation with on-policy sampled prefixes",
            "RL_coefficient": coefficient,
            "RL_coefficient_calibration": str(calibration_path.relative_to(ROOT)),
            "KL_direction": "reference P0 || current policy on reference-generated prefixes",
            "KL_target": 0.01, "KL_beta_initial": 0.5, "KL_beta_bounds": [0.5, 20.0],
            "optimizer": f"AdamW; policy LR={policy_lr:g}, transition LR={transition_lr:g}",
            "batch_size": batch_size, "updates_per_batch": updates,
            "rollout_group": group, "epochs": epochs,
            "sampling": "on-policy full-catalog multinomial with fresh exact replay",
            "training_rows": "two strict-TRAIN tail windows per user per epoch",
            "TEST_used_for_selection": False,
            "validation_selection": "greedy SNDCG@10",
            "alignment_loss": 0.0, "extra_dense_users": False,
        }
        save(folder / "config.json", config)
        beta, step = 0.5, 0
        epoch_reports = []
        tick = time.monotonic()
        for epoch in range(1, epochs + 1):
            rows = balanced_tail_rows(data, 2024 + epoch, per_user=2)
            epoch_tick, logs = time.monotonic(), []
            for begin in range(0, len(rows), batch_size):
                selected = rows[begin:begin + batch_size]
                b = batch(data, "train", selected, device=device)
                b["query_weights"] = torch.ones(len(selected), device=device)
                for _ in range(updates):
                    actor.eval()
                    optimizer.zero_grad(set_to_none=True)
                    torch.manual_seed(20240000 + step)
                    sup = supervised(actor, b)
                    policy, policy_detail = stage2_policy_loss(
                        actor, b, transition, use_transition, continuation,
                        202600000 + epoch * 100000 + step, group, horizon)
                    kl = reference_kl(actor, reference, b, 20250000 + step, 2, horizon)
                    warm = min((step + 1) / 32.0, 1.0)
                    loss = sup + coefficient * warm * policy + beta * kl
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite relation-aware Stage-II objective")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0,
                                                          error_if_nonfinite=True)
                    optimizer.step()
                    kl_value = float(kl.detach())
                    beta = float(np.clip(beta * math.exp(
                        0.05 * np.clip(kl_value / 0.01 - 1, -1, 1)), 0.5, 20.0))
                    logs.append({"step": step, "loss": float(loss.detach()),
                                 "supervised_loss": float(sup.detach()),
                                 "policy_loss": float(policy.detach()), "kl": kl_value,
                                 "beta": beta, "rl_warmup": warm,
                                 "gradient_norm": float(norm), "policy": policy_detail})
                    step += 1
            validation = evaluate(actor, data, "val", horizon, bs=64)
            score = float(validation["SNDCG@10"])
            if score > best:
                best, selected_epoch = score, epoch
                torch.save({"state": actor.state_dict(), "feedback": actor.feedback_table,
                            "epoch": epoch, "score": best, "stage": "relation_aware"},
                           folder / "selected.pt")
            validations.append({"epoch": epoch, "SNDCG@10": score,
                                "best_validation_SNDCG@10": best,
                                "validation": validation})
            summary = {"epoch": epoch, "optimizer_steps": step,
                       "seconds": time.monotonic() - epoch_tick,
                       "mean_loss": float(np.mean([x["loss"] for x in logs])),
                       "mean_policy_loss": float(np.mean([x["policy_loss"] for x in logs])),
                       "mean_KL": float(np.mean([x["kl"] for x in logs])),
                       "beta": beta, "validation": validation}
            save(folder / "progress.json", {"status": "running", "config": config,
                 "epoch_reports": validations, "selected_epoch": selected_epoch,
                 "optimizer_steps": step, "last_epoch": epoch, "epoch_summary": summary})
            print(json.dumps({"phase": "relation_aware_stage2", "dataset": dataset,
                              "transition": use_transition, "continuation": continuation,
                              **{k: v for k, v in summary.items() if k != "validation"}}),
                  flush=True)

        selected = torch.load(folder / "selected.pt", map_location=device, weights_only=False)
        actor.feedback_table = selected["feedback"]
        actor.load_state_dict(selected["state"])
        actor.eval()
        test_metrics = evaluate(actor, data, "test", horizon, bs=64,
                                path=folder / "test_predictions.npz")
        result = {"status": "complete", "config": config,
                  "initial_validation": initial, "selected_epoch": selected_epoch,
                  "selected_validation": next(x["validation"] for x in validations
                                               if x["epoch"] == selected_epoch),
                  "epoch_reports": validations, "optimizer_steps": step,
                  "training_wall_seconds": time.monotonic() - tick,
                  "test_metrics": test_metrics, "test_evaluations": 1,
                  "test_used_for_selection": False,
                  "checkpoint_sha256": digest(folder / "selected.pt")}
        save(result_path, result)
        save(folder / "progress.json", result)
        results.append(result)
    return results


def train_dataset(dataset, device="cuda", epochs=3, batch_size=32, updates=2, group=8,
                  policy_lr=1e-6, transition_lr=1e-5):
    actor, data, state, horizon, source_path, supervised = load_stage1(dataset)
    return run_arms(dataset, actor, data, state, horizon, source_path, supervised,
                    device, epochs, batch_size, updates, group, policy_lr, transition_lr)


def report_dataset(dataset):
    cfg = read_json(ROOT / f"configs/{dataset}.json")
    horizon = cfg["stage1"]["K"] if dataset == "beauty" else cfg.get("K")
    if horizon is None:
        horizon = cfg["full"]["K"] if dataset == "ml1m" else cfg["K"]
    reports = []
    for use_transition, continuation in ARMS:
        path = stage2_folder(dataset, horizon, use_transition, continuation) / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage-II result: {path.relative_to(ROOT)}")
        result = read_json(path)
        reports.append({"arm": path.parent.name, "selected_epoch": result["selected_epoch"],
                        "validation": result["selected_validation"],
                        "test_metrics": result["test_metrics"]})
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.evaluate_only and (not torch.cuda.is_available() or args.device != "cuda"):
        raise RuntimeError("Relation-aware Stage II requires a CUDA-capable GPU")
    torch.set_num_threads(4)
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cuda.matmul.allow_tf32 = True
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        print(f"[{dataset}]", flush=True)
        if args.evaluate_only:
            print(json.dumps(report_dataset(dataset), indent=2), flush=True)
        else:
            cfg = read_json(ROOT / f"configs/{dataset}.json")["stage2"]
            train_dataset(dataset, device=args.device, epochs=cfg["epochs"],
                          batch_size=cfg["batch_size"],
                          updates=cfg["updates_per_batch"],
                          group=cfg["rollout_group"],
                          policy_lr=cfg["policy_learning_rate"],
                          transition_lr=cfg["transition_learning_rate"])


if __name__ == "__main__":
    main()
