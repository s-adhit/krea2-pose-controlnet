"""Fresh or explicitly resumed isolated 32-sample R64 flow-MSE capacity trainer.

This is a thin harness over ``train.py``'s existing fresh model, flow loss,
optimizer, scheduler, latent/text cache, telemetry, and checkpoint machinery.
Fresh invocations always build new LoRA weights.  Resuming is deliberately
explicit and only accepts a validated checkpoint from the same experiment.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import train
from pose_controlnet.checkpointing import load_training_state, save_training_state
from pose_controlnet.config import TrainConfig
from pose_controlnet.data import PreparedLatentShardDataset, collate
from pose_controlnet.capacity_resolution import (
    AlternateResolutionDataset, load_alternate_resolution_cache, native_resolution_provenance,
    prepare_alternate_resolution_cache,
)
from pose_controlnet.diffusion import forward_pose_control, make_flow_pair, patchify_and_position, sample_controlled_pose_exposure_timestep
from pose_controlnet.keypoint_critic import FixedBoxKeypointRCNNCritic, differentiable_pose_loss
from pose_controlnet.keypoint_critic_audit import assert_frozen_no_parameter_grad
from pose_controlnet.capacity_pose import load_capacity_pose_records
from pose_controlnet.model import audit_control_model, build_pose_model, trainable_params, trainable_state_dict
from pose_controlnet.overfit_capacity import (
    OVERFIT_ACCUMULATION, OVERFIT_LR, OVERFIT_MAX_STEPS, OVERFIT_MICROBATCH, OVERFIT_STEPS,
    OVERFIT_WARMUP, OVERFIT_CHECKPOINT_STEPS, SelectedDeterministicBatches, SelectedLatentShardDataset,
    CapacityScientificConfig, assert_fresh_initialization, assert_overfit_contract, buckets_for_resolution,
    canonical_resolution_policy, experiment, is_overfit_checkpoint_step, manifest_stems, parameter_audit,
    per_step_exposures, should_continue_overfit, validate_capacity_scientific_config, validate_manifest,
)
from pose_controlnet.pose_reward_tools import combine_flow_and_pose_loss
from pose_controlnet.seed import set_seed
from pose_controlnet.vae_preprocessing import decode_normalized_latents_autograd, load_krea_vae, qwen_decoded_to_unit_rgb
from pose_controlnet.wandb_logging import TrainingTelemetry
from scripts.audit_keypoint_critic import _person_tensors, _usable
from scripts.audit_keypoint_critic_timestep import unpatchify_latent_tokens


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment", help="Optional explicit identity; must match the dynamic scientific configuration.")
    value.add_argument("--base-experiment", default="mixed32", choices=("mixed32",))
    value.add_argument("--resolution", default="native", choices=("native", "current", "768"))
    value.add_argument("--pose-loss", default="none", choices=("none", "normalized_coordinate_huber"))
    value.add_argument("--lambda-pose", type=float, default=0.0)
    value.add_argument("--forced-pose-exposure-probability", type=float, default=0.0)
    value.add_argument("--pose-timestep-min", type=float)
    value.add_argument("--pose-timestep-max", type=float)
    value.add_argument("--pose-target-sidecar", type=Path, help="Required immutable authoritative targets when pose loss is enabled.")
    value.add_argument("--raw-ckpt", default="/lambda/nfs/adhit/krea2-pose/models/krea-2-raw/raw.safetensors")
    value.add_argument("--latent-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_latents")
    value.add_argument("--text-conditioning-root", default="/lambda/nfs/adhit/krea2-pose/text_conditioning")
    value.add_argument("--checkpoint-root", default="/lambda/nfs/adhit/krea2-pose/overfit_capacity/checkpoints")
    value.add_argument("--evaluation-root", default="/lambda/nfs/adhit/krea2-pose/overfit_capacity/evaluation")
    value.add_argument("--resolution-cache-root", default="/lambda/nfs/adhit/krea2-pose/overfit_capacity/resolution_cache")
    value.add_argument("--dataset-root", default="/lambda/nfs/adhit/krea2-pose/posebridge_hf")
    value.add_argument("--wandb-project", default="Krea-2-PoseControl-Lora")
    value.add_argument("--wandb-mode", default="online")
    value.add_argument("--no-wandb", action="store_true")
    value.add_argument("--resume", help="Explicit exact capacity checkpoint from this experiment; never auto-selects")
    value.add_argument("--preflight", action="store_true", help="CPU-only contract validation; does not build a model or write outputs")
    return value


def scientific_config(args: argparse.Namespace) -> CapacityScientificConfig:
    # Preserve read-only resume validation for the already-completed legacy
    # named experiments.  The public dynamic runner always supplies
    # --base-experiment and therefore cannot create new legacy namespaces.
    if (not hasattr(args, "base_experiment") and getattr(args, "experiment", None)) or (
        getattr(args, "experiment", "") in {
            "overfit32-coco-r64-mse", "overfit32-humanart-painting-r64-mse", "overfit32-humanart-real-r64-mse",
            "overfit32-humanart-sculpture-r64-mse", "overfit32-danbooru-r64-mse", "overfit32-mixed-r64-mse",
        } and args.resolution in ("native", "current") and args.pose_loss == "none" and args.lambda_pose == 0
    ):
        return validate_capacity_scientific_config(CapacityScientificConfig(base_experiment="mixed32"))
    config = validate_capacity_scientific_config(CapacityScientificConfig(
        base_experiment=args.base_experiment, resolution=args.resolution, pose_loss=args.pose_loss,
        lambda_pose=args.lambda_pose, forced_pose_exposure_probability=args.forced_pose_exposure_probability,
        pose_timestep_min=args.pose_timestep_min, pose_timestep_max=args.pose_timestep_max,
    ))
    if args.experiment is not None and args.experiment != config.experiment_name:
        raise ValueError(f"--experiment={args.experiment!r} conflicts with dynamic identity {config.experiment_name!r}")
    args.experiment = config.experiment_name
    return config


def build_config(args: argparse.Namespace) -> TrainConfig:
    scientific = scientific_config(args)
    cfg = TrainConfig(raw_ckpt=args.raw_ckpt, shard_dir=args.latent_root, ckpt_dir=args.checkpoint_root,
        run_name=args.experiment, rank=64, alpha=64, lr=OVERFIT_LR, microbatch_size=OVERFIT_MICROBATCH,
        gradient_accumulation_steps=OVERFIT_ACCUMULATION, max_steps=OVERFIT_MAX_STEPS,
        allow_extended_training=True, warmup_steps=OVERFIT_WARMUP, caption_dropout=.1, control_dropout=.0,
        # Generic cadence is disabled here; required_checkpoint_steps and the
        # authoritative helper below are the sole capacity save schedule.
        val_every=OVERFIT_MAX_STEPS + 1, save_every=OVERFIT_MAX_STEPS + 1, diagnostics_every=50,
        required_checkpoint_steps=OVERFIT_CHECKPOINT_STEPS, wandb_project=args.wandb_project,
        wandb_enabled=not args.no_wandb, wandb_mode=args.wandb_mode,
        metrics_jsonl_path=str(Path(args.checkpoint_root) / args.experiment / "metrics.jsonl"))
    assert_overfit_contract(rank=cfg.rank, warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps, lr=cfg.lr,
        microbatch_size=cfg.microbatch_size, accumulation_steps=cfg.gradient_accumulation_steps,
        pose_reward_enabled=scientific.pose_loss != "none", critic_enabled=scientific.pose_loss != "none",
        objective="flow_mse_plus_pose" if scientific.pose_loss != "none" else "flow_mse",
        checkpoint_steps=cfg.required_checkpoint_steps, scientific_config=scientific)
    return cfg


_RESUME_CONFIG_FIELDS = (
    "raw_ckpt", "shard_dir", "text_conditioning_root", "run_name", "rank", "alpha", "lr",
    "microbatch_size", "gradient_accumulation_steps", "max_steps", "warmup_steps", "max_grad_norm",
    "caption_dropout", "control_dropout", "compile", "fused_adamw", "gradient_checkpointing",
    "gradient_checkpointing_blocks", "mu_x1", "mu_y1", "mu_x2", "mu_y2",
    "timestep_aux_prob", "timestep_aux_min", "timestep_aux_max",
)


def _resume_error(message: str) -> ValueError:
    return ValueError(f"Unsafe overfit-capacity resume refused: {message}")


def _canonical_json(value: Any) -> str:
    """Compare JSON metadata and torch state despite tuple/list serialization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_resume_metadata(metadata: dict[str, Any], *, args: argparse.Namespace,
                              stems: tuple[str, ...], checkpoint_dir: Path) -> None:
    if metadata.get("experiment") != args.experiment:
        raise _resume_error("checkpoint metadata belongs to another experiment")
    if Path(metadata.get("checkpoint_dir", "")).resolve() != checkpoint_dir.resolve():
        raise _resume_error("checkpoint metadata has an incompatible checkpoint namespace")
    if tuple(metadata.get("stems", ())) != stems:
        raise _resume_error("checkpoint metadata does not match the exact current COCO-32 manifest")
    if metadata.get("checkpoint_steps") != list(OVERFIT_STEPS):
        raise _resume_error("checkpoint metadata has a different scientific checkpoint schedule")
    scientific = scientific_config(args)
    expected_objective = "flow_mse_plus_pose" if scientific.pose_loss != "none" else "flow_mse_only"
    legacy_metadata = "scientific_config" not in metadata and expected_objective == "flow_mse_only"
    if metadata.get("objective") != expected_objective or (not legacy_metadata and metadata.get("scientific_config") != scientific.__dict__):
        raise _resume_error("checkpoint metadata loss/resolution scientific configuration differs")
    if bool(metadata.get("pose_reward_enabled")) != (scientific.pose_loss != "none") or bool(metadata.get("critic_enabled")) != (scientific.pose_loss != "none"):
        raise _resume_error("checkpoint metadata critic configuration differs")
    if metadata.get("fresh_lora_checkpoint_loaded") is not False:
        raise _resume_error("checkpoint metadata does not prove a fresh-LoRA experiment")
    audit = metadata.get("parameter_audit")
    if not isinstance(audit, dict) or audit != parameter_audit():
        raise _resume_error("checkpoint metadata has an incompatible rank-64 trainable architecture")
    actual_audit = metadata.get("actual_model_audit")
    if not isinstance(actual_audit, dict) or actual_audit.get("lora_rank") != 64 or actual_audit.get("lora_target_modules") != 224:
        raise _resume_error("checkpoint metadata has an incompatible LoRA architecture audit")


def validate_overfit_resume_checkpoint(resume: str | Path, *, args: argparse.Namespace, cfg: TrainConfig,
                                       stems: tuple[str, ...], checkpoint_dir: Path) -> tuple[Path, dict]:
    """Read-only, fail-closed validation for an exact overfit continuation."""
    path = Path(resume).expanduser().resolve()
    if path.parent != checkpoint_dir.resolve() or not path.is_file():
        raise _resume_error("checkpoint must be an explicit file in this experiment's checkpoint directory")
    metadata_path = checkpoint_dir / "experiment_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _resume_error(f"required experiment metadata is unreadable: {error}") from error
    if not isinstance(metadata, dict):
        raise _resume_error("required experiment metadata is malformed")
    _validate_resume_metadata(metadata, args=args, stems=stems, checkpoint_dir=checkpoint_dir)
    state = load_training_state(path)
    step = state["global_step"]
    if path.name != f"step_{step:06d}.pt" or step not in OVERFIT_STEPS:
        raise _resume_error("checkpoint filename or embedded global step is not an authoritative capacity milestone")
    if state["scheduler"].get("step_count") != step or state["scheduler"].get("warmup_steps") != OVERFIT_WARMUP:
        raise _resume_error("checkpoint scheduler progress or warmup is incompatible")
    expected_epoch = 0 if step == 0 else (step * OVERFIT_ACCUMULATION - 1) // 32
    expected_position = 0 if step == 0 else (step * OVERFIT_ACCUMULATION - 1) % 32 + 1
    if (state["epoch"], state["batch_position"]) != (expected_epoch, expected_position):
        raise _resume_error("checkpoint sampler progress is incompatible with exact 32-sample cycling")
    capacity = state.get("overfit_capacity")
    if not isinstance(capacity, dict) or capacity.get("fresh_lora_checkpoint_loaded") is not False:
        raise _resume_error("checkpoint lacks the required overfit-capacity provenance")
    if any(capacity.get(key) != value for key, value in per_step_exposures(step).items()):
        raise _resume_error("checkpoint exposure accounting does not match its embedded step")
    saved_config = state["config"]
    if _canonical_json(saved_config) != _canonical_json(metadata.get("config")):
        raise _resume_error("checkpoint configuration does not match experiment metadata")
    current_config = asdict(cfg)
    if any(saved_config.get(field) != current_config.get(field) for field in _RESUME_CONFIG_FIELDS):
        raise _resume_error("checkpoint configuration is incompatible with this requested experiment")
    if saved_config.get("rank") != 64 or saved_config.get("alpha") != 64 or saved_config.get("lr") != OVERFIT_LR:
        raise _resume_error("checkpoint does not use rank 64 with the authoritative LR")
    return path, state


def reconcile_metrics_for_resume(path: Path, step: int) -> Path | None:
    """Keep <= checkpoint metrics authoritative and preserve re-executed tail verbatim.

    A crash can leave logged updates newer than the last durable checkpoint.
    Those values will be recomputed exactly after resume, so retain them in a
    sidecar and atomically trim only the authoritative JSONL tail.
    """
    if not path.exists():
        return None
    raw = path.read_bytes()
    retained: list[bytes] = []
    replayed: list[bytes] = []
    for line in raw.splitlines(keepends=True):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise _resume_error(f"metrics history is malformed: {error}") from error
        logged_step = item.get("global_step")
        if not isinstance(logged_step, int) or logged_step < 0:
            raise _resume_error("metrics history has an invalid global_step")
        (retained if logged_step <= step else replayed).append(line if line.endswith(b"\n") else line + b"\n")
    if not replayed:
        return None
    backup = path.with_name(f"{path.stem}.pre_resume_after_step_{step:06d}.jsonl")
    if backup.exists():
        if backup.read_bytes() != raw:
            raise _resume_error(f"metrics backup already exists with unexpected contents: {backup}")
    else:
        _atomic_write_bytes(backup, raw)
    _atomic_write_bytes(path, b"".join(retained))
    return backup


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def restore_overfit_resume_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                 scheduler: train.OptimizerStepWarmup, generator: torch.Generator,
                                 state: dict) -> tuple[int, int, int]:
    global_step, epoch, batch_position, flow_state = train.restore_full_training_state(model, optimizer, scheduler, state)
    if scheduler.step_count != global_step or flow_state is None:
        raise _resume_error("restored scheduler or flow-generator state is invalid")
    generator.set_state(flow_state)
    return global_step, epoch, batch_position


def preflight(args: argparse.Namespace) -> dict:
    scientific = scientific_config(args)
    # Dynamic experiment names deliberately do not become manifest names.
    legacy = args.experiment in {
        "overfit32-coco-r64-mse", "overfit32-humanart-painting-r64-mse", "overfit32-humanart-real-r64-mse",
        "overfit32-humanart-sculpture-r64-mse", "overfit32-danbooru-r64-mse", "overfit32-mixed-r64-mse",
    }
    manifest_experiment = args.experiment if legacy else "overfit32-mixed-r64-mse"
    stems = validate_manifest(manifest_experiment)
    cfg = build_config(args)
    checkpoint_dir, evaluation_dir = Path(args.checkpoint_root) / args.experiment, Path(args.evaluation_root) / args.experiment
    if checkpoint_dir == evaluation_dir or checkpoint_dir in evaluation_dir.parents or evaluation_dir in checkpoint_dir.parents:
        raise ValueError("Capacity checkpoint and evaluation namespaces must be disjoint")
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint, _ = validate_overfit_resume_checkpoint(args.resume, args=args, cfg=cfg, stems=stems, checkpoint_dir=checkpoint_dir)
    else:
        assert_fresh_initialization()
    if scientific.pose_loss != "none" and args.pose_target_sidecar is None:
        raise ValueError("--pose-target-sidecar is required when pose loss is enabled")
    return {"experiment": args.experiment, "manifest": str(experiment(manifest_experiment).manifest), "stems": list(stems),
        "config": asdict(cfg), "checkpoint_steps": list(OVERFIT_STEPS), "parameter_audit": parameter_audit(),
        "fresh_lora_checkpoint_loaded": False, "training_set_equals_evaluation_set": True,
        "checkpoint_dir": str(checkpoint_dir), "evaluation_dir": str(evaluation_dir),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "scientific_config": scientific.__dict__, "resolution_policy": scientific.resolution,
        "resolution_bucket_shapes": [list(bucket) for bucket in (buckets_for_resolution(scientific.resolution) or ())]}


def _resolution_provenance(native: dict[str, Any], alternate: dict[str, Any] | None,
                           scientific: CapacityScientificConfig) -> dict[str, Any]:
    """Join native and requested geometry without allowing either to be inferred."""
    rows: dict[str, Any] = {}
    for stem, native_geometry in native["samples"].items():
        requested = native_geometry if alternate is None else alternate["samples"][stem]["geometry"]
        rows[stem] = {
            "original_source_dimensions": requested["source_size"],
            "original_native_bucket": native_geometry["bucket"],
            "requested_bucket": requested["bucket"], "resize_geometry": requested["resized_size"],
            "crop_geometry": requested["crop_box"], "resulting_latent_dimensions": requested["latent_size"],
        }
    return {"format_version": 1, "scientific_config": scientific.__dict__,
            "native_geometry_source": native["source"], "samples": rows}


def _pose_records_for_resolution(sidecar: Path, data, stems: tuple[str, ...], *,
                                 experiment_name: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Use only the immutable exact-manifest JSONL through paired 768 geometry."""
    return load_capacity_pose_records(
        sidecar=sidecar, experiment_name=experiment_name,
        data=data, stems=stems,
    )


def main() -> None:
    args = parser().parse_args(); proof = preflight(args)
    if args.preflight:
        print(json.dumps(proof, indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError("Run capacity training only from the GH200 host shell with CUDA visible")
    checkpoint_dir = Path(proof["checkpoint_dir"])
    if not args.resume and checkpoint_dir.exists(): raise FileExistsError(f"Refusing to collide with an existing capacity run: {checkpoint_dir}")
    scientific = scientific_config(args)
    set_seed(42); cfg = build_config(args); device = torch.device("cuda")
    base = PreparedLatentShardDataset(cfg.shard_dir, "train", text_conditioning_root=args.text_conditioning_root)
    selected = SelectedLatentShardDataset(base, proof["stems"])
    native = native_resolution_provenance(selected, proof["stems"])
    vae = None
    alternate = None
    if scientific.resolution == "native":
        data = selected
    else:
        vae = load_krea_vae(device)
        alternate = prepare_alternate_resolution_cache(
            selected=selected, config=scientific, dataset_root=args.dataset_root,
            cache_root=args.resolution_cache_root, vae=vae, device=device,
        )
        data = AlternateResolutionDataset(selected, alternate, args.resolution_cache_root, scientific.experiment_name)
    resolution_manifest = _resolution_provenance(native, alternate, scientific)
    plan = SelectedDeterministicBatches(data, cfg.microbatch_size)
    model = build_pose_model(cfg.raw_ckpt, 64, 64, "cuda"); actual_audit = audit_control_model(model, rank=64); model.train()
    if actual_audit["trainable_parameters"] != parameter_audit()["trainable_parameter_count"]: raise AssertionError("Fresh model parameter audit disagrees with static contract")
    optimizer = train.build_optimizer(model, cfg); scheduler = train.OptimizerStepWarmup(optimizer, cfg.warmup_steps)
    pose_metadata = None
    pose_records = None
    critic = None
    if scientific.pose_loss != "none":
        if vae is None:
            vae = load_krea_vae(device)
        critic = FixedBoxKeypointRCNNCritic().to(device).eval()
        assert_frozen_no_parameter_grad(vae, critic)
        pose_metadata, pose_records = _pose_records_for_resolution(
            args.pose_target_sidecar, data, tuple(proof["stems"]), experiment_name=scientific.experiment_name,
        )
    if not args.resume:
        _write(checkpoint_dir / "resolution_manifest.json", resolution_manifest)
        _write(checkpoint_dir / "experiment_metadata.json", {**proof, "actual_model_audit": actual_audit,
            "base_checkpoint_provenance": getattr(model, "_krea_checkpoint_report", None), "fresh_lora_checkpoint_loaded": False,
            "objective": "flow_mse_plus_pose" if scientific.pose_loss != "none" else "flow_mse_only",
            "pose_reward_enabled": scientific.pose_loss != "none", "critic_enabled": scientific.pose_loss != "none",
            "scientific_config": scientific.__dict__, "resolution_manifest": "resolution_manifest.json",
            "pose_target_sidecar": str(args.pose_target_sidecar) if args.pose_target_sidecar else None,
            "pose_target_metadata": pose_metadata,
            "caption_dropout": "existing deterministic 0.10 cached-unconditional substitution; no spatial augmentation"})
    stopped = False
    def stop_handler(signum, _frame):
        nonlocal stopped; stopped = True; print(f"received signal {signum}; stopping at optimizer boundary", flush=True)
    signal.signal(signal.SIGINT, stop_handler); signal.signal(signal.SIGTERM, stop_handler)
    generator = torch.Generator(device=device).manual_seed(42); optimizer.zero_grad(set_to_none=True)
    epoch = batch_position = global_step = 0
    def save(step: int) -> None:
        path = checkpoint_dir / f"step_{step:06d}.pt"
        if step not in OVERFIT_STEPS:
            raise AssertionError(f"Refusing to save outside the authoritative capacity schedule: {step}")
        save_training_state(path, {"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "global_step": step, "epoch": epoch, "batch_position": batch_position, "rng": train._capture_rng(),
            "flow_generator_state": generator.get_state(), "config": asdict(cfg), "overfit_capacity": {"fresh_lora_checkpoint_loaded": False, **per_step_exposures(step)}}, overwrite=False)
    if args.resume:
        resume_path, resume_state = validate_overfit_resume_checkpoint(args.resume, args=args, cfg=cfg,
            stems=tuple(proof["stems"]), checkpoint_dir=checkpoint_dir)
        global_step, epoch, batch_position = restore_overfit_resume_state(model, optimizer, scheduler, generator, resume_state)
        backup = reconcile_metrics_for_resume(Path(cfg.metrics_jsonl_path), global_step)
        print(f"[resume] restored exact state from {resume_path} at optimizer step {global_step} "
              f"(epoch={epoch}, batch_position={batch_position}); metrics backup={backup}", flush=True)
    else:
        save(0)
    telemetry = TrainingTelemetry(cfg, cfg.run_name)
    try:
        while should_continue_overfit(global_step) and not stopped:
            batches = plan.for_epoch(epoch); started = time.monotonic(); losses = []
            for accumulation_index in range(OVERFIT_ACCUMULATION):
                if batch_position == len(batches): epoch, batch_position, batches = epoch + 1, 0, plan.for_epoch(epoch + 1)
                batch_items = [data[index] for index in batches[batch_position]]; batch = collate(batch_items); batch_position += 1
                batch["stems"] = [item["stem"] for item in batch_items]
                train.apply_cached_caption_dropout(batch, data.text_conditioning.unconditional, cfg.caption_dropout, cfg.seed,
                    global_step * OVERFIT_ACCUMULATION + accumulation_index)
                if scientific.pose_loss == "none":
                    with torch.autocast("cuda", dtype=torch.bfloat16): loss, diagnostics = train._flow_loss(model, None, batch, cfg, device, generator, gradient_checkpointing_blocks=0)
                else:
                    # Reuse the audited fixed-box critic semantics verbatim;
                    # This historical capacity branch shares the canonical
                    # pose-consistency implementation with production.
                    from pose_controlnet.pose_consistency import production_pose_consistency_loss
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, diagnostics = production_pose_consistency_loss(
                            model, vae, critic, batch, pose_records, cfg, device, generator,
                            pose_loss_name=scientific.pose_loss, lambda_pose=scientific.lambda_pose,
                            timestep_min=scientific.pose_timestep_min, timestep_max=scientific.pose_timestep_max,
                            forced_exposure_probability=scientific.forced_pose_exposure_probability,
                        )
                (loss / OVERFIT_ACCUMULATION).backward(); losses.append(float(loss.item()))
            lr = scheduler.current_update_learning_rates[0]; grad = train.optimizer_update(optimizer, scheduler, trainable_params(model), cfg.max_grad_norm)
            global_step += 1; elapsed = time.monotonic() - started
            telemetry.log_train(loss=sum(losses) / len(losses), learning_rate=lr, global_grad_norm=grad, sec_per_step=elapsed, samples_per_second=OVERFIT_ACCUMULATION / elapsed, step=global_step)
            if is_overfit_checkpoint_step(global_step): save(global_step)
    finally:
        telemetry.close()


if __name__ == "__main__": main()
