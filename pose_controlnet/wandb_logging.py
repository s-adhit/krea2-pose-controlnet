"""wandb_logging.py — guardrail 2: explicit login + init, no silent no-op."""
import os


def init_wandb(cfg, run_name: str):
    if not cfg.wandb_enabled:
        print("[wandb] disabled via config — no metrics will be logged")
        return None

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("[wandb] WARNING: WANDB_API_KEY not set. Run `wandb login` or "
              "`export WANDB_API_KEY=...` before training. Continuing WITHOUT "
              "wandb logging.")
        return None

    import wandb
    wandb.login(key=api_key)
    run = wandb.init(project=cfg.wandb_project, name=run_name, config=vars(cfg))
    print(f"[wandb] connected — project={cfg.wandb_project} run={run_name}")
    return run