"""
Main script for training + evaluating LLMs using Tinker (remote GPU service)
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

import tinker
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log

from critic_actor.environments import get_env
from critic_actor.replay_buffer import get_replay_buffer
from critic_actor.trainer import get_trainer
from critic_actor.utils import get_args, print_config, seed_everything
from critic_actor.utils.logging import AnsiColorLoggingFormatter

handler = logging.StreamHandler()
handler.setFormatter(AnsiColorLoggingFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def update_configs(
    args: argparse.Namespace,
    *configs: DictConfig | None,
) -> tuple[DictConfig | None, ...]:
    """Update configs with any specified + applicable command-line arguments."""
    for config in configs:
        if config is not None:
            for argname, argval in vars(args).items():
                if argval is not None and argname in config:
                    config[argname] = argval
    return configs


async def main() -> None:
    """Main async training function."""
    args = get_args(is_tinker=True)
    args.is_async = True
    seed_everything(args.seed)
    load_dotenv(override=True)

    # Load configs
    env_cfg = OmegaConf.load(f"./configs/environments/{args.env_config}.yaml")
    generator_cfg = OmegaConf.load(f"./configs/generator/{args.generator_config}.yaml")
    trainer_cfg = OmegaConf.load(f"./configs/trainer/{args.trainer_config}.yaml")
    replay_buffer_cfg = OmegaConf.load(
        f"./configs/replay_buffer/{args.replay_buffer_config}.yaml"
    )

    # Merge CLI overrides into all configs
    updated_cfgs = update_configs(
        args,
        env_cfg,
        generator_cfg,
        trainer_cfg,
        replay_buffer_cfg,
    )
    if args.verbose:
        cfg_names = ["env", "generator", "trainer", "replay_buffer"]
        for cfg_item, cfg_name in zip(updated_cfgs, cfg_names):
            if cfg_item is not None:
                print_config(cfg_item, cfg_name.upper())
    env_cfg, generator_cfg, trainer_cfg, replay_buffer_cfg = (
        updated_cfgs
    )
    cfg = trainer_cfg  # Primary config (owns all Tinker training attributes)
    cfg.run_name = args.run_name

    # WandB logging config: merge all sub-configs into one flat dict
    cfg_for_logger: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    cfg_for_logger.update(
        {k: v for k, v in vars(args).items() if k not in cfg_for_logger and v is not None}
    )
    for _cfg in [env_cfg, generator_cfg, replay_buffer_cfg]:
        if _cfg is not None:
            cfg_for_logger.update(
                {k: v for k, v in _cfg.items() if k not in cfg_for_logger and v is not None}
            )

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg_for_logger,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    cfg.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
    cfg.run_cmd = " ".join(sys.argv)

    # ------------------------------------------------------------------
    # Tinker client initialization + optional checkpoint resume
    # ------------------------------------------------------------------
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    if resume_info and args.resume_run:
        # Resume interrupted training — restore optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info("Resumed training from %s", resume_info["state_path"])
    elif cfg.load_checkpoint_path:
        # Start from a specific checkpoint — fresh optimizer, pre-loaded weights
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info("Loaded weights from %s", cfg.load_checkpoint_path)
    else:
        # Fresh run — create LoRA adapter on top of the base model
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )
        logger.info(
            "Created fresh LoRA (rank=%d) on top of %s", cfg.lora_rank, cfg.model_name
        )

    # ------------------------------------------------------------------
    # Environments and replay buffer
    # ------------------------------------------------------------------
    env = get_env(**env_cfg)
    # Propagate run metadata to environments (e.g. for logging)
    env.run_url = cfg.get("run_url")
    env.run_cmd = cfg.get("run_cmd")
    eval_env = env

    # Enable prompt logging for rl_prompt envs
    if hasattr(env, "prompt_log_dir"):
        env.prompt_log_dir = os.path.join(cfg.log_path, "prompts")
        logger.info("Prompt logs will be saved to %s", env.prompt_log_dir)

    hf_tokenizer = training_client.get_tokenizer()
    replay_buffer = get_replay_buffer(hf_tokenizer=hf_tokenizer, **replay_buffer_cfg)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = get_trainer(
        cfg.trainer_name,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        generator_cfg=generator_cfg,
        replay_buffer=replay_buffer,
        env=env,
        eval_env=eval_env,
        ml_logger=ml_logger,
        run_name=args.run_name,
    )
    # Full training run
    await trainer.train(start_batch=start_batch, end_batch=cfg.num_batches)

    # Save final checkpoint (skip if training was already complete)
    if start_batch < cfg.num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            loop_state={"batch": cfg.num_batches},
            kind="both",
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
