"""
Argument parsing helpers
"""

import argparse
import logging
import os

from omegaconf import OmegaConf

from .setup import get_run_name

logger = logging.getLogger(__name__)


def get_args(is_tinker: bool = False) -> argparse.Namespace:
    """
    Load and process experiment arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="cria-tinker")

    # Necessary arguments + configs (to load default args from)
    parser.add_argument("--is_async", action="store_true", help="Use asynchronous environment")
    parser.add_argument(
        "--resume_run",
        action="store_true",
        default=False,
        help="Resume from checkpoint in log_path",
    )

    parser.add_argument("--env_config", type=str, help="Environment config to load default args")
    parser.add_argument("--generator_config", type=str, help="Generator config; ditto")
    parser.add_argument("--trainer_config", type=str, help="Trainer config; ditto")
    parser.add_argument("--replay_buffer_config", type=str, help="Replay buffer config; ditto")

    # Optional arguments -> overrides the defaults in trainer_config if specified
    ## Model (specified in trainer_config)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--lora_rank", type=int)

    ## Critic-Actor
    parser.add_argument("--num_actions", type=int)

    ## Claude Agent SDK
    parser.add_argument("--model", type=str)
    parser.add_argument("--permission_mode", type=str)
    parser.add_argument("--effort", type=str)
    parser.add_argument("--max_agent_turns", type=int)
    parser.add_argument("--use_claude_tools", action="store_true")

    ## Environment    
    parser.add_argument(
        "--num_fewshots",
        type=int,
        help="Number of few-shot samples to add to the prompt (only supported for some envs)",
    )
    parser.add_argument(
        "--actions_only",
        action="store_true",
        default=None,
        help="If True, remove thoughts / reasoning traces from observed assistant messages",
    )
    parser.add_argument(
        "--hide_observations",
        action="store_true",
        default=None,
        help="If True, hide observations prior to the last one, e.g., for more 'human-like' context",
    )

    ## Evaluation / Eval Environment
    parser.add_argument(
        "--best_metric",
        type=str,
        help="Metric to save best checkpoints on",
    )
    ### Number of tasks per split
    parser.add_argument("--num_train_samples", type=int, help="Number of training samples")
    parser.add_argument("--num_eval_samples", type=int, help="Number of evaluation samples")
    parser.add_argument("--max_concurrent_judges", type=int, help="Max concurrent LLM judge calls")
    parser.add_argument("--num_test_samples", type=int, help="Number of test samples")

    ## Tinker logging + checkpointing
    parser.add_argument("--base_url", type=str, help="Tinker base URL")
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs",
        help=(
            "Parent directory for logging and saving Tinker checkpoints. Actual path is "
            "automatically determined (and created) based on our specified argparse args"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints",
        help=(
            "Parent directory for saving other checkpoints and data (e.g., replay buffer samples)."
            " Similar to above, actual path is automatically determined (and created)"
        ),
    )
    parser.add_argument("--load_checkpoint_path", type=str, help="Path to load Tinker checkpoint")

    ## Model Generation
    parser.add_argument("--num_batches", type=int, help="Number of training batches")
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float)

    ## Training Rollouts
    parser.add_argument(
        "--mean_center",
        action="store_true",
        default=None,
        help="Mean-center rewards",
    )
    parser.add_argument("--discount_factor", type=float)
    parser.add_argument(
        "--group_size",
        type=int,
        help="Number of rollouts to generate per sample; will override if specified",
    )
    parser.add_argument(
        "--eval_group_size",
        type=int,
        help="Group size for evaluation; will override if specified. Set >1 if we want error bars",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of unique tasks or problems used for training in each training step",
    )

    ## Training Updates
    parser.add_argument("--negative_rewards", action="store_true", help="Use negative rewards")
    parser.add_argument("--advantage_threshold", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--kl_penalty_coef", type=float)
    parser.add_argument("--kl_discount_factor", type=float)
    parser.add_argument(
        "--num_substeps",
        type=int,
        help=(
            "Number of actual updates per training step. "
            "Splits total_episode_steps = batch_size * group_size * len(traejectory) into "
            "`num_substeps` mini-batches, where each mini-batch is used for one optimizer update. "
            "By default, we adjust total_episode_steps to be a multiple of `num_substeps`."
        ),
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        help=(
            "Alternative to --num_substeps; size of each mini-batch during updates. "
            "If specified and num_substeps is None, then we set num_substeps = "
            "total_episode_steps // mini_batch_size. If both specified, then we (super)sample the "
            "training data s.t. len(training_data) = mini_batch_size * num_substeps."
        ),
    )
    
    ## Eval and checkpointing
    parser.add_argument("--eval_every", type=int, help="Iters to evaluate, 0 = disabled")
    parser.add_argument("--eval_mode", type=str, help="Eval mode: 'majority_vote' or 'per_lora'")
    parser.add_argument("--save_every", type=int, help="Iters to save checkpoint, 0 = disabled")

    ## Other miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_seed", type=int, help="Random seed for data")
    parser.add_argument("--replicate", type=str, default="0", help="Unique identifier for run")
    parser.add_argument("--verbose", action="store_true", default=False, help="Extra details")
    parser.add_argument("--streamer", action="store_true", help="Stream generations (for PyTorch)")
    parser.add_argument("--debug", action="store_true", default=None, help="Debug mode")

    args = parser.parse_args()

    # Get run (i.e., experiment) name
    _ignore_args = [
        "base_url",
        "checkpoint_path",
        "log_path",
        "load_checkpoint_path",
        "project_name",
        "verbose",
        "streamer",
    ]
    _ignore_args.extend(
        [argn for argn in vars(args) if argn.endswith("_every") and argn != "eval_every"]
    )
    args.run_name = get_run_name(args, prefix=args.project_name, ignore_args=_ignore_args)
    logger.info("Run name: %s", args.run_name)

    # Setup log path and checkpoint / data-saving path
    # -> construct as args.log_path/args.env_config/model_name/args.run_name/
    # -> similar for checkpointing
    created_dir = False
    try:
        _model_name = OmegaConf.load(f"./configs/trainer/{args.trainer_config}.yaml")["model_name"]
    except Exception as trainer_config_error:
        logger.warning("%s: %s", trainer_config_error.__class__.__name__, trainer_config_error)

        try:
            _model_name = OmegaConf.load(f"./configs/model/{args.model_config}.yaml")[
                "model_config"
            ]["pretrained_model_name_or_path"]
        except Exception as model_config_error:
            logger.warning("%s: %s", model_config_error.__class__.__name__, model_config_error)
            assert args.model_name, "args.model_name must be specified if not in trainer_config"
            _model_name = args.model_name

    _model_name = args.model_name or _model_name
    _model_name = _model_name.split("/")[-1].replace("-", "_")
    _env_config = args.env_config.replace("/", "_")

    for argname in ["log_path", "checkpoint_path"]:
        for new_dir in [_env_config, _model_name, args.run_name]:
            # setattr(args, argname, os.path.join(args.log_path, new_dir))
            try:
                setattr(args, argname, os.path.join(getattr(args, argname), new_dir))
            except Exception as e:
                _error_class = e.__class__.__name__
                print(f"{_error_class}: {e}")
                # setattr(args, argname, os.path.join(args.log_path, new_dir))
                pass  # breakpoint removed
            if not os.path.exists(getattr(args, argname)):
                os.makedirs(getattr(args, argname))
                created_dir = True
        if created_dir:
            logger.info("Created %s at: %s", argname, getattr(args, argname))
        else:
            logger.info("Using %s at: %s", argname, getattr(args, argname))

    # So Tinker doesn't load, delete checkpoints.jsonl at args.log_path if it exists
    if not args.resume_run and os.path.exists(os.path.join(args.log_path, "checkpoints.jsonl")):
        os.remove(os.path.join(args.log_path, "checkpoints.jsonl"))
        logger.info(
            "Deleted checkpoints.jsonl at: %s", os.path.join(args.log_path, "checkpoints.jsonl")
        )

    # Setup tinker-cookbook WandB logging
    args.wandb_project = args.project_name
    args.wandb_name = args.run_name

    return args
