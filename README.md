# Critic-Actor

## Setup

### Dependencies

To install dependencies and manage packages, we use `uv`. You can install it from [here](https://docs.astral.sh/uv/installation/).

Then, install dependencies with `uv sync`. Or, just run one of the example scripts below (`uv` will automatically install / update dependencies in `pyproject.toml` as needed).

#### `claude-agent-sdk`

Sometimes as a special case, we need to install `claude-agent-sdk` separately from the rest of the `uv` dependencies. We can do so via:

```bash
uv pip install claude-agent-sdk
```


### Tinker

We currently use [Tinker](https://thinkingmachines.ai/tinker/) to run experiments. You'll want to:  
1. Sign up for Tinker [here](https://auth.thinkingmachines.ai/sign-up)  
2. Create an API key from the [console](https://tinker-console.thinkingmachines.ai/)
3. Either export this as an environment variable (e.g., `export TINKER_API_KEY="<your_api_key>"`) or add it to a `.env` file (recommended, see below).

### Setting Environment Variables

To manage API keys for Tinker, WandB, and Hugging Face, we use `dotenv` to load environment variables from a `.env` file. 
Create a `.env` file in this project's root directory (e.g., `vim .env`), and add your environment variables, e.g.,

```markdown
TINKER_API_KEY="<your_tinker_api_key>"
HF_TOKEN="<your_huggingface_token>"
WANDB_API_KEY="<your_wandb_api_key>"
WANDB_ENTITY="hazy-research"
```

If you haven't already, add this `.env` file to your `.gitignore` file to avoid leaking keys and committing it to the repository.

## Sample Experiment Commands

### Snorkel Insurance

```bash
# Critic-Actor with Claude Haiku 4.5
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config cria_claude_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 8 \
--num_substeps 4 \
--seed 42 --replicate 0 --verbose

# Standard Policy Gradient
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config default \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 8 \
--num_substeps 4 \
--seed 42 --replicate 0 --verbose

# Critic-Actor with Claude Sonnet 4.6
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 8 \
--num_substeps 4 \
--seed 42 --replicate 0 --verbose
```

### Snorkel Finance

```bash
# Critic-Actor with Claude Haiku 4.5
uv run python main_tinker.py \
--env_config finqa/reasoning_haiku \
--generator_config cria_claude_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 8 \
--num_substeps 4 \
--seed 42 --replicate 0 --verbose

# Standard Policy Gradient
uv run python main_tinker.py \
--env_config finqa/reasoning_haiku \
--generator_config default \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 8 \
--num_substeps 4 \
--seed 42 --replicate 0 --verbose

# Critic-Actor with Claude Sonnet 4.6
uv run python main_tinker.py \
--env_config finqa/reasoning_haiku \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 8 \
--num_substeps 4 \
--seed 42 --replicate 0 --verbose
```


### Eval-only Runs

```bash
# Haiku Judge, Insurance
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config cria_claude_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose --max_turns 20

# Haiku Judge, Insurance
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose --max_turns 20

# Sonnet Judge, Insurance
uv run python main_tinker.py \
--env_config insurance/default \
--generator_config cria_claude_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose --max_turns 20

# Sonnet Judge, Insurance
uv run python main_tinker.py \
--env_config insurance/default \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose --max_turns 20

# Haiku Judge, Finance Reasoning
uv run python main_tinker.py \
--env_config finqa/reasoning_haiku \
--generator_config cria_claude_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose --max_turns 20

# Haiku Judge, Finance Reasoning
uv run python main_tinker.py \
--env_config finqa/reasoning_haiku \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose

# Sonnet Judge, Finance Reasoning
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config cria_claude_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose

# Sonnet Judge, Finance Reasoning
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 \
--seed 42 --replicate no_train --verbose --max_turns 20
```

