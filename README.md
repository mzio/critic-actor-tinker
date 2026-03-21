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

### Snorkel Environments

To setup the real-world multi-turn insurnace writing and finance reasoning environments, you'll need to clone their respective environment repos from our wonderful Snorkel AI friends.

One-liners for both are as follows:

#### Snorkel Insurance

```bash
mkdir -p data/snorkel_insurance && cd data/snorkel_insurance && git clone 
https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation _repo && cp -r
_repo/resources ./resources && cp -r _repo/tool_data ./tool_data && mkdir -p task_data && cp
_repo/task_data/downsampled_task_set.json task_data
```

#### Snorkel Finance (FinQA)

```bash
mkdir -p data/snorkel_finance && cd data/snorkel_finance && git clone
https://github.com/snorkel-ai/FinQABenchmark _repo && cp -r _repo/data/raw ./raw && mkdir -p benchmark && cp
_repo/data/benchmark/finqa.csv _repo/data/benchmark/finqa_reasoning.csv benchmark/
```



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
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 10 \
--seed 42 --replicate 1 --verbose

# Standard Policy Gradient
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config default \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 10 \
--seed 42 --replicate 1 --verbose

# Critic-Actor with Claude Sonnet 4.6
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 10 \
--seed 42 --replicate 1 --verbose

# Critic-Actor Chat with Claude Haiku 4.5
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config chat_claude_handoff_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 10 \
--seed 42 --replicate 1 --verbose

# Critic-Actor Chat with Claude Haiku 4.5, no Cria prompt (eval only)
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config chat_claude_handoff_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 --no_cria_prompt \
--eval_every 10 --max_turns 10 \
--seed 42 --replicate 1 --verbose 

# Critic-Actor Chat with Claude Haiku 4.5, no Cria prompt (eval only, but also run on train samples)
uv run python main_tinker.py \
--env_config insurance/default_haiku \
--generator_config chat_claude_handoff_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 40 --batch_size 8 --group_size 4 --num_substeps 1 --no_cria_prompt \
--eval_every 10 --max_turns 10 \
--seed 42 --replicate 1 --verbose 

```

### Snorkel Finance

```bash
# Critic-Actor with Claude Haiku 4.5
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config cria_claude_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 20 \
--seed 42 --replicate 1 --verbose

# Standard Policy Gradient
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config default \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 20 \
--seed 42 --replicate 1 --verbose

# Critic-Actor with Claude Sonnet 4.6
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config cria_claude_sonnet \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 20 \
--seed 42 --replicate 1 --verbose

# Critic-Actor Chat with Claude Haiku 4.5
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config chat_claude_handoff_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--batch_size 8 --group_size 4 \
--num_substeps 4 \
--eval_every 10 --max_turns 20 \
--seed 42 --replicate 1 --verbose

# Critic-Actor Chat with Claude Haiku 4.5, no Cria prompt (eval only)
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config chat_claude_handoff_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 1 --batch_size 1 --group_size 1 --num_substeps 1 --no_cria_prompt \
--eval_every 10 --max_turns 20 \
--seed 42 --replicate 1 --verbose 

# Critic-Actor Chat with Claude Haiku 4.5, no Cria prompt (eval only, but also run on train samples)
uv run python main_tinker.py \
--env_config finqa/reasoning \
--generator_config chat_claude_handoff_haiku \
--effort low \
--trainer_config pg \
--replay_buffer_config default \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--num_actions 1 --num_batches 40 --batch_size 8 --group_size 4 --num_substeps 1 --no_cria_prompt \
--eval_every 10 --max_turns 20 \
--seed 42 --replicate 1 --verbose 
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
--seed 42 --replicate no_train --verbose --max_turns 10

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
--seed 42 --replicate no_train --verbose --max_turns 10

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

