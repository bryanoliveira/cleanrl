#!/bin/bash
# Experiment script to compare standard PPO vs PPO with correct truncation handling
# This script runs both implementations across multiple seeds and environments
# to evaluate the impact of proper termination/truncation distinction.
#
# The script automatically skips experiments that already have TensorBoard logs
# in the runs/ directory, so it's safe to re-run after partial completion.
#
# Usage: ./experiments/compare_truncation_handling.sh
#
# Requirements: uv must be installed (https://github.com/astral-sh/uv)

set -e

# Configuration
SEEDS=(1 2 3 4 5)

# Discrete action environments (use ppo.py / ppo_correct_truncation.py)
DISCRETE_ENVS=(
    "CartPole-v1"      # Truncates at 500 steps
    "Acrobot-v1"       # Truncates at 500 steps
)

# Continuous action environments (use ppo_continuous_action.py / ppo_continuous_action_correct_truncation.py)
CONTINUOUS_ENVS=(
    "MountainCarContinuous-v0"  # Truncates at 999 steps
    "HalfCheetah-v4"            # Truncates at 1000 steps
    "Humanoid-v4"               # Truncates at 1000 steps
)

DISCRETE_TOTAL_TIMESTEPS=1000000
CONTINUOUS_TOTAL_TIMESTEPS=1000000
DISCRETE_NUM_ENVS=8
CONTINUOUS_NUM_ENVS=8
TRACK=true  # Set to true to enable wandb tracking

# Output directory for results
RESULTS_DIR="experiments/results/truncation_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/experiment.log"

echo "=============================================" | tee "$LOG_FILE"
echo "PPO Truncation Handling Comparison Experiment" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Seeds: ${SEEDS[*]}" | tee -a "$LOG_FILE"
echo "Discrete environments: ${DISCRETE_ENVS[*]}" | tee -a "$LOG_FILE"
echo "Continuous environments: ${CONTINUOUS_ENVS[*]}" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if a run already exists in the runs/ directory
run_exists() {
    local env_id=$1
    local exp_name=$2
    local seed=$3
    # Look for directories matching {env_id}__{exp_name}__{seed}__*
    local pattern="runs/${env_id}__${exp_name}__${seed}__*"
    local matches=( $pattern )
    if [ -e "${matches[0]}" ]; then
        return 0  # exists
    fi
    return 1  # does not exist
}

# Function to run a single experiment (with skip logic)
run_experiment() {
    local script=$1
    local env_id=$2
    local seed=$3
    local exp_name=$4
    local total_timesteps=$5
    local num_envs=$6

    # Skip if run already exists
    if run_exists "$env_id" "$exp_name" "$seed"; then
        echo "[$(date +%H:%M:%S)] SKIP (already exists): $exp_name on $env_id with seed $seed" | tee -a "$LOG_FILE"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Running $exp_name on $env_id with seed $seed" | tee -a "$LOG_FILE"

    # Build the command
    cmd="uv run python cleanrl/$script \
        --env-id $env_id \
        --seed $seed \
        --total-timesteps $total_timesteps \
        --num-envs $num_envs \
        --exp-name $exp_name"

    if [ "$TRACK" = true ]; then
        cmd="$cmd --track --wandb-project-name ppo-truncation-comparison"
    fi

    # Run and capture output
    if eval $cmd >> "$RESULTS_DIR/${exp_name}_${env_id}_seed${seed}.log" 2>&1; then
        echo "[$(date +%H:%M:%S)] Completed: $exp_name on $env_id with seed $seed" | tee -a "$LOG_FILE"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $exp_name on $env_id with seed $seed" | tee -a "$LOG_FILE"
    fi
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:" | tee -a "$LOG_FILE"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" | tee -a "$LOG_FILE"
    exit 1
fi

# Sync dependencies (including mujoco extra for continuous-action envs)
echo "Installing/syncing dependencies with uv..." | tee -a "$LOG_FILE"
uv sync --extra mujoco 2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Store start time
START_TIME=$(date +%s)

# Run all experiments
echo "Starting experiments..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- Discrete action environments ---
for env_id in "${DISCRETE_ENVS[@]}"; do
    echo "===== Environment: $env_id (discrete) =====" | tee -a "$LOG_FILE"

    for seed in "${SEEDS[@]}"; do
        run_experiment "ppo.py" "$env_id" "$seed" "ppo_standard" "$DISCRETE_TOTAL_TIMESTEPS" "$DISCRETE_NUM_ENVS"
        run_experiment "ppo_correct_truncation.py" "$env_id" "$seed" "ppo_correct_truncation" "$DISCRETE_TOTAL_TIMESTEPS" "$DISCRETE_NUM_ENVS"
    done

    echo "" | tee -a "$LOG_FILE"
done

# --- Continuous action environments ---
for env_id in "${CONTINUOUS_ENVS[@]}"; do
    echo "===== Environment: $env_id (continuous) =====" | tee -a "$LOG_FILE"

    for seed in "${SEEDS[@]}"; do
        run_experiment "ppo_continuous_action.py" "$env_id" "$seed" "ppo_continuous_action" "$CONTINUOUS_TOTAL_TIMESTEPS" "$CONTINUOUS_NUM_ENVS"
        run_experiment "ppo_continuous_action_correct_truncation.py" "$env_id" "$seed" "ppo_continuous_action_correct_truncation" "$CONTINUOUS_TOTAL_TIMESTEPS" "$CONTINUOUS_NUM_ENVS"
    done

    echo "" | tee -a "$LOG_FILE"
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "=============================================" | tee -a "$LOG_FILE"
echo "Experiment completed!" | tee -a "$LOG_FILE"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "Results saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "TensorBoard logs saved to: runs/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To view results with TensorBoard:" | tee -a "$LOG_FILE"
echo "  uv run tensorboard --logdir runs/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To generate plots:" | tee -a "$LOG_FILE"
echo "  uv run python experiments/plot_truncation_comparison.py --logdir runs/" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
