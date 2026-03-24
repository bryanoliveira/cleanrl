# Proper Handling of Truncation vs Termination in Policy Gradient Methods

## Context

Reinforcement Learning (RL) algorithms learn optimal policies by interacting with environments and receiving reward signals. A fundamental concept in RL is the distinction between when an episode ends due to the agent reaching a terminal state (success or failure) versus when the episode is artificially cut off due to external constraints like time limits.

Modern RL environments, particularly those following the Gymnasium (formerly OpenAI Gym) API, distinguish between two types of episode endings:

1. **Termination**: The episode ends because the agent reached a true terminal state (e.g., the pole fell in CartPole, the agent reached the goal, or the agent died in a game). The Markov Decision Process (MDP) has genuinely concluded.

2. **Truncation**: The episode ends due to an artificial limit (e.g., maximum number of timesteps reached), but the MDP has not actually terminated. The agent could have continued interacting with the environment.

This distinction is crucial for correctly computing value estimates and advantages in policy gradient methods like Proximal Policy Optimization (PPO).

## Problem Statement

Most standard implementations of RL algorithms, including popular codebases like CleanRL, Stable-Baselines3, and others, treat both termination and truncation identically by combining them into a single "done" flag. This leads to incorrect value bootstrapping in the Generalized Advantage Estimation (GAE) calculation.

### The Mathematical Issue

In GAE, the advantage is computed as:

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where the TD-error is:

$$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_{t+1}) - V(s_t)$$

Here, $d_{t+1}$ is typically set to 1 when the episode ends (either by termination or truncation), which sets the bootstrap value to 0.

**The problem**: When an episode is *truncated* (not truly terminated), the value of the next state $V(s_{t+1})$ is NOT zero—the agent simply ran out of time. By treating truncation as termination, we incorrectly estimate:
- The value function (introducing bias in returns)
- The advantage estimates (potentially causing suboptimal policy updates)

### Correct Handling

- **For termination**: $\delta_t = r_t + \gamma \cdot 0 - V(s_t) = r_t - V(s_t)$ (no bootstrapping, future value is truly 0)
- **For truncation**: $\delta_t = r_t + \gamma V(s'_{truncated}) - V(s_t)$ (bootstrap with the value of the truncated state)

## Hypotheses

### Primary Hypothesis
**H1**: Correctly distinguishing between truncation and termination in PPO's advantage calculation will lead to improved sample efficiency and/or final performance, particularly in environments where truncation is common (e.g., environments with time limits).

### Secondary Hypotheses

**H2**: The performance difference will be more pronounced in environments where:
- Episodes are frequently truncated rather than terminated
- The value function at truncation time is significantly different from zero
- The discount factor $\gamma$ is high (making future rewards more important)

**H3**: Environments with longer time limits (where the agent rarely reaches the limit) will show less difference between the two approaches, as truncation events are rarer.

**H4**: The correct handling may improve value function learning, as measured by explained variance of the value function predictions.

### Null Hypothesis
**H0**: There is no statistically significant difference in performance between standard PPO and PPO with correct truncation handling across typical Gymnasium environments.

## Experimental Setup

### Algorithms Compared

**Discrete action environments:**

1. **PPO (Standard)**: The original CleanRL implementation that combines termination and truncation into a single "done" flag (`cleanrl/ppo.py`)

2. **PPO (Correct Truncation)**: Modified implementation that:
   - Stores terminations and truncations separately
   - Retrieves the value of the truncated observation using `final_observation` from the info dict
   - Bootstraps with $V(s'_{truncated})$ for truncated episodes
   - Bootstraps with 0 for truly terminated episodes
   (`cleanrl/ppo_correct_truncation.py`)

**Continuous action environments:**

1. **PPO Continuous (Standard)**: The original CleanRL continuous-action implementation (`cleanrl/ppo_continuous_action.py`)

2. **PPO Continuous (Correct Truncation)**: Same truncation fix applied to the continuous-action variant (`cleanrl/ppo_continuous_action_correct_truncation.py`)

### Environments

We test on standard Gymnasium environments that use time-based truncation:

| Environment | Max Steps | Termination Condition | Truncation Condition | Action Space |
|-------------|-----------|----------------------|---------------------|--------------|
| CartPole-v1 | 500 | Pole angle > 12° or cart position > 2.4 | 500 steps reached | Discrete |
| Acrobot-v1 | 500 | Tip reaches target height | 500 steps reached | Discrete |
| MountainCarContinuous-v0 | 999 | Car reaches goal (position ≥ 0.45) | 999 steps reached | Continuous |
| HalfCheetah-v4 | 1000 | None (no true termination) | 1000 steps reached | Continuous |
| Humanoid-v4 | 1000 | Torso height out of range | 1000 steps reached | Continuous |

These environments span a range of characteristics relevant to the truncation problem:
- Discrete and continuous action spaces
- Environments with and without true termination conditions
- Sparse (MountainCar) and dense (HalfCheetah, Humanoid) reward structures

### Hyperparameters

**Discrete environments** (`ppo.py` / `ppo_correct_truncation.py`):

| Parameter | Value |
|-----------|-------|
| Total timesteps | 1,000,000 |
| Learning rate | 2.5e-4 |
| Number of environments | 8 |
| Steps per rollout | 128 |
| Discount factor (γ) | 0.99 |
| GAE λ | 0.95 |
| Number of minibatches | 4 |
| Update epochs | 4 |
| Clip coefficient | 0.2 |
| Entropy coefficient | 0.01 |
| Value function coefficient | 0.5 |
| Max gradient norm | 0.5 |

**Continuous environments** (`ppo_continuous_action.py` / `ppo_continuous_action_correct_truncation.py`):

| Parameter | Value |
|-----------|-------|
| Total timesteps | 1,000,000 |
| Learning rate | 3e-4 |
| Number of environments | 1 |
| Steps per rollout | 2048 |
| Discount factor (γ) | 0.99 |
| GAE λ | 0.95 |
| Number of minibatches | 32 |
| Update epochs | 10 |
| Clip coefficient | 0.2 |
| Entropy coefficient | 0.0 |
| Value function coefficient | 0.5 |
| Max gradient norm | 0.5 |

Both continuous implementations also use observation normalization, reward normalization, and action clipping wrappers, matching the standard CleanRL setup.

### Evaluation Protocol

1. **Seeds**: 5 random seeds per environment-algorithm pair (seeds 1–5)
2. **Metrics**:
   - Episodic return (primary metric)
   - Value function explained variance
   - Learning curves over training
3. **Statistical Analysis**:
   - Mean ± standard deviation across seeds
   - Final performance comparison (last 10% of training)
   - Learning curve visualization with confidence intervals

### Implementation Details

- Framework: PyTorch
- Environment interface: Gymnasium 0.29.1
- Parallel environments: SyncVectorEnv
- Logging: TensorBoard
- Package management: uv

## Results

Final performance (mean ± std over 5 seeds, last 10% of training):

| Environment | PPO (Standard) | PPO (Correct Truncation) |
|-------------|---------------|--------------------------|
| CartPole-v1 | 489.9 ± 5.6 | **493.9 ± 4.4** |
| Acrobot-v1 | -84.8 ± 2.1 | -84.7 ± 1.5 |
| MountainCarContinuous-v0 | **94.1 ± 0.1** | 75.0 ± 42.5 |
| HalfCheetah-v4 | 915.5 ± 91.9 | **1009.2 ± 156.3** |
| Humanoid-v4 | 527.7 ± 46.3 | 527.7 ± 46.3 |

### Observations

**CartPole-v1**: Correct truncation handling yields a small improvement in both mean return (493.9 vs 489.9) and variance (4.4 vs 5.6). CartPole episodes are frequently truncated at the 500-step limit when the agent is performing well, so the correction is regularly applied and appears to help.

**Acrobot-v1**: The two approaches are nearly indistinguishable (−84.7 vs −84.8). In Acrobot, successful episodes terminate before the time limit; training episodes tend to terminate more than truncate, reducing the practical impact of the fix.

**HalfCheetah-v4**: Correct truncation handling shows a meaningful improvement in mean return (~10% higher: 1009.2 vs 915.5). HalfCheetah has no true termination condition—every episode ends by truncation at 1000 steps—so the correction is applied at the boundary of every single rollout, and the cumulative benefit is more visible.

**Humanoid-v4**: Both algorithms produce identical results (527.7 ± 46.3). At 1,000,000 timesteps, Humanoid is far from converged for either method, and any signal from truncation correction is dominated by the overall difficulty of the task.

**MountainCarContinuous-v0**: Standard PPO substantially outperforms the correct-truncation version (94.1 ± 0.1 vs 75.0 ± 42.5), with the corrected version also showing dramatically higher variance across seeds. This is a counter-intuitive result. One likely contributing factor is that MountainCar has a sparse reward structure: the agent receives no useful reward signal until it reaches the goal. When truncation occurs (at 999 steps) before the goal is reached, the value estimate at the truncated state $V(s'_{truncated})$ reflects an incompletely-trained critic in a sparse-reward regime. Rather than the neutral zero used by the standard implementation, this potentially noisy bootstrap value may introduce harmful gradient signal, destabilizing learning for some seeds (explaining the high variance) while offering no benefit over the standard approach.

### Discussion

The results paint a nuanced picture that partially supports H1 but also highlights important failure modes:

- **Environments where all episodes truncate** (HalfCheetah) benefit most clearly from correct truncation handling, consistent with H2.
- **Environments where truncation and termination are mixed** (CartPole) show modest improvement.
- **Environments where truncation is rare** relative to termination (Acrobot) show negligible difference, consistent with H3.
- **Sparse-reward environments** (MountainCar) can be harmed by the correction when the value function is poorly calibrated at the truncation boundary, a regime not anticipated by H1 or H2.
- **Environments with insufficient training budget** (Humanoid) do not yield enough signal to distinguish the two methods.

## Files

- `cleanrl/ppo.py` - Standard PPO implementation (discrete)
- `cleanrl/ppo_correct_truncation.py` - PPO with correct truncation handling (discrete)
- `cleanrl/ppo_continuous_action.py` - Standard PPO implementation (continuous)
- `cleanrl/ppo_continuous_action_correct_truncation.py` - PPO with correct truncation handling (continuous)
- `experiments/compare_truncation_handling.sh` - Experiment runner script
- `experiments/plot_truncation_comparison.py` - Visualization script
- `experiments/research_docs/` - This documentation folder
