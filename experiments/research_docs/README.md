# Research Documentation: Truncation vs Termination in PPO

This folder contains research documentation for the investigation of proper truncation handling in Proximal Policy Optimization (PPO).

## Key Finding from Literature Review

**The problem we're investigating has already been solved!** 

Pardo et al. (2018) published "Time Limits in Reinforcement Learning" at ICML 2018, which:
1. Formally identified the truncation vs termination problem
2. Proposed **Partial-Episode Bootstrapping (PEB)** as the solution
3. Demonstrated improvements on MuJoCo tasks

The paper has 248+ citations and influenced modern RL libraries like Tianshou and the Gymnasium API design.

## Files in this Directory

| File | Description |
|------|-------------|
| `research_overview.md` | Context, problem statement, hypotheses, experimental setup, and results |
| `literature_review.tex` | LaTeX literature review document |
| `references.bib` | BibTeX bibliography with 18 key references |
| `README.md` | This file |

## Compiling the Literature Review

To compile the LaTeX document:

```bash
cd experiments/research_docs
pdflatex literature_review.tex
bibtex literature_review
pdflatex literature_review.tex
pdflatex literature_review.tex
```

Or using latexmk:
```bash
latexmk -pdf literature_review.tex
```

## Key References

1. **Pardo et al. (2018)** - "Time Limits in Reinforcement Learning" - ICML 2018
   - Directly addresses our research question
   - Introduces Partial-Episode Bootstrapping (PEB)
   - [Paper link](http://proceedings.mlr.press/v80/pardo18a.html)

2. **Towers et al. (2024)** - "Gymnasium: A Standard Interface for Reinforcement Learning Environments"
   - Introduces `terminated`/`truncated` API distinction
   - Provides `final_observation` for correct bootstrapping

3. **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
   - Original PPO paper (34,843+ citations)
   - Does not address truncation handling

4. **Schulman et al. (2015)** - "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
   - Introduces GAE (5,623+ citations)
   - Foundation for advantage computation in PPO

## Research Status

Given that the solution already exists in the literature, our experimental contribution is:

1. **Replication**: Verify PEB improvements with a modern PPO implementation
2. **Benchmark expansion**: Test on both classic control and continuous-control environments
3. **Implementation guide**: Provide clean, well-documented implementations in CleanRL style for both discrete (`ppo_correct_truncation.py`) and continuous action spaces (`ppo_continuous_action_correct_truncation.py`)

## Summary of Results

Experiments were run across 5 environments (2 discrete, 3 continuous) with 5 seeds each. Final performance (mean ± std, last 10% of training):

| Environment | PPO (Standard) | PPO (Correct Truncation) |
|-------------|---------------|--------------------------|
| CartPole-v1 | 489.9 ± 5.6 | **493.9 ± 4.4** |
| Acrobot-v1 | -84.8 ± 2.1 | -84.7 ± 1.5 |
| MountainCarContinuous-v0 | **94.1 ± 0.1** | 75.0 ± 42.5 |
| HalfCheetah-v4 | 915.5 ± 91.9 | **1009.2 ± 156.3** |
| Humanoid-v4 | 527.7 ± 46.3 | 527.7 ± 46.3 |

The correction provides clear benefit in environments where every episode ends by truncation (HalfCheetah) and modest benefit where truncation is common and the value function is well-calibrated (CartPole). It shows negligible effect where termination dominates (Acrobot) or training is far from convergence (Humanoid). Notably, it hurts performance in MountainCarContinuous, a sparse-reward setting where value function estimates at the truncation boundary are noisy during training.

Full analysis and per-environment discussion are in `research_overview.md`.

## Implications

Since this problem has a known solution:
- Our `ppo_correct_truncation.py` and `ppo_continuous_action_correct_truncation.py` implement the established PEB technique
- The experiments serve as verification and educational demonstration across a broader benchmark suite
- Results partially align with Pardo et al.'s findings but also highlight that the correction is not universally beneficial, particularly in sparse-reward settings where the value function quality at episode boundaries matters
