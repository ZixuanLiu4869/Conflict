# Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLMs Alignment

This repository contains the official implementation for our paper:

**"Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLMs Alignment"**

## Overview

Our work proposes a novel framework for detecting and mitigating conflicts between base policies and imperfect proxy rewards in LLM alignment. We introduce conflict-aware metrics (nRPCS and Kendall-Tau distance) and a selective human-in-the-loop algorithm for refining the reward model and improving alignment performance.

---

## Usage 
### 1. Supervised Fine-Tuning (SFT)
Train the base policy model using supervised fine-tuning:

```bash
python sft.py




### 2. Train the Proxy Reward Model
Train a proxy reward model on preference data:

```bash
python reward_trainer.py

### 3. Identify Conflict Examples
Sample responses and compute reward-policy conflict metrics (nRPCS and Kendall-Tau):

```bash
python sample.py


### 4. Train Refined Reward Model

Retrain the reward model using feedback collected on conflict examples:

```bash
python reward_trainer_refined.py

### 5. RM-Based Policy Fine-Tuning (PPO)
Train the refined base policy using PPO with the refined proxy reward model:

```bash
python ppo.py

### 6. Final Evaluation

Sample responses using the refined model and evaluate using a high-quality gold reward model:
```bash
python sample_gold_reward.py && python evaluation.py

