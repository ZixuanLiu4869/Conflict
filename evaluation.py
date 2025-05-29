import json
from scipy.stats import kendalltau
import numpy as np


new_dataset_path = "evaluation_1.5.json"


# === Step 1: Load JSON ===
with open(new_dataset_path, "r") as f:
    new_examples = json.load(f)

# === Step 2: Compute nRPCS and Kendall Tau ===
nRPCS_per_prompt = []
kendall_tau_list = []
reward_list = []


for i, (logprobs, rewards) in enumerate(zip(new_examples["response_prob"], new_examples["response_reward"])):
    reward_list.append(np.mean(rewards))
    logprobs = np.array(logprobs)
    rewards = np.array(rewards)
    

    # Skip degenerate entries
    if len(logprobs) < 2 or np.std(logprobs) < 1e-6 or np.std(rewards) < 1e-6:
        nRPCS_per_prompt.append([0.0] * len(logprobs))
        kendall_tau_list.append(0.0)
        continue

    # Normalize
    logprobs_norm = (logprobs - logprobs.mean()) / (logprobs.std() + 1e-8)
    rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Pointwise conflict score
    nRPCS = rewards_norm - logprobs_norm
    nRPCS_list = nRPCS.tolist()
    nRPCS_per_prompt.append(nRPCS_list)

    # Ranking disagreement
    tau, _ = kendalltau(logprobs, rewards)
    kendall_tau_list.append(tau if not np.isnan(tau) else 0.0)



# === Step 3: Aggregate scores ===
flat_nRPCS = [abs(val) for sublist in nRPCS_per_prompt for val in sublist]
mean_nRPCS = np.mean(flat_nRPCS)
mean_kendall_tau = np.mean(kendall_tau_list)
mean_reward = np.mean(reward_list)

print(f"✅ Mean abs nRPCS: {mean_nRPCS:.4f}")
print(f"✅ Mean Kendall-Tau: {mean_kendall_tau:.4f}")
print(f"✅ Mean Reward: {mean_reward:.4f}")
