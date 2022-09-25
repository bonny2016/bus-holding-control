import numpy as np

import torch

from lib import environ

METRICS = (
    'episode_reward',
    'episode_steps',
)


def validation_run(env, net, episodes=1, epsilon=0, device="cpu"):
    stats = { metric: [] for metric in METRICS }

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        episode_steps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            #print(f"validation obs data:{obs_v.shape}")
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                print("random")
                action_idx = env.action_space.sample()

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if done:
                stats['episode_reward'].append(total_reward)
                stats['episode_steps'].append(episode_steps)
                break

    return { key: np.mean(vals) for key, vals in stats.items() }