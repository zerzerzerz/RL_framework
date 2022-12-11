import numpy as np
import torch


def evaluate(actor_critic, eval_envs, device, num_steps, num_processes=1):
    with torch.no_grad():
        episode_reward = 0
        obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)

        for step in range(num_steps):
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True
            )

            # Observe reward and next obs
            obs, reward, done, infos = eval_envs.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device
            )

            episode_reward += reward.mean().item()
            if done[0]:
                break
        eval_envs.close()
        
        return episode_reward