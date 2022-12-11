import gym
import torch
from einops import rearrange, reduce, repeat


class MyBaseEnv():
    def __init__(self, observation_space, action_space, device) -> None:
        assert issubclass(observation_space.__class__, gym.Space)
        assert issubclass(action_space.__class__, gym.Space)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError
    

class MyEnv(MyBaseEnv):
    def __init__(self, observation_space, action_space, device) -> None:
        # super().__init__(observation_space, action_space, device)
        # self.env = gym.make('Acrobot-v1')
        self.env = gym.make('LunarLander-v2')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        super().__init__(self.observation_space, self.action_space, device)

    def reset(self):
        '''[...] -> [1,...]'''
        obs, info = self.env.reset()
        obs = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)
        obs.unsqueeze_(0)
        return obs

    def step(self, action):
        '''[1,...] -> [...] -> [1,...]'''
        with torch.no_grad():
            action.squeeze(0)
            if self.action_space.__class__.__name__ == "Box":
                action = action.detach().cpu().numpy()
            elif self.action_space.__class__.__name__ == "Discrete":
                action = action.item()
            else:
                raise NotImplementedError
            
            next_state, reward, terminated, truncated, info = self.env.step(action)

            next_state = torch.from_numpy(next_state).to(dtype=torch.float32, device=self.device).unsqueeze(0)
            reward = torch.FloatTensor([[reward]]).to(device=self.device)
            terminated = [terminated]
            info = [info]
            return next_state, reward, terminated, info
    
    def close(self):
        self.env.close()