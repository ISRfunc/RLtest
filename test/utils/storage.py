from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Dict
from gym import spaces
import torch

class RollOut:

    def __init__(self, num_steps, num_envs, num_workers, observation_space, action_space):    
        
        self.obs = {}
        if isinstance(observation_space, spaces.Dict):
            for key, space in observation_space.spaces.items():
                self.obs[key] = torch.zeros(num_steps, num_envs, *space.shape)


        # assuming all actions are float 
        self.action_log_probs = torch.zeros(num_steps, num_envs)
        self.actions = torch.zeros((num_steps, num_envs, *action_space.shape),
                                   dtype=torch.float32) #important


        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps, num_envs, 1)
        self.returns = torch.zeros(num_steps, num_envs, 1)


        self.num_steps = num_steps
        self.step = 0
        self.workerCount = 0
        self.num_workers = num_workers



        """
        self.obs = torch.zeros(num_steps + 1, num_envs, *obs_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_envs,
                                      rec_state_size)
        self.rewards = torch.zeros(num_steps, num_envs)
        self.value_preds = torch.zeros(num_steps + 1, num_envs)
        self.returns = torch.zeros(num_steps + 1, num_envs)


        self.action_log_probs = torch.zeros(num_steps, num_envs)
        self.actions = torch.zeros((num_steps, num_envs, self.n_actions),
                                   dtype=action_type)
        

        self.num_steps = num_steps
        self.step = 0
        """

    def to(self, device):
        if isinstance(self.obs, Dict):
            for key in self.obs:
                self.obs[key] = self.obs[key].to(device)
        else:
            self.obs.to(device)
        
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
            
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        
        return self
    

    def insert(self, obs, actions, action_log_probs, value_preds, rewards):

        with torch.no_grad():

            if isinstance(self.obs, Dict):
                for key in self.obs:
                    self.obs[key][self.step][self.workerCount:self.workerCount + self.num_workers].copy_(obs[key])
            else:
                self.obs[self.step][self.workerCount:self.workerCount + self.num_workers].copy_(obs)
            
            self.actions[self.step][self.workerCount:self.workerCount + self.num_workers].copy_(actions)
            self.action_log_probs[self.step][self.workerCount:self.workerCount + self.num_workers].copy_(action_log_probs)


            self.value_preds[self.step][self.workerCount:self.workerCount + self.num_workers].copy_(value_preds)
            self.rewards[self.step][self.workerCount:self.workerCount + self.num_workers].copy_(rewards)


            self.step = (self.step + 1) % self.num_steps

            if self.step == 0:
                self.workerCount = (self.workerCount + 1) % self.num_workers

        del obs, actions, action_log_probs, value_preds, rewards

        torch.cuda.empty_cache()  


    def compute_returns_and_gae(self, next_value, use_gae, gamma, tau):

        with torch.no_grad():

            if use_gae:
                #next_value = next_value.to(device)
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    if step == self.rewards.size(0) - 1:
                        delta = self.rewards[step][self.workerCount:self.workerCount + self.num_workers] + gamma * next_value - self.value_preds[step][self.workerCount:self.workerCount + self.num_workers]
                    else:
                        delta = self.rewards[step][self.workerCount:self.workerCount + self.num_workers] + gamma * self.value_preds[step + 1][self.workerCount:self.workerCount + self.num_workers] - self.value_preds[step][self.workerCount:self.workerCount + self.num_workers]
                    gae = delta + gamma * tau * gae
                    self.returns[step][self.workerCount:self.workerCount + self.num_workers] = gae + self.value_preds[step][self.workerCount:self.workerCount + self.num_workers]
            else:
                #self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    if step == self.rewards.size(0) - 1:
                        self.returns[step][self.workerCount:self.workerCount + self.num_workers] = next_value * gamma + self.rewards[step][self.workerCount:self.workerCount + self.num_workers]
                    else:
                        self.returns[step][self.workerCount:self.workerCount + self.num_workers] = self.returns[step + 1][self.workerCount:self.workerCount + self.num_workers] * gamma + self.rewards[step][self.workerCount:self.workerCount + self.num_workers]

        del next_value

        torch.cuda.empty_cache()  


    def batchSample(self, advantages, mini_batch_size):

        num_steps, num_envs = self.rewards.size()[0:2]
        batch_size = num_envs * num_steps
        

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs': {f"{key}": self.obs[key].view(-1, *self.obs[key].size()[2:])[indices] for key in self.obs},
                'actions': self.actions.view(-1, *self.actions.size()[2:])[indices],
                'value_preds': self.value_preds.view(-1)[indices],
                'returns': self.returns.view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
            }


"""
# Storage
g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                  num_scenes, g_observation_space.shape,
                                  g_action_space, g_policy.rec_state_size,
                                  1).to(device)
"""
