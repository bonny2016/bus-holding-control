import gym
from torch import preserve_format
from ptan.agent import DQNAgent
"""
This class is the special environment wrapper, that help to truncate first N steps. 
because environment is set fresh after reset() is called, i.e no running buses, no waiting passengers. 
this empty state is not so valid to begin with. We truncate the first N steps so we treat the first N steps as staging steps, 
and experince start from the N+1 step. see class SkipFirstNExperienceSource in file experience.py
"""
class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, skip_first_n_steps=0, max_episode_steps=None):
        super(CustomEnvWrapper, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self._skip_first_n_steps = skip_first_n_steps
        self._to_skip = 0

    def step(self, action):
        """
        proceed one step according to action. In order to truncate first n steps, 
        the first n(self._to_skip) observations are labeld as 'SkipFirstN.truncated' in returned info value 
        """
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        if self._to_skip > 0:
            self._to_skip -= 1
            info['SkipFirstN.truncated'] = True
        else:
            info['SkipFirstN.truncated'] = False
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        random = (self._skip_first_n_steps>0)
        obs = self.env.reset(pre_steps=self._skip_first_n_steps, random=random)
        # self.env.pre_steps maybe different from above
        self._to_skip = self.env.pre_steps 
        return obs
        
# class CustomedDQNAgent(DQNAgent):
#     def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=DQNAgent.default_states_preprocessor):
#         self.min_headway = 3
#         self.max_headway = 12
#         super.__init__(dqn_model, action_selector, device, preprocessor)
    
#     def __call__(self, states, agent_states=None):
#         headway = states[-1,0]
#         if headway < self.min_headway:
#             return 
#         if states[-1,0] > self.min_headway and states[-1,0] < self.max_headway:
#             return super.__call__(states, agent_states)
