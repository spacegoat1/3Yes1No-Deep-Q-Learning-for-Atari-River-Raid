import gym
import time
import numpy as np
from collections import deque
from gym import spaces
import numpy as np
import time
import tensorflow as tf


class ConcatObs(gym.Wrapper):
    def __init__(self, env, k, DEATH_REWARD=0):
        """
        Wrapper to concatenate the last 4 observations from the environment. 
        DEATH_REWARD is a reward to penalize death, so that the model is encouraged to 'stay alive'
            Suggested value: DEATH_REWARD=-1000
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.DEATH_REWARD = DEATH_REWARD
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = \
            spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):

        # Take k steps at once
        total_reward = 0.0
        done = False
        for i in range(self.k):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.frames.append(obs)
            
            # only count one live each episode
            if info['lives'] < 4:
                total_reward = total_reward - self.DEATH_REWARD
                done = True
                break

        return self._get_ob(), total_reward, done, info

    def _get_ob(self):
        return np.array(self.frames)


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, GRAYSCALE=False, NORMALIZE=False):
        """
        Wrapper to preprocess observations. 
        Supply GRAYSCALE=True to convert observations to grayscale
        Supply NORMALIZE=True to scale obs values to [0,1]
        """
        self.GRAYSCALE = GRAYSCALE
        self.NORMALIZE = NORMALIZE
        super().__init__(env)
    
    def observation(self, obs):
        # Normalise observation by 255
        if self.NORMALIZE:
            obs = obs / 255.0
            
        if self.GRAYSCALE:
            obs = tf.image.rgb_to_grayscale(obs)
                    
        image = obs[:,2:-9,8:,:]
        image = tf.image.resize(image,[84,84])
        image = tf.transpose(tf.reshape(image, image.shape[:-1]),perm = [1,2,0])
        return image


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, CLIP=False, SCALE=False):
        """
        Reward wrapper to adjust rewards returned by environment. 
        Supply CLIP=False to change negative rewards to -1 and positive rewards to +1, as implemented in DeepMind paper
        Supply SCALE=False to use custom reward function for RiverRaid
        """
        self.CLIP = CLIP
        self.SCALE = SCALE
        self.reward_dict = {
            30: 0.1,
            60: 0.15,
            80: 0.2,
            100: 0.3,
            500: 0.5,
            -1000: -1,
        }
        super().__init__(env)
    
    def reward(self, reward):
        # Clip reward between 0 to 1
        if self.CLIP:
            # reward = np.clip(reward, 0, 1)
            reward = np.sign(reward)
        if self.SCALE:
            # Creating custom function (dict) to scale rewards
            # Scores based on http://www.atarimania.com/game-atari-2600-vcs-river-raid_s6826.html (Check instructions)
            # tanker: 30
            # helicopter: 60
            # fuel depot: 80
            # jet: 100
            # bridge: 500
            # death: SET in ConcatObs (by default, 0. Currently using 1000)
            reward = self.reward_dict.get(reward, 0)
        return reward


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        """
        Action wrapper, currently not modifying actions. 
        """
        super().__init__(env)
    
    def action(self, action):
        return action


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)
