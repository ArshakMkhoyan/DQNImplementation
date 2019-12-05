import numpy as np
import cv2
import gym
from gym.core import ObservationWrapper
from gym.core import Wrapper
from gym.spaces.box import Box


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, img_size=(84, 84, 1), make_grey=True, crop=lambda x: x[30:203, 6:154, :]):
        super().__init__(env)
        self.img_size = img_size
        self.make_grey = make_grey
        self.crop = crop
        self.observation_space = Box(0.0, 1.0, img_size)

    def observation(self, observation):
        observation = self.crop(observation)
        if self.make_grey:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, self.img_size[:2])
        observation = np.expand_dims(observation, -1)
        observation = observation.astype('uint8')
        return observation


class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames):
        super().__init__(env)
        height, width, n_channels = env.observation_space.shape
        obs_shape = [height, width, n_channels * n_frames]
        self.framebuffer = np.zeros(obs_shape, dtype='uint8')
        self.axis = -1
        self.observation_space = Box(0.0, 1.0, self.framebuffer.shape)

    def reset(self):
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.framebuffer = self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        new_img, r, done, info = self.env.step(action)
        self.framebuffer = self.update_buffer(new_img)
        return self.framebuffer, r, done, info

    def update_buffer(self, obs):
        old_frames = self.framebuffer[:, :, 1:]
        return np.concatenate([old_frames, obs], axis=self.axis)


def make_env(game_name, img_size=(84, 84, 1), make_grey=True, crop=lambda x: x[30:203, 6:154, :], n_frames=4):
    env = gym.make(game_name)
    env = PreprocessAtari(env, img_size, make_grey, crop)
    env = FrameBuffer(env, n_frames=n_frames)
    return env
