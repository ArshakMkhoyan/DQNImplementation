from collections import deque
import numpy as np


class ExperienceBuffer:
    """
    Replay buffer to collect the observations from the playing the game.
    """

    def __init__(self, size=100, n_frames=4):
        self.maxsize = size
        self.states = deque([])
        self.actions = deque([])
        self.done = deque([])
        self.rewards = deque([])
        self.n_frames = n_frames
        self.sum_rewards = 0

    def __len__(self):
        return len(self.states)

    @staticmethod
    def reward_clipping(r):
        if r > 0:
            return 1
        elif r == 0:
            return 0
        else:
            return -1

    def add(self, s, a, r, done):
        r = self.reward_clipping(r)
        if len(self.states) == self.maxsize:
            self.done.popleft()
            self.states.popleft()
            self.actions.popleft()
            rew = self.rewards.popleft()
            self.sum_rewards -= rew
        self.done.append(done)
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.sum_rewards += r

    def _get_valid_indexes(self, batch_size=32):
        indexes = []
        for _ in range(batch_size):
            while True:
                ind = np.random.randint(self.n_frames, len(self.states) - 1)
                if any([self.done[index] for index in range(ind - self.n_frames + 1, ind + 1)]):
                    continue
                else:
                    break
            indexes.append(ind)
        return indexes

    def sample(self, batch_size=32):
        """
        Sample batch from buffer
        observations with higher rewards
        :param batch_size: size of the batch
        :return:
        """
        indexes = self._get_valid_indexes(batch_size=batch_size)
        states = [np.stack([self.states[index] for index in range(ind - self.n_frames, ind)]) for ind in indexes]
        actions = [self.actions[ind] for ind in indexes]
        rewards = [self.rewards[ind] for ind in indexes]
        next_ss = [np.stack([self.states[index] for index in range(ind - self.n_frames + 1, ind + 1)]) for ind in
                   indexes]
        dones = [self.done[ind] for ind in indexes]
        return np.array(states).transpose((0, 2, 3, 1)), np.array(actions), np.array(rewards), np.array(
            next_ss).transpose((0, 2, 3, 1)), np.array(dones)
