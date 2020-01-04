from collections import deque
import numpy as np
import random


def combine_trajectories(trajectory_trackers):
    states = np.concatenate([np.array(t.states) for t in trajectory_trackers])
    visit_counts = np.concatenate([np.array(t.visit_counts) for t in trajectory_trackers])
    rewards = np.concatenate([np.array(t.rewards) for t in trajectory_trackers])

    return states, visit_counts, rewards


class TrajectoryTracker:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []


class ReplayBuffer:
    '''
    ReplayBuffer class for training an off-policy Reinforcement Learning model.
    '''
    def __init__(self, max_size, random_seed=42):
        '''
        Args:
          max_size (int): maximum number of tuples that the buffer can store
        '''
        self.n_experiences = 0
        self.max_size = max_size

        # Initialize replay buffer as an empty deque -> O(1) left & right pop & appends.
        self.replay_buffer = deque()

        random.seed(random_seed)

    def append_trajectory(self, states, visit_counts, rewards):
        '''
        Adds data to the ReplayBuffer as an experience tuple
        '''
        # 1. If the buffer is full, remove the oldest entry (FIFO) before adding
        # the next experience. In this case, n_experiences remains unchanged.
        if self.n_experiences >= self.max_size:
            self.replay_buffer.popleft()

        # 2. Otherwise there is space in the buffer to add next experience and
        # increment n_experiences.
        else:
            self.n_experiences += 1

        # 3. Either way, can now add experience tuple
        # TODO - accelerate with mappings to append in one-shot
        n_experiences = states.shape[0]
        for i in range(n_experiences):
            experience = (states[i], visit_counts[i], rewards[i])
            self.replay_buffer.append(experience)

    def sample_batch(self, batch_size):
        '''
        Samples a mini_batch of size batch_size.
        Args:
          batch_size: batch size number
        Returns:
          mini batch of batch_size tuples
        '''
        # 1. Sample the min of batch_size and n_experiences from the buffer.
        samples = random.sample(self.replay_buffer, min(batch_size, self.n_experiences))

        # TODO - speed up in new implementation
        states = []
        visit_counts = []
        rewards = []
        for sample in samples:
            states.append(sample[0])
            visit_counts.append(sample[1])
            rewards.append(sample[2])

        states = np.array(states)
        visit_counts = np.array(visit_counts)
        rewards = np.array(rewards)
        
        return states, visit_counts, rewards

    def size(self):
        return self.n_experiences

    def reset(self):
        self.n_experiences = 0
        self.replay_buffer.clear()
