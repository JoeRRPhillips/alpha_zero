import numpy as np

from .replay_buffer import combine_trajectories


class Trainer():
    '''
    Trainer class used to perform training and validation steps in a given environment.

    Args:
        model (tf.keras.Model): ResNet with Actor and Critic Heads for policy and value function approximations.
        replay_buffer (ReplayBuffer): stores episode trajectories.
        batch_size (int): size of sample from replay buffer.
        n_training_steps_per_episode (int): number of SGD updates per game played.
    '''
    def __init__(self, model, replay_buffer, batch_size, n_training_steps_per_episode):
        self.model = model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.n_training_steps_per_episode = n_training_steps_per_episode

    def simulate_game(self, env, player_1, player_2, p1_trajectory_tracker, p2_trajectory_tracker):
        '''
        Simulates game using self play.
        Stores game outcome in replay buffer.
        '''
        game_state = env.reset()

        players = {0: player_1, 1: player_2}
        toggle = 0 ^ 1
        current_player = 1  # The 1st toggle will cause player_1 to start

        p1_trajectory_tracker.begin_episode()
        p2_trajectory_tracker.begin_episode()

        is_game_over = False

        # Play full games to generate experience
        while not is_game_over:
            # Toggle player - note AZ receives a representation of this in the state tensor.
            current_player ^= toggle
            action = players[current_player].select_action(game_state, env, current_player)
            game_state, reward, is_game_over = env.step(game_state, action, current_player)

        if players[current_player] == player_1:
            p1_trajectory_tracker.complete_episode(reward)
            p2_trajectory_tracker.complete_episode(-1*reward)
        else:
            p1_trajectory_tracker.complete_episode(-1*reward)
            p2_trajectory_tracker.complete_episode(reward)

        states, visit_counts, rewards = combine_trajectories([p1_trajectory_tracker, p2_trajectory_tracker])

        # Add states, visit_counts, rewards to replay_buffer
        self.replay_buffer.append_trajectory(states, visit_counts, rewards)

    
    def train_networks(self):
        '''
        Train policy head towards outcome of MCTS & value head towards MC rewards
        '''
        # SGD steps
        for step in range(self.n_training_steps_per_episode):
            print(f'Step {step + 1}/{self.n_training_steps_per_episode}')
            
            # 1. Sample de-correlated batches of experiences from buffer
            states, visit_counts, rewards = self.replay_buffer.sample_batch(self.batch_size)
            
            # 2. Normalize visit_counts
            visit_sums = np.sum(visit_counts, axis=1).reshape((self.batch_size, 1))
            mcts_policy_labels = visit_counts / visit_sums

            # 3. Update model parameters. Prefer fir over train_on_batch for logging.
            self.model.fit(states, [mcts_policy_labels, np.expand_dims(rewards, -1)], epochs=1)

        # 4. Save model
        self.model.save('saved_models/alpha_zero')
