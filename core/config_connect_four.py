# Hyperparameters for AlphaZero ConnectFour


# NEURAL NETWORK PARAMETERS
n_training_steps_per_episode = 20  # SGD steps per episode.

# MCTS PARAMETERS
n_simulated_games = 40
mcts_sims_per_move = 100
c_puct = 2.0  # Exploration-exploitation trade-off. (See discussion in mcts_scoring.)
add_dirichlet_noise = False
dir_epsilon = 0.8
dir_noise = 0.5

# EXPERIENCE REPLAY PARAMETERS
batch_size = 32
replay_buffer_max_size = 500
