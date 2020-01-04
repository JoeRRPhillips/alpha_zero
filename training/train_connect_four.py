import tensorflow as tf

from agent.alpha_zero import AlphaZero
from core import config_connect_four
from core.mcts import MCTS
from core.replay_buffer import ReplayBuffer, TrajectoryTracker
from core.trainer import Trainer
from environments.connect_four_env import ConnectFourEnv, ConnectFourEnvEncoder


def train_connect_four(config):
    '''
    Runs full end-to-end training of AlphaZero in the ConnectFour environment.
    '''
    env = ConnectFourEnv()
    encoder = ConnectFourEnvEncoder()

    alpha_zero = AlphaZero(action_size=encoder.n_actions())

    # TODO - make subclassed model accept custom output names.
    # For now: 'output_1' <--> 'policy_head', 'output_2' <--> 'value_head'.
    # alpha_zero.compile(loss={'output_1': tf.keras.losses.SparseCategoricalCrossentropy(),
    alpha_zero.compile(loss={'output_1': tf.keras.losses.CategoricalCrossentropy(),
                        'output_2': tf.keras.losses.MeanSquaredError()},
                optimizer=alpha_zero.optimizer(),
                loss_weights={'output_1': 0.5, 'output_2': 0.5}
                )
    
    # For tracking trajectories of each player during a game.
    p1_trajectory_tracker = TrajectoryTracker()
    p2_trajectory_tracker = TrajectoryTracker()

    # Note there are 2 players for convenience, without duplication - only 1 MCTS runs at a time.
    # Still only 1 neural network.
    player_1 = MCTS(alpha_zero, env, encoder, p1_trajectory_tracker, config.mcts_sims_per_move, config.c_puct)
    player_2 = MCTS(alpha_zero, env, encoder, p2_trajectory_tracker, config.mcts_sims_per_move, config.c_puct)

    replay_buffer = ReplayBuffer(config.replay_buffer_max_size)
    trainer = Trainer(alpha_zero, replay_buffer, config.batch_size, config.n_training_steps_per_episode)

    for game in range(config.n_simulated_games):
        print(f'Starting Game {game+1} of {config.n_simulated_games}')
        trainer.simulate_game(env, player_1, player_2, p1_trajectory_tracker, p2_trajectory_tracker)
        
        if trainer.replay_buffer.size() >= config.batch_size:
            trainer.train_networks()

    print('Training Complete')

if __name__ == '__main__':
    train_connect_four(config_connect_four)
