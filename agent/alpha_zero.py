import tensorflow as tf
from .resnet import ResNet


class ActorHead(tf.keras.layers.Layer):
    '''
    Implements the actor head of an actor-critic neural network for policy function approximation.

    Args:
        action_size (int): size of action space.
        hidden_size (int): size of latent space.
    Returns:
        (tf.Tensor): policy distribution over available actions in action space.
    '''
    def __init__(self, action_size, hidden_size, n_filters=2, name='policy_head'):
        super(ActorHead, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(units=hidden_size)
        self.dense2 = tf.keras.layers.Dense(units=action_size)  # name='policy_head'
        
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = tf.keras.activations.relu(x)
        x = self.dense2(x)

        # Softmax activation for valid probability distribution in policy.
        return tf.keras.activations.softmax(x)
    
    
class CriticHead(tf.keras.layers.Layer):
    '''
    Implements the critic head of an actor-critic neural network for value function approximation.

    Args:
        hidden_size (int): size of latent space.
    Returns:
        (tf.Tensor): evaluation of current state.
    '''
    def __init__(self, hidden_size, n_filters=1, name='value_head'):
        super(CriticHead, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(units=hidden_size)
        self.dense2 = tf.keras.layers.Dense(units=1)  # name='value_head'
        
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = tf.keras.activations.relu(x)
        x = self.dense2(x)
        
        # Tanh activation for positive and negative score evaluations.
        return tf.keras.activations.tanh(x)


class AlphaZero(tf.keras.Model):
    def __init__(self, action_size, name='alpha_zero', lr=3e-4):
        super(AlphaZero, self).__init__(name=name)
        self.lr = lr
        self.resnet = ResNet()
        self.actor_head = ActorHead(action_size, hidden_size=10, name='policy_head')
        self.critic_head = CriticHead(hidden_size=10, name='value_head')
        
    def call(self, state):
        x = self.resnet(state)
        pi = self.actor_head(x)
        v = self.critic_head(x)
        return pi, v

    def optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999)


'''
EXAMPLE USAGE:
alpha_zero = AlphaZero()
alpha_zero.compile(loss={'output_1': tf.keras.losses.SparseCategoricalCrossentropy(),
                    'output_2': tf.keras.losses.MeanSquaredError()},
            optimizer=agent.optimizer(),
            loss_weights={'output_1': 0.5, 'output_2': 0.5}
            )
b = 6
data = np.zeros([b, 14, 14, 1], dtype=float)
target_actor = np.ones([b, 1], dtype=int)
target_critic = np.ones([b, 1], dtype=float)
alpha_zero.fit(data,
          [target_actor, target_critic],
          epochs=1)
'''
