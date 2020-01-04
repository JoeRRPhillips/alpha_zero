from copy import copy
import numpy as np


class Edge:
    '''
    Data structure for tracking edge statistics
    for each action from a node.
    '''
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class Node:
    '''Class for defining a node on a search tree for MCTS.'''
    def __init__(self, state, value, priors, parent, env):
        self.state = state
        self.value = value
        self.parent = parent
        self.total_visit_count = 1
        self.edges = {}
        for action, p in priors.items():
            if env.is_action_available(state, action):
                self.edges[action] = Edge(p)
        self.children = {}

    def actions(self):
        return self.edges.keys()

    def add_child(self, action, child_node):
        self.children[action] = child_node

    def has_child(self, action):
        return action in self.children

    def get_child(self, action):
        return self.children[action]

    def update_statistics(self, action, value):
        self.total_visit_count += 1
        self.edges[action].visit_count += 1
        self.edges[action].total_value += value

    def expected_value(self, action):
        edge = self.edges[action]
        if edge.visit_count == 0:
            return 0.0
        return edge.total_value / edge.visit_count

    def prior(self, action):
        return self.edges[action].prior

    def visit_count(self, action):
        if action in self.edges:
            return self.edges[action].visit_count
        return 0


class MCTS():
    '''
    Implements the MCTS algorithm for AGZ using ResNet function approximators in place of rollout.
    '''
    def __init__(self, model, env, encoder, trajectory_tracker, mcts_sims_per_move, c_puct):
        self.model = model
        self.env = env
        self.encoder = encoder

        self.trajectory_tracker = trajectory_tracker

        self.mcts_sims_per_move = mcts_sims_per_move
        self.c_puct = c_puct

    def select_action(self, root_state, env, current_player):
        '''
        Runs mcts_sims_per_move simulations in a simulated copy of the environment.
        AZ neural network guides the search using policy predictions as a prior.
        Choses next action according to maximum PUCT score across the nodes visited
        in the simulated trajectory from the current node (root_state).

        Args:
          root_state (np.array): current state of the game, from which simulations are run.
          env (Environment): current state of the full game environment.
          current_player (int): used to encode whose turn it is directly in the
                                input state-vectors at each step of simulated play.
                                I.e. allows AZ to know turns. As per AZ paper.
        Returns:
          action: next action to be played by the current player in the real (not simulated) env
        '''

        # Make a temporary environment for the simulations. TODO - remove need for this
        env_sim = copy(env)
        root = self.create_node(root_state, action=None, parent=None, env=env_sim)

        for i in range(self.mcts_sims_per_move):
            node = root
            action = self.select_best_child(node)
            
            while node.has_child(action):
                node = node.get_child(action)
                action = self.select_best_child(node)

            # Next state is a copy.
            state, _, _ = env_sim.step(node.state, action, current_player)

            # Create new child to expand frontier of MCTS
            child_node = self.create_node(state, action, parent=node, env=env_sim)

            # Since we took a step, the next value is from the opponent's perspective.
            # Correct it to current player's perspective.
            value = -1 * child_node.value

            # Update root with search results for current simulation
            self.backpropagate(node, value)
            
        root_state = self.encoder.encode(state)
        visit_counts = np.array([root.visit_count(self.encoder.decode_action_index(idx))
                                    for idx in range(self.encoder.n_actions())
                                ])

        # Track trajectory and fill in MC reward at end of episode.
        self.trajectory_tracker.record_decision(root_state, visit_counts)

        return max(root.actions(), key=root.visit_count)

    def select_best_child(self, node):
        '''
        Selects the 'best' child node according to Polynomial Upper Confidence score.
        Args:
          node: (Node) this function evaluates possible actions for this node.
        Returns:
          best_child: (Node) child node with highest PUCT.
        '''
        # Defined as the parent from the perspective of the child
        N_parent = node.total_visit_count

        def puct(action):
            Q = node.expected_value(action)
            prior = node.prior(action)
            n_visits_child = node.visit_count(action)
            return Q + self.c_puct * prior * np.sqrt(N_parent) / (n_visits_child + 1)

        return max(node.actions(), key=score_branch)

    def create_node(self, state, action, parent, env):
        '''
        Creates a new leaf node for a previously unvisited state-action pair.
        Evaluates each possible child and initializes them with a score derived
        from the prior output by the policy network.

        In AlphaZero, the new leaf is the result with NN priors.
        In AlphaGo, it would be the terminal node after expansion and rollout.
        '''
        state_tensor = self.encoder.encode(state)

        priors, values = self.model.predict(np.array([state_tensor]))
        priors = priors[0]
        value = values[0][0]

        action_priors = {
            self.encoder.decode_action_index(idx): p
            for idx, p in enumerate(priors)
        }

        new_node = Node(state, value, action_priors, parent, env)
        
        # Only when adding root
        if parent is not None:
            parent.add_child(action, new_node)
        
        return new_node

    def backpropagate(self, node):
        '''
        Backpropagates information from the terminal node to the root node.
        
        Args:
          node (Node) newly expanded leaf node.
        '''
        while node is not None:
            node.update_statistics(action, value)
            node = node.parent
            value = -1 * value
