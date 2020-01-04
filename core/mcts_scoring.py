# More detailed implementations of scoring metrics in MCTS.
# Not used in the main implementation, but supplied for reference and testing.


def puct(self, node, child_node):
        '''
        Computes PUCT (Polynomial Upper Confidence Bound 1 applied to trees).
        Variant used in AlphaZero paper.
        https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5

        At time step i:
            a_t = argmax_a { Q(s_i, a) + u(s_i, a) }

        Where:
            • Q(s,a) = combined mean action value for the edge (action a).
                       I.e. the average game result across the current simulations that took action a.
            • u(s,a) = c_puct * P(s,a) * (sum_b[N_r(s,b)] / (1 + N_r(s,a)))
                
                - b = all possible actions explored from s, a = taken action.
                        --> Less tried action --> more exploration, encouraged by larger c_puct.
                - P(s,a) = prior probabilities of chosing action a, according to policy fetched from the neural network.
                           I.e. how the NN guides the MCTS.
                - N_r(s,a) = leaf node visit count.
                             I.e. number of times this action has been taken during the current simulation.
        '''
        assert child_node.prior is not None, "Child node prior not set."

        # average_value_at_s_i - estimates the Q value over possible actions in a given node.
        q_val_action = self._compute_q_value(child_node)

        prior = child_node.prior * mask(state)

        # c_puct * P(s,a) * sqrt(N_i / (1+n_i)).  Avoid ÷ 0.
        action_utility = self.c_puct * prior * np.sqrt( node.visit_count / (1 + child_node.visit_count) )

        return q_val_action + action_utility

    
    def uct(self, node, child_node):
        '''
        Computes UCT (Upper Confidence Bound 1 applied to trees).
        Variant proposed by Auer, Cesa-Bianchi, and Fischer.
        Variant used in AlphaGo paper.
        https://www.youtube.com/watch?v=UXW2yZndl7U&t=1s

        UCT = w_i/n_i + c_puct*sqrt(log(N_i) / n_i),
        
        Where:
            w_i / n_i = average state value at node
            N_i = (parent) node visit count
            n_i = child node visit count
        '''
        # w_i/n_i
        average_state_value = sum(node.total_action_value) / child_node.visit_count

        # c_puct * sqrt(log(N_i) / n_i). Avoid ÷ 0.
        action_utility = self.c_puct * np.sqrt( np.log(node.visit_count) / child_node.visit_count ) \
                            if (child_node.visit_count > 0) else np.inf

        return average_state_value + action_utilitys
