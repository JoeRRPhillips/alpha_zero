# alpha_zero
This is my own implementation of DeepMind's AlphaZero, using TensorFlow 2.0.  

Usage:  
• Clone this repository  
• cd alpha_zero  
• RUN: python training/train_connect_four.py  

Neural Network Architecture:  
• ResNet backbone to encode game state.  
• ActorHead to approximate policy function.  
• CriticHead to approximate value function.  

Training Method:  
• Run Monte Carlo Tree Search simulations to select each action (move) in the game.  
• Update the policy function approximator towards the MCTS node visit counts.  
• Update the value function approximator towards the real Monte Carlo return (reward) gained at the end of an episode (game).  

Comments:  
• Current implementation is on a custom Connect4 environment.  
• This is for learning more about RL; environment design can be made more efficient.  
• This was undertaken as a fun project - the next one will be AlphaZero for (amateur) Go, building upon lessons learned in this project.  
