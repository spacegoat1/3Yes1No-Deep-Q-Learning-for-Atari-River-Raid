# Deep Q Learning for Atari River Raid 

![plot](https://github.com/7ericany/3Yes1No-Deep-Q-Learning-for-Atari-River-Raid/blob/main/poster/2470%20Final%20Project%20Poster.png)

## Group members

Nange Li (nli32) nange_li@brown.edu

Yingfei Hong (yhong28) yingfei_hong@brown.edu

Yuetian Li (yli41) yuetian_li@brown.edu

Yash Mehta (ymehta3) yash_b_mehta@brown.edu



## Introduction

In the past couple years, through deep reinforcement learning, artificial agents have achieved a human-level of performance and generality in playing games and solving tasks. With enormous amounts of training, they can even outperform humans. In the current project, we focused on training deep reinforcement learning models to play Atari 2000 River Raid. Specifically, we used Deep Q-Learning Networks (DQN). In 2013, the paper by Deepmind Technologies explored the method of using Deep-Q learning on seven Atari 2600 games and showed with DQN, the agent can learn control policies directly from high-dimensional sensory input. 

In this project, we first started with a vanilla DQN in combination with a convolutional neural network. To improve the training accuracy and stability, we applied two strategies: fixed Q-targets and replay memory mechanism. Fixed Q-targets handle the oscillation in training due to shifting Q target values. Replay memory is to break the correlation between consecutive samples of experience in the environment that could lead to inefficient learning. River Raid is a complex game with 18 actions. Our goal was to train the agent to outperform the random model and to achieve human-level performance. With the optimization techniques and given enough training time, we hope the agent can outperform our team members. 


## Results
![plot](https://github.com/7ericany/3Yes1No-Deep-Q-Learning-for-Atari-River-Raid/blob/main/visualization/Trained_20k-episode-1.gif)



## Ethics

#### What broader societal issues are relevant to your chosen problem space?

We believe this has a direct impact on things like automated driving. We are also concerned about fair play in games - though the performance of an agent is way worse than humans, it’s not guaranteed that algorithms never outperform people in some certain games. It’s not a problem for light-weighted arcade games like the one we chose. But with regards to Esports, the game designers need to take the non-human players into account. Will those agents be used as plugins, and bring unfairness to the rankings, or even cause bad experiences to human players? 

#### Why is Deep Learning a good approach to this problem?

This problem seems tailor made for reinforcement learning given the nature of the task, and the fact that the RL agent will have to deal with sequential albeit sparse and noisy data. A CNN should also work well to accept an image of the game state and convert it into features that the model can use. 

