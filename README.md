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
### Game Recordings
Our trained agent playing the game (speed-up):

![plot](https://github.com/7ericany/3Yes1No-Deep-Q-Learning-for-Atari-River-Raid/blob/main/visualization/Trained_20k-episode-1.gif)

Random agent (speed-up):

![plot](https://github.com/7ericany/3Yes1No-Deep-Q-Learning-for-Atari-River-Raid/blob/main/visualization/Random_model-episode-1.gif)

### Graphs
![plot](https://github.com/7ericany/3Yes1No-Deep-Q-Learning-for-Atari-River-Raid/blob/main/visualization/Training_reward_history.png)
![plot](https://github.com/7ericany/3Yes1No-Deep-Q-Learning-for-Atari-River-Raid/blob/main/visualization/rewards_historgram_comparison.png)

## Ethics

#### What broader societal issues are relevant to your chosen problem space?

We believe this has a direct impact on things like automated driving. We are also concerned about fair play in games - though the performance of an agent is way worse than humans, it’s not guaranteed that algorithms never outperform people in some certain games. It’s not a problem for light-weighted arcade games like the one we chose. But with regards to Esports, the game designers need to take the non-human players into account. Will those agents be used as plugins, and bring unfairness to the rankings, or even cause bad experiences to human players? 

#### Why is Deep Learning a good approach to this problem?

This problem seems tailor made for reinforcement learning given the nature of the task, and the fact that the RL agent will have to deal with sequential albeit sparse and noisy data. A CNN should also work well to accept an image of the game state and convert it into features that the model can use. 

## Reflections
The final model performs a little better than a human beginner in the game. One of our team members played online RiverRaid on this website a few times, and the best score with 1 life (4 lives for each game, but we adopt only one life for model rewards) is around 1500. A human expert on YouTube can get a score of 10k - 100k. Given these human records, we believe we already meet the base and target goals, and partially reach our stretch goal of achieving human level performance. However, as we evaluated the training process, the jet agent failed to distinguish the fuel depots from other enemies (helicopters, tanks), but just shooting them with 80 points each. The agent should learn to refuel instead of firing for points all the time, otherwise it would run out of fuel. However, our model has not reached that far. To improve, we should extract the information from the energy bar and add a term to the reward function.

More experiments can be extended. For instance, we can tune k values in the k-skipping method. A larger k value provides the model with more information to predict an action, but as the action is repeated k times and it might not be the optimal strategy. So there is a tradeoff between faster training of large k and fine-grained action decision of small k. Tests on how the training rewards interact with different k values may provide more insights.
Given that we were building a model for a specific game, I think we should have spent some more time upfront in understanding the game mechanics, what it takes to get a high score in the game, what the elements of the game screen are etc. As things stood, when we implemented our models, we were taking a more trial-and-error approach to making model improvements. 

RiverRaid is a fairly complex game, and I think we could have implemented a more complex model to play the game. However, we didn’t have the time to work on it. It may also have been interesting to consider different approaches, like supplying the states to the model. Eg. instead of passing the last 4 screenshots, we could have passed the most recent screenshots, and the differences between the previous screenshots. However, this would not be equivalent to how a human user sees the game. 

Recalled from the Vanilla DQN, we use Q(s,a) to represent how good it is to be at state s and taking the action a. So the Q(s,a) is actually the combination of the value of being at state s and the advantage of taking action at the state. And this is the idea of DDQN. The DDQN uses the Q network to estimate the advantage of taking action at the state and another V network to estimate the state value. And the ultimate Q value will be the sum of the state value V(s) and the centered advantage (Q(s,a) - the average advantages of all possible actions). This is also a good way to reduce the instability of the DQN training by calculating the state value without learning the effect of actions at a given state. And it is really useful in our case because for most of the actions, like left or right or accelerate, they do not matter unless the plane hits the wall or hits the moving enemy. And in most states, the choice of actions has no effect on what happens. However, due to the time limit, although we have implemented the dual DQN model, we are not able to train this model and also the combination of this model with other mechanisms like memory mechanism.

In summary, the current project is successful in training the agent to learn control policies based on high-dimensional sensory input through deep Q-learning networks. The additional techniques including k-skipping method, fixed Q-targets, and replay memory mechanism were incorporated into input preprocessing and  model building to achieve a more stable and efficient training. We have achieved our base goal and target goal to achieve similar levels or even outperform our team members. Future work could implement additional techniques such as Dual DQN and tuning k values in k-skipping method  to improve agent learning. 

