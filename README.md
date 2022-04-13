# Deep Q Learning for Atari River Raid 



## Group members

Nange Li (nli32) nange_li@brown.edu

Yingfei Hong (yhong28) yingfei_hong@brown.edu

Yuetian Li (yli41) yuetian_li@brown.edu

Yash Mehta (ymehta3) yash_b_mehta@brown.edu



## Introduction

####What problem are you trying to solve and why?

The goal of our project is to develop a deep reinforcement model to play the Atari 2000 River Raid.

In recent years, AI has broken human records across a variety of games using reinforcement learning. Our motivation comes from a desire to learn more about this field in a hands-on, yet fun manner. We plan to use the atari-py package to simulate the environment and interface with the game space. Our broad goal is to use a convolutional neural network to identify game elements and pass them to a reinforcement learning model, trying a few different architectures to compare performance. Some of our considerations will be to chain together sequences of game states to estimate movement and velocity of game elements, and how the algorithms can balance greed vs longer term optimization. 

#### What kind of problem is this?

This is a Reinforcement Learning problem. Specifically, we believe it is a Deep-Q Learning problem, however we aren't sure of this since we haven't covered this material in class yet!



## Related Work

#### Are you aware of any, or is there any prior work that you drew on to do your project?

Yes, there is significant work done on playing video games using deep reinforcement learning. 

#### Please read and briefly summarize (no more than one paragraph) at least one paper/article/blog relevant to your topic beyond the novel idea you are researching.

We are particularly inspired by the "Playing Atari with Deep Reinforcement Learning" paper put together by the folks at DeepMind Technologies. In that, they use CNNs and Q Learning to build a single "general purpose" network to play 7 Atari games. The approach was so successful that it beat human high scores at 6 out of the 7 games! 

#### In this section, also include URLs to any public implementations you find of the paper you're trying to implement. Please keep this as a "living list"--if you stumble across a new implementation later down the line, add it to this list. 

https://arxiv.org/abs/1312.5602

https://cs.stanford.edu/~rpryzant/data/rl/paper.pdf



## Data

#### What data are you using (if any)?

As this is a reinforcement learning problem, there isn't an existing dataset that we are working on. We plan to use OpenAI's Gym environment (https://gym.openai.com/) to set up the environment to interact with the network. 



## **Methodology**

#### What is the architecture of your model?

We don't know for sure, but we expect to have a few convolutional layers (2-4) followed by a few dense layers which end with a layer with number of nodes corresponding to the number of possible actions in the game space. We will experiment with the architecture, with the learning rate, consider using dropout etc. The "second part" of the network is going to be the reinforcement learning agent. For this we'll have to make a number of decisions regarding policy, memory, etc. but we haven't covered this material in class so we aren't very familiar with it. We will consider it in more detail once we have a better understanding of the theory. We also note that since successive game states are closely linked to each other, the network we build will also consider a time dimension, and we expect to supply a sequence of states to the network at each time point. 

#### How are you training the model?

Reinforcement learning models take a large amount of time, so we expect that we will have to train it on a GPU, possibly on GCP. 

#### If you are doing something new, justify your design. Also note some backup ideas you may have to experiment with if you run into issues.

We believe that our design makes intuitive sense in that the CNN will convert the game space into a set of features that the model can use to infer the state of the game, and the RL agent will help the network learn what actions to take accordingly. In terms of backup ideas, we can consider switching to a simpler game. 



## Metrics

#### What experiments do you plan to run?

We plan to experiment with different architectures, learning rates, dropout, memory sequence lengths and policies. We will come up with a structured way to explore this space since training the network is expected to take a long time.

#### Does the notion of "accuracy" apply for your project, or is some other metric more appropriate?

We’ll not use accuracy, instead we will use the final score of the game (rewards).

#### If you are doing something new, explain how you will assess your model's performance.

We believe that a natural metric for assessing model performance is to view the average reward obtained by different models. Since this is directly accessible through the gym environment, we will focus on this. We will also look at things like duration for which the game is played and conduct visual examinations to understand how the model is performing. If possible, we will also try to understand the features that the CNN is parsing from the game state. 

#### What are your base, target, and stretch goals?

* Base goal: Build a network that does better than random at playing the game.
* Target goal: Build a network that does 'much' better than random at playing the game. Experiment with different architectures and understand how different hyperparameters affect model performance.
* Stretch goal: Understand the features being extracted by the CNN and how the RL agent is behaving. Try to achieve human level performance on the game. 



## Ethics

#### What broader societal issues are relevant to your chosen problem space?

We believe this has a direct impact on things like automated driving. We are also concerned about fair play in games - though the performance of an agent is way worse than humans, it’s not guaranteed that algorithms never outperform people in some certain games. It’s not a problem for light-weighted arcade games like the one we chose. But with regards to Esports, the game designers need to take the non-human players into account. Will those agents be used as plugins, and bring unfairness to the rankings, or even cause bad experiences to human players? 

#### Why is Deep Learning a good approach to this problem?

This problem seems tailor made for reinforcement learning given the nature of the task, and the fact that the RL agent will have to deal with sequential albeit sparse and noisy data. A CNN should also work well to accept an image of the game state and convert it into features that the model can use. 



## Division of labor

* Yash - GYM environment setup, adjust to GCP settings
* Nange - Previous projects investigation & papers review
* Yingfei - Modeling: Game data generation, DL pipeline construction
* Yuetian - Report outline and visualization
