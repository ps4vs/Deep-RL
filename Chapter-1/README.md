---
title: "Notes on Chapter 1"
date: 2023-12-03T16:26:49+05:30
draft: true
---

## Introduction to RL
Reinforcement Learning is computational approach to goal-directed "Learning from Interaction" which is foundational idea underlying nearly all theories of learning and intelligence, starting from animals to humans.

In RL, The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (ie, The learner learns how to map situations to actions, so as to maximize a numerical reward signal by interacting with the environment).

Two most important distinguishing features of RL are:
* trial-and-error search
* delayed reward

## RL comparision with Supervised and Unsupervised Learning

*Supervised Learning*, which is learning from a training set of labeled examples, which helps the system to extrapolate, or generalize, its responses so that it acts correctly in situations not present in training set.

Agent learning from interaction (ie, RL problems) using supervised learning is impractical to obtain examples of desired behaviour that are both correct and representative of all the situations in which agent has to act.

*Unsupervised Learning*, which is about finding structure hidden in collections of unlabeled data. 

Unsupervised Learning might seem like RL, but RL is trying to maximize a reward signal instead of trying to find hidden structure (which can be useful in RL)

## My Favourite

In contrast to many approaches that consider subproblems without addressing how they might fit into a larger picture. Reinforcement Learning explicitly considers the whole problem of a goal-directed agent interacting with an uncertain environment. (ie, takes the opposite track, starting with a complete, interactive, goal-seeking agent. ) 

* Example 1: Many researchers have studied Supervised Learning, but doesn't specify how such an ability would be ultimately be useful contributing to the development of AGI.
* Example 2: Other reseachers have developed theories of planning with general goals, but without considering planning's role in real-time decision making, or the question of where the predictive models necessary for planning would come from.

When RL involves planning, it has to address the interplay b/w planning and real-time action selection as well as the question of how environment models are acquired and improved, despite significant uncertainty about the environment.


## Elements of RL
Four main subelements of a RL system are:
* a policy: a learning agent's way of behaving at a given time
* a reward signal: goal of the reinforcement learning problem
* a value function: predictions of rewards, we are most concerned when making and evaluating decisions
* a model of the environment: mimics the behaviour of the environment

Reward signal indicates what is good in an immediate sense, a value function specifies what is good in the long run. 

"Without rewards there could be no values, and the only purpose of estimating values is to achieve more rewards"

Rewards are basically given directly by the environment, but values must be estimated and re-estimated from the sequence of observations an agent makes over its entire lifetime.

Models are used for planning, by which we mean any way of deciding on a course of action by considering possible future situations before they are actually experienced.

Methods for solving reinforcement learning problems that use models and planning are called model-based methods, as opposed to simpler model-free methods that are explicity trial-and--error learners - viewed as opposite of planning.

## playing Tic Tac Toe 
We want to create an Agent which can play against an fixed opponent.

It can be solved using multiple ways:
* using dynamic programming which requires creating an opponent probabilistic model, through multiple iterations of plays against the opponent to get an estimated model, then use dynamic programming to win against him.
* using evolutionary method, which will search the space of possible policies for one with high probability of winning against the opponent. It will hill climb in policy space. (genetic-style algorithm can be used, maintaining and evaluating a population of policies)
* many others.

The initial value are set to 1 for all the winning states, 0 for all the loosing and draw states, and 0.5 for remaining states. The values are updated using temporal difference learning method to learn optimal policy, ie, changes are based on difference V(S_{t+1}) - V(S_t), between estimates at two successive times.

It involves both exploratory and greedy moves, where exploratory creates different situations, instead of exploiting a strategy.

### Evolutionary vs Value Function methods
In evolutionary method, what happens during the game is ignored, credit is given to all moves based on the final result. Value function method, in contrast, allow individual states to be evaluated. 

Both of them search the space of policies, but learning a value function takes advantage of information available during the course of play.

The use of value functions distinguishes reinforcement learning methods from evolutionary methods that search directly in policy space guided by evaluations of entire policies.

### RL x Supervised Learning
How well RL system can work in problems with large state sets is tied to how appropriately it can generalize from past experience, which is where supervised learning methods aid RL

## Observations from implementation
* The agent learns to block, if I try to use the same action.
* The agent can learn and unlearn depending on the final state it reaches, based on the type of temporal update it gets, ie, negative or postive.
* The agent when played against opponent who takes moves randomly takes too much time to learn, because the random play teaches the agent randomly.

## Exercises
*Exercise 1.1 Self-Play* Suppose, instead of playing against a random opponent, the
reinforcement learning algorithm described above played against itself, with both sides
learning. What do you think would happen in this case? Would it learn a di↵erent policy
for selecting moves?

The agent updates different set of states depending on whether it plays first player or second player. Initially due to randomness someone will be win, agent as first player or agent as second player. This updates the value of winning states and loosing states continously. The agent will try to continue the learning, but will achieve average play. As the opponent is continously evolving, ie, self evolving, it might not be able to converge if we reduce the exploration factor.

*Exercise 1.2: Symmetries* Many tic-tac-toe positions appear di↵erent but are really
the same because of symmetries. How might we amend the learning process described
above to take advantage of this? In what ways would this change improve the learning
process? Now think again. Suppose the opponent did not take advantage of symmetries.
In that case, should we? Is it true, then, that symmetrically equivalent positions should
necessarily have the same value? 

If the agent is able to utilize symmetries, the number of states will reduce, and the convergence will happen faster, because of the symmetric, it can get more updates for a single state. If opponent did not take advantage, and use symmetric advantage against us, then we would not be able to find optimal policy. If the opponent don't take advantage of our symmetric advantage then all symmetrically equivalent positions will have same value.

*Exercise 1.3: Greedy Play* Suppose the reinforcement learning player was greedy, that is,
it always played the move that brought it to the position that it rated the best. Might it learn to play better, or worse, than a nongreedy player? What problems might occur?

The greedy player will not explore, so it will perform poorly compared to the non-greedy player.

*Exercise 1.4: Learning from Exploration* Suppose learning updates occurred after all
moves, including exploratory moves. If the step-size parameter is appropriately reduced
over time (but not the tendendcy to explore), then the state values would converge to
a different set of probabilities. What (conceptually) are the two sets of probabilities
computed when we do, and when we do not, learn from exploratory moves? Assuming
that we do continue to make exploratory moves, which set of probabilities might be better
to learn? Which would result in more wins?

The updates related to exploratory moves might or might not be beneficial.

The probabilities computed when including exploratory move is optimal policy along with some randomness, and other is optimal policy can be considered as mean and mean+std.

*Exercise 1.5: Other Improvements* Can you think of other ways to improve the reinforcement learning player? Can you think of any better way to solve the tic-tac-toe problem
as posed?

something like running average of updates, like rmsprop, adam etc.
ranking draws as better than losses, giving them more value than losses around 0.2 ish etc.
