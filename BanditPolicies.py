#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment


class EgreedyPolicy:

    # Initialize policy: set number of actions and initialize estimates.
    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.estimates = np.zeros(n_actions)
        self.steps = np.zeros(n_actions)
        pass

    # Select action: select action according to epsilon-greedy policy.
    def select_action(self, epsilon):
        rand = np.random.uniform(0, 1)
        if rand < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.estimates)

        return a

    # Update policy: update estimates of action values.
    def update(self, a, r):
        self.steps[a] += 1
        self.estimates[a] += (1/self.steps[a])*(r-self.estimates[a])
        pass


class OIPolicy:

    # Initialize policy: set number of actions and initialize estimates.
    # Set learning rate and optimisitc initial value.
    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.estimates = np.zeros(n_actions)
        self.learning_rate = learning_rate
        self.estimates += initial_value
        pass

    # Select action: select action according to greedy policy.
    def select_action(self):
        return np.argmax(self.estimates)

    # Update policy: update estimates of action values.
    def update(self, a, r):
        self.estimates[a] += self.learning_rate * (r-self.estimates[a])
        pass


class UCBPolicy:

    # Initialize policy: set number of actions and initialize estimates.
    # Set steps to small non-zero value to avoid division by zero.
    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.estimates = np.zeros(n_actions)
        self.steps = np.zeros(n_actions)
        self.steps += 1e-3
        pass

    # Select action: select action according to UCB policy.
    def select_action(self, c, t):
        x = np.argmax(self.estimates + c * np.sqrt(np.log(t) / (self.steps)))
        return x

    # Update policy: update estimates of action values.
    def update(self, a, r):
        self.steps[a] += 1
        self.estimates[a] += (1/self.steps[a])*(r-self.estimates[a])
        pass

class GradientBanditAlgorithm:

    # Initialize policy: set number of actions and initialize preferences.
    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.totalReward = 0
        self.preferences = np.zeros(n_actions)
        self.steps = 1
        pass

    def policy(self):
        return np.exp(self.preferences) / np.sum(np.exp(self.preferences))

    # Select action: select action according to gradient bandit policy.
    def select_action(self):
        return np.argmax(self.policy())

    # Update policy: update preferences of action values.
    # The update rule is given by the gradient of the log-likelihood.
    # Note: alpha is the learning rate.
    def update(self, a, r, alpha):
        self.totalReward += r
        self.preferences[a] += alpha * (r - self.totalReward / self.steps) * (1 - self.policy()[a])
        for (i, p) in enumerate(self.preferences):
            if i != a:
                self.preferences[i] -= alpha * (r - self.totalReward / self.steps) * self.policy()[i]
        self.steps += 1
        pass


def test():

    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions)  # Initialize environment

    pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(epsilon=0.5)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a, r))

    pi = OIPolicy(n_actions=n_actions, initial_value=1.0)  # Initialize policy
    a = pi.select_action()  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a, r))

    pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(c=1.0, t=1)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test UCB policy with action {}, received reward {}".format(a, r))

    pi = GradientBanditAlgorithm(n_actions=n_actions)  # Initialize policy
    a = pi.select_action()  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r, 0.1)  # update policy
    print("Test gradientBandit policy with action {}, received reward {}".format(a, r))


if __name__ == '__main__':
    test()
