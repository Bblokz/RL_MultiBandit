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
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy, GradientBanditAlgorithm
from Helper import LearningCurvePlot, ComparisonPlot, smooth


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):

    # Bonus: Gradiant Bandit
    gLearningHyper = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    
    # run_gradient
    sumResultGradient = np.zeros(len(gLearningHyper))
    plotHelper = LearningCurvePlot(
        title="Average performance Gradient Bandit over {} repetitions".format(n_repetitions))

    for index, learningRate in enumerate(gLearningHyper):
        print("running Gradient value " + str(learningRate))
        result = run_gradient(n_actions, n_timesteps, n_repetitions, learningRate)
        # sumResultOI[index] = sum(result)/(n_timesteps * n_repetitions)
        plotHelper.add_curve(
            smooth(result/float(n_repetitions), window=smoothing_window), label='Gradient value ' + str(learningRate))

    plotHelper.save("Gradient.png")

    pass


def run_egreedy(n_actions, n_timesteps, n_repetitions, eHyper):
    vectorResult = np.zeros(n_timesteps)
    for j in range(n_repetitions):
        # Initialize policy
        env = BanditEnvironment(n_actions=n_actions)
        pi = EgreedyPolicy(n_actions=n_actions)

        for i in range(1, n_timesteps+1):
            a = pi.select_action(epsilon=eHyper)  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r)  # update policy

    return vectorResult


def run_OI(n_actions, n_timesteps, n_repetitions, learnHyper, initHyper):
    vectorResult = np.zeros(n_timesteps)
    for j in range(n_repetitions):
        # Initialize policy
        env = BanditEnvironment(n_actions=n_actions)
        pi = OIPolicy(n_actions=n_actions,
                      initial_value=initHyper, learning_rate=learnHyper)
        for i in range(1, n_timesteps+1):
            a = pi.select_action()  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r)  # update policy

    return vectorResult


def run_ucb(n_actions, n_timesteps, n_repetitions, cHyper):
    vectorResult = np.zeros(n_timesteps)
    for j in range(n_repetitions):
        env = BanditEnvironment(n_actions=n_actions)
        pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
        for i in range(1, n_timesteps+1):
            a = pi.select_action(c=cHyper, t=i)  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r)  # update policy

    return vectorResult

def run_gradient(n_actions, n_timesteps, n_repetitions, learningRate):
    vectorResult = np.zeros(n_timesteps)
    
    for j in range(n_repetitions):
        env = BanditEnvironment(n_actions=n_actions)
        pi = GradientBanditAlgorithm(n_actions=n_actions)  # Initialize policy
        for i in range(1, n_timesteps+1):
            a = pi.select_action()  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r, learningRate)  # update policy

    return vectorResult


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
