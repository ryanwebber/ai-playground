import gym
import numpy as np
from time import sleep

env = gym.make("CartPole-v0")

def _run_once(env, max_steps, model, render):
    score = 0
    data = []
    obs = env.reset()
    for _ in range(max_steps):
        if render:
            env.render()

        if model:
            action = model.predict(obs)
        else:
            action = env.action_space.sample()

        one_hot = [1 if i == action else 0 for i in range(env.action_space.n)]
        data.append([obs, one_hot])

        obs, reward, done, info = env.step(action)
        score += reward

        if done:
            break

    return data, score

def play(iterations=1, render=False, model=False, dumb=False, max_steps=1000, iter_sleep=0):
    if not model and not dumb:
        raise Exception('Must either provide a model or use a dumb ai')

    if model and dumb:
        raise Exception('Cannot execute given model and dumb model at the same time')

    scores = []
    data = []
    for i in range(iterations):
        d,s = _run_once(env, max_steps, model, render)
        data.append(d)
        scores.append(s)
        sleep(iter_sleep)

    return data, scores

