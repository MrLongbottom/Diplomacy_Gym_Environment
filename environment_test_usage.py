from diplomacy_gym_environment import DiplomacyEnvironment
import random
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython import display

# This is a template setup on how to actually use the Gym environment I made, by setting up a RL agent.
env = DiplomacyEnvironment(prints=False, render_path=None)


def random_move():
    actions = {}
    for power_name in env.game.powers.keys():
        actions[power_name] = [random.choice(env.game.get_all_possible_orders()[loc]) for loc in
                               env.game.get_orderable_locations(power_name)]
    return actions


def random_nn_move():
    actions = {}
    for power_name in env.game.powers.keys():
        actions[power_name] = [random.random() for _ in env.action_list]
    return actions


# This will run through one game taking random actions using a neural network-like output in order to test the gym environment
if __name__ == '__main__':
    finish = False
    wait_time = 1

    while not finish:
        # Take random action
        action = random_nn_move()

        # Apply the sampled action in our environment
        state_next, reward, done, info, rendering = env.step(action, render=False)
        finish = done[0]

        # display actions commited state
        display.display(display.SVG(rendering))
        time.sleep(wait_time)
        display.clear_output(wait=True)

        # print reward
        print(f'reward: {reward[0]}')

        # display resulting state
        rendering = env.render()
        display.display(display.SVG(rendering))
        time.sleep(wait_time)
        display.clear_output(wait=True)
