import os.path
from diplomacy_gym_environment import DiplomacyEnvironment
import random


# This is a template setup on how to actually use the Gym environment I made, by setting up a RL agent.
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


def generator():
    while not env.game.is_game_done:
        yield


# This will run through one game taking random actions using a neural network-like output in order to test the gym
# environment
if __name__ == '__main__':
    # My own custom-made gym environment
    maps_path = 'maps/'
    if not os.path.exists(maps_path):
        os.makedirs(maps_path)

    env = DiplomacyEnvironment(prints=False, render_path=None)

    finish = False

    while not finish:
        # Take random action
        action = random_nn_move()

        # Apply the sampled action in our environment
        state_next, reward, done, info = env.step(action)
        finish = done[0]
        print(f'reward: {reward[0]}')
        test = env.render()
