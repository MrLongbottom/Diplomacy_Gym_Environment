from diplomacy_gym_environment import DiplomacyEnvironment
import random
from tqdm import tqdm


def random_move():
    actions = {}
    for power_name in env.game.powers.keys():
        actions[power_name] = [random.choice(env.game.get_all_possible_orders()[loc]) for loc in
                               env.game.get_orderable_locations(power_name)]
    return env.step(actions)


def random_nn_move():
    actions = {}
    for power_name in env.game.powers.keys():
        actions[power_name] = [random.random() for _ in env.action_list]
    return env.step(actions)


def generator():
    while not env.game.is_game_done:
        yield


if __name__ == '__main__':
    env = DiplomacyEnvironment()

    for _ in tqdm(generator()):
        obs, reward, done, info = random_move()

    print('game done.')


