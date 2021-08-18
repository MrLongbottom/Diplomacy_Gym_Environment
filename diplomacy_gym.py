import random

from Policy.diplomacy_environment import DiplomacyEnvironment


def move():
    actions = {}

    for power_name, power in env.game.powers.items():
        actions[power_name] = [random.choice(env.game.get_all_possible_orders()[loc]) for loc in
                               env.game.get_orderable_locations(power_name)]

    return env.step(actions)


env = DiplomacyEnvironment()

while not env.game.is_game_done:
    obs, reward, done, info = move()

print('game done.')


