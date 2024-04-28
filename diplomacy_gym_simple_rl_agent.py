import os
import pickle
import random
from diplomacy_gym_environment import DiplomacyEnvironment
from tqdm import tqdm


class DiplomacySimpleRlAgent:
    def __init__(self, env: DiplomacyEnvironment, render=False, alpha=0.1, load=True):
        self.env = env
        self.render = render
        self.player = None
        self.game_reward_total = 0
        self.new_state = 0
        self.old_state = 0
        self.explore_rate = alpha

        # simple dict which takes state and returns a (action, reward) list
        if load and os.path.isfile('experience.pkl'):
            self.experiences = self.load_experiences_from_file('experience.pkl')
        else:
            self.experiences = {}

    def random_move(self):
        actions = {}
        for power_name in self.env.game.powers.keys():
            actions[power_name] = [random.choice(self.env.game.get_all_possible_orders()[loc]) for loc in
                                   self.env.game.get_orderable_locations(power_name)]
        return actions

    def random_nn_move(self):
        actions = {}
        for power_name in self.env.game.powers.keys():
            actions[power_name] = [random.random() for _ in self.env.action_list]
        return actions

    def random_nn_player_move(self, power_name, actions):
        actions[power_name] = [random.random() for _ in self.env.action_list]
        return actions

    def decide_player_action(self, power_name, state, actions):
        key = (power_name, tuple(state))
        random_number = random.random()
        if random_number > self.explore_rate and key in self.experiences:
            max_reward_value = max([y for (x,y) in self.experiences[key]])
            max_reward_action = next(x for (x,y) in self.experiences[key] if y == max_reward_value)
            actions[power_name] = max_reward_action
            self.old_state += 1
        else:
            actions = self.random_nn_player_move(power_name, actions)
            self.new_state += 1
        return actions

    def log_experience(self, power_name, state, action, player_reward):
        self.game_reward_total += player_reward
        key = (power_name, tuple(state))
        if key in self.experiences:
            self.experiences[key].append((action, player_reward))
        else:
            self.experiences[key] = [(action, player_reward)]

    def save_experiences_to_file(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.experiences, file)

    def load_experiences_from_file(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def play(self, save=True):
        self.game_reward_total = 0
        self.new_state = 0
        self.old_state = 0
        current_state = self.env.reset()
        player = 0, list(self.env.game.powers.keys())[0]
        finish = False

        with tqdm(total=500, position=0, leave=True) as pbar:
            while not finish:
                actions = {}
                for power_name in self.env.game.powers.keys():
                    if power_name == player[1]:
                        actions = self.decide_player_action(power_name, current_state, actions)
                    else:
                        actions = self.random_nn_player_move(power_name, actions)

                # Apply the sampled action in our environment
                state_next, reward, done, info = self.env.step(actions, render=self.render)
                self.log_experience(player[1], current_state, actions[player[1]], reward[player[0]])

                current_state = state_next[player[0]]
                finish = done[player[0]]
                pbar.set_description(f'turn: {info[0]}, centers: {len(info[1][player[1]])}')
                pbar.update(1)

        print(f"Game done. Total reward: {self.game_reward_total}, new states: {self.new_state}, old states: {self.old_state}. total experiences: {len(self.experiences)}")
        if save:
            self.save_experiences_to_file('experience.pkl')