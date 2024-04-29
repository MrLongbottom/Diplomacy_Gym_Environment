import os
import pickle
import random
import time

import numpy as np
from IPython import display

from diplomacy_gym_environment import DiplomacyEnvironment
from tqdm import tqdm


class DiplomacyMCAgent:
    def __init__(self, env: DiplomacyEnvironment, render=False, explore_rate=0.1, learning_rate=0.2, reward_discount = 0.9, load=True, use_nn_states=True, path='mc_experience.pkl'):
        # Settings variables
        self.env = env
        self.render = render
        self.path = path
        self.use_nn_states = use_nn_states

        # Statistical variables
        self.new_state = 0
        self.old_state = 0
        self.errors = []

        # Learning variables
        self.explore_rate = explore_rate
        self.learning_rate = learning_rate
        self.reward_discount = reward_discount

        # simple dict which takes state and returns a (action, reward) list
        self.experiences = {}
        if load and os.path.isfile(path):
            self.experiences = self.load_experiences_from_file(path)

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

    def random_player_move(self, power_name, actions):
        actions[power_name] = [random.choice(self.env.game.get_all_possible_orders()[loc]) for loc in
                               self.env.game.get_orderable_locations(power_name)]
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
            if self.use_nn_states:
                actions = self.random_nn_player_move(power_name, actions)
            else:
                actions = self.random_player_move(power_name, actions)
            self.new_state += 1
        return actions

    def log_experience(self, power_name, player_states, player_actions, player_rewards):
        running_reward = 0
        for state, action, reward in zip(reversed(player_states), reversed(player_actions), reversed(player_rewards)):
            running_reward = (running_reward * self.reward_discount) + reward
            key = (power_name, tuple(state))
            # Known state
            if key in self.experiences:
                action_rewards = self.experiences[key]
                same_action = next(((a, r) for a, r in action_rewards if a == action), None)
                # New action
                if same_action is None:
                    self.experiences[key].append((action, running_reward))
                # Known action (adjust reward)
                else:
                    error = (running_reward - same_action[1])
                    new_reward = same_action[1] + (error * self.learning_rate)
                    new_action = (same_action[0], new_reward)
                    action_index = self.experiences[key].index(same_action)
                    self.experiences[key][action_index] = new_action
                    self.errors.append(abs(error))
            # New State
            else:
                self.experiences[key] = [(action, running_reward)]

    def save_experiences_to_file(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.experiences, file)

    def load_experiences_from_file(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def update_active_players(self, players, done):
        active_players = 0
        new_players = []
        for x, y, z in players:
            if z:
                new_players.append((x, y, z))
            else:
                new_players.append((x, y, done[active_players]))
                active_players += 1
        return new_players

    def render_state(self, rendering, wait_time):
        # display actions commited state
        display.display(display.SVG(rendering))
        time.sleep(wait_time)
        display.clear_output(wait=True)

        # display resulting state
        rendering = self.env.render()
        display.display(display.SVG(rendering))
        time.sleep(wait_time)
        display.clear_output(wait=True)

    def play(self, save=True, render=None):
        if render is None:
            render = self.render
        self.new_state = 0
        self.old_state = 0
        self.errors = []

        current_state = self.env.reset()
        players = [(x, y, False) for x, y in enumerate(list(self.env.game.powers.keys()))]

        episode_states = {x[1]: [] for x in players}
        episode_actions = {x[1]: [] for x in players}
        episode_rewards = {x[1]: [] for x in players}
        done = [False]

        with tqdm(total=200, position=0, leave=True) as pbar:
            while False in done:
                # Decide actions for each player
                actions = {}
                for player_num, player_name, player_finish in players:
                    if not player_finish:
                        actions = self.decide_player_action(player_name, current_state, actions)
                    else:
                        actions[player_name] = []

                # Apply the sampled action in our environment
                if render:
                    state_next, reward, done, info, rendering = self.env.step(actions, render=render, input_is_nn=self.use_nn_states)
                    self.render_state(rendering, 1)
                else:
                    state_next, reward, done, info = self.env.step(actions, render=render, input_is_nn=self.use_nn_states)

                # Log experiences
                active_players = 0
                for player_num, player_name, player_finish in players:
                    if not player_finish:
                        episode_states[player_name].append(current_state)
                        episode_actions[player_name].append(actions[player_name])
                        episode_rewards[player_name].append(reward[active_players])
                        active_players += 1

                current_state = state_next
                players = self.update_active_players(players, done)

                pbar.set_description(f'Turn: {info[0]}')
                if 'R' in info[0] or 'W' in info[0]:
                    pbar.total += 1
                pbar.update(1)

        # log experience
        for player_num, player_name, player_finish in players:
            self.log_experience(player_name, episode_states[player_name], episode_actions[player_name], episode_rewards[player_name])

        print(f"Game done. New states: {self.new_state}, old states: {self.old_state}, total experiences: {len(self.experiences)}, avg. error: {sum(self.errors) / max(len(self.errors), 1)}, total error {sum(self.errors)}")
        if save:
            self.save_experiences_to_file(self.path)