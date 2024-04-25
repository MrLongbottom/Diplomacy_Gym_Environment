import random
from diplomacy_gym_environment import DiplomacyEnvironment


class DiplomacySimpleRlAgent:

    def __init__(self, env: DiplomacyEnvironment, render=False):
        self.env = env
        self.render = render
        # simple dict which takes state and returns a (action, reward) list
        self.experiences = {}
        self.player = None

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
        if key in self.experiences:
            max_reward_value = max([y for (x,y) in self.experiences[key]])
            max_reward_action = next(x for (x,y) in self.experiences[key] if y == max_reward_value)
            actions[power_name] = max_reward_action
        else:
            actions = self.random_nn_player_move(power_name, actions)
        return actions

    def log_experience(self, power_name, state, action, player_reward):
        key = (power_name, tuple(state))
        if key in self.experiences:
            self.experiences[key].append((action, player_reward))
        else:
            self.experiences[key] = [(action, player_reward)]

    def play(self):
        current_state = self.env.reset()
        player = 0, list(self.env.game.powers.keys())[0]
        finish = False

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
            print(f'turn: {info[0]}, reward: {reward[player[0]]}')
        print(f"game done.")
