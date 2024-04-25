import random
import gym


class DiplomacySimpleRlAgent():

    def __init__(self, env: gym.Env, render=False):
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
        if (power_name, tuple(state)) in self.experiences:
            actions[power_name] = self.experiences[(power_name, state)][0]
        else:
            actions = self.random_nn_player_move(power_name, actions)
        return actions

    def log_experience(self, power_name, state, action, player_reward):
        if (power_name, state) in self.experiences:
            self.experiences[(power_name, tuple(state))] += (action, player_reward)
        else:
            self.experiences[(power_name, tuple(state))] = [(action, player_reward)]

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
            state_next, reward, done, info, rendering = self.env.step(actions, render=self.render)
            self.log_experience(player[1], current_state, actions[player[1]], reward[player[0]])

            current_state = 0, state_next
            finish = done[0]
            print(f'reward: {reward[0]}')
