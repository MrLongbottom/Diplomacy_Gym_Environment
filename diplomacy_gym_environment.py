import re
import time
from itertools import product
from numbers import Number
import diplomacy
import gym
import numpy as np
import collections
from gym import spaces
from diplomacy.utils.export import to_saved_game_format
from tqdm import tqdm


class DiplomacyEnvironment(gym.Env):

    def __init__(self, prints=False, render_path=None):
        self.prints = prints
        self.render_path = render_path
        super(DiplomacyEnvironment, self).__init__()
        self.game = diplomacy.Game()
        self.action_list, self.action_loc_dict, self.action_order_dict = self._action_list()
        self.reward_range = (-self.game.win, self.game.win)
        high = np.array([1.0 for _ in self.action_list])
        self.action_space = spaces.Box(high=high, low=-high, dtype=np.float32)
        self.observation_space = spaces.MultiBinary(len(self.observation()))
        if self.prints:
            print('Initialization done.')

    def step(self, action_n, render=False):
        old_state = self.game.get_state()
        info = {}
        # diplomacy package string orders input
        if isinstance(list(action_n.values())[0][0], str):
            for power, orders in action_n.items():
                self.game.set_orders(power, orders)
        # nn list(numbers) input
        elif isinstance(list(action_n.values())[0][0], Number):
            if self.prints:
                print('Converting nn input to actions.')
            for power, orders in self._nn_input_to_orders(action_n).items():
                self.game.set_orders(power, orders)
                info[power] = [self.action_list.index(self._order_to_action(x)) for x in orders]
        # Wrong action format
        else:
            raise Exception('wrong action input. Either do dict(str,str) based on the diplomacy package '
                            'or dict(str,list(numbers.Number)), with list having len = len(self.action_list)')
        if self.prints:
            print('Orders committed.')
        if render:
            self.render()

        self.game.process()
        # update state and observation
        new_state = self.game.get_state()
        obs = self.observation()
        # Check to see if all possible orders are contained within action list
        self._check_for_unaccounted_possible_actions()

        # Reward = new centers
        reward_n = [len(new_state['centers'][power]) - len(old_state['centers'][power]) for power in action_n.keys()]
        # Reward = curr centers
        #reward_n = [len(new_state['centers'][power]) for power in action_n.keys()]
        done = self.game.is_game_done
        return [obs for _ in action_n], reward_n, [done for _ in action_n], info

    def reset(self):
        self.game = diplomacy.Game()
        return self.observation()

    def render(self, mode='human', path=None):
        name = self.game.current_short_phase[1:5]+self.game.current_short_phase[0]+self.game.current_short_phase[5:]
        if path:
            return self.game.render(output_path=path + name + '.svg')
        elif self.render_path:
            return self.game.render(output_path=self.render_path + name + '.svg')
        else:
            return self.game.render()

    def observation(self):
        state = self.game.get_state()
        map = self.game.map
        nn_input = []
        # Armies
        for power in map.powers:
            nn_input.extend(['A ' + loc in state['units'][power] for loc in map.loc_name.values()])
        # Fleets
        for power in map.powers:
            nn_input.extend(['F ' + loc in state['units'][power] for loc in map.loc_name.values()])
        # Centers
        for power in map.powers:
            nn_input.extend([loc in state['centers'][power] for loc in map.scs])
        # Retreats
        # Army retreats
        for power in map.powers:
            nn_input.extend(['A ' + loc in state['retreats'][power] for loc in map.loc_name.values()])
        # Fleet retreats
        for power in map.powers:
            nn_input.extend(['F ' + loc in state['retreats'][power] for loc in map.loc_name.values()])
        return nn_input

    def _nn_input_to_orders(self, action_n):
        # convert numbers to action_list indexes
        possible_orders = self.game.get_all_possible_orders()
        orders = {}
        for power, numbers in action_n.items():
            power_orders = []
            for loc in self.game.get_orderable_locations(power):
                # dict of action_list index and value for actions on orderable locations
                if len(self.game.map.loc_coasts[loc]) > 1:
                    orderable_action_values = {x: y for x, y in dict(zip(self.action_list, numbers)).items()
                                               if self.action_loc_dict[x] == loc[0] or
                                               self.action_loc_dict[x] == loc[1] or
                                               self.action_loc_dict[x] == loc[2] or
                                               x == 'WAIVE'}
                else:
                    orderable_action_values = {x: y for x, y in dict(zip(self.action_list, numbers)).items()
                                           if self.action_loc_dict[x] == loc or x == 'WAIVE'}

                # reduce to only possible orders
                action_orders = {x: self.action_order_dict[x] for x in orderable_action_values.keys()}
                possible_order_values = \
                    {[y for y in v if y in possible_orders[loc]][0]: orderable_action_values[k]
                     for k, v in action_orders.items() if any(y in possible_orders[loc] for y in v)}

                # check all possible orders are covered
                if not set(possible_order_values.keys()) == set(possible_orders[loc]):
                    # Account for diplomacy package including available split-coast orders in all three split-locations
                    if len(self.game.map.loc_coasts[loc]) > 1 and len(possible_order_values.keys()) == 0:
                        continue
                    else:
                        raise Exception('Not all possible orders are covered by converted actions')

                # find order with most value
                max_order_value_index = [x for x, y in possible_order_values.items()
                                         if y == max(possible_order_values.values())][0]
                power_orders.append(max_order_value_index)
            orders[power] = power_orders
        return orders

    @staticmethod
    def _action_to_orders(action):
        action_variations = []
        # add two F/A to convoy and support actions
        if ' S ' in action:
            parts = action.split(' S ')
            action_variations.append('F ' + parts[0] + ' S F ' + parts[1])
            action_variations.append('A ' + parts[0] + ' S A ' + parts[1])
            action_variations.append('F ' + parts[0] + ' S A ' + parts[1])
            action_variations.append('A ' + parts[0] + ' S F ' + parts[1])
        elif ' C ' in action:
            parts = action.split(' C ')
            action_variations.append('F ' + parts[0] + ' C F ' + parts[1])
            action_variations.append('A ' + parts[0] + ' C A ' + parts[1])
            action_variations.append('F ' + parts[0] + ' C A ' + parts[1])
            action_variations.append('A ' + parts[0] + ' C F ' + parts[1])
        # add one F/A to hold and move actions
        elif ' - ' in action:
            action_variations.append('F ' + action)
            action_variations.append('A ' + action)
            # retreats
            action_variations.append('F ' + action.replace(' - ', ' R '))
            action_variations.append('A ' + action.replace(' - ', ' R '))
        elif action[-2:] == ' H' or action[-2:] == ' D':
            action_variations.append('F ' + action)
            action_variations.append('A ' + action)
        elif action[-2:] == ' B':
            action_variations.append(action)
        elif action == 'WAIVE':
            action_variations.append('WAIVE')
        else:
            raise Exception('Action type not covered')

        return action_variations

    #@staticmethod
    def _order_to_action(self, order):
        if order[-2:] == ' B':
            action = order
        elif ' R ' in order:
            action = order[2:].replace(' R ', ' - ')
        else:
            matches = re.split(r"^[AF] | [AF] ", order)
            matches = [x for x in matches if len(x) != 0]
            action = ' '.join(matches)
        if action not in self.action_list:
            raise Exception('Order to action translation unsuccessful')
        return action

    def _action_conversion(self, action, loc):
        order = [x for x in self.action_order_dict[action] if x in self.game.get_all_possible_orders()[loc]]
        if len(order) == 0:
            return None
        elif len(order) == 1:
            return order[0]
        else:
            raise Exception('Action could be converted into multiple orders')

    def _check_action_possibility(self, action, loc):
        return not self._action_conversion(action, loc) is None

    def _check_for_unaccounted_possible_actions(self):
        for loc, orders in self.game.get_all_possible_orders().items():
            for order in orders:
                neworder = order
                if self.game.phase_type == 'M':
                    neworder = neworder[2:]
                    neworder = neworder.replace(' F ', ' ').replace(' A ', ' ')
                elif self.game.phase_type == 'A':
                    if neworder == 'WAIVE':
                        neworder = 'WAIVE'
                    if neworder[-2:] == ' D':
                        neworder = neworder[2:]
                elif self.game.phase_type == 'R':
                    neworder = neworder[2:]
                    neworder = neworder.replace(' R ', ' - ')
                if neworder not in self.action_list:
                    raise Exception('could not find order {} in action list'.format(neworder))

    def _add_split_coasts_to_list(self, list):
        game = self.game
        result = [x for x in list]
        for x in list:
            if len(game.map.loc_coasts[x]) > 1:
                for loc in game.map.loc_coasts[x]:
                    if loc not in result:
                        result.append(loc)
        return result

    def _add_middle_split_coast_to_list(self, list):
        game = self.game
        result = [x for x in list]
        for n in result:
            if n in game.map.loc_coasts.keys():
                splits = game.map.loc_coasts[n]
                if len(splits) > 1:
                    for split in splits:
                        if split not in result and '/' not in split:
                            result.append(split)
        return result

    def _action_list(self):
        game = self.game
        # The action list consist of all theoretically possible orders without considering which units
        action_list = []
        action_loc_dict = {}
        action_order_dict = {}
        # convert location names to all caps
        locs = {}
        split_coasts = {x: y for x, y in game.map.loc_coasts.items() if len(y) > 1}
        for loc, neighbors in game.map.loc_abut.items():
            loc = str.upper(loc)
            neighbors = [str.upper(x) for x in neighbors]
            # Fix that coasts don't have connection to middle split-coast
            if game.map.area_type(loc) == 'COAST':
                for n in neighbors:
                    if n in split_coasts.keys():
                        for split in split_coasts[n]:
                            if split not in neighbors and '/' not in split:
                                neighbors.append(split)
            locs[loc] = neighbors

        convoys = {x: [] for x in locs if game.map.area_type(x) == 'COAST'}

        for loc, neighbors in locs.items():
            for action in [loc + ' H', loc + ' D']:
                action_list.append(action)
                action_loc_dict[action] = loc
                action_order_dict[action] = self._action_to_orders(action)
            # region Build Orders
            if loc in [z for c in [y for x in game.map.homes.values() for y in x] for z in game.map.loc_coasts[c]]:
                for action in ['A ' + loc + ' B', 'F ' + loc + ' B', 'WAIVE']:
                    action_list.append(action)
                    action_loc_dict[action] = loc
                    action_order_dict[action] = self._action_to_orders(action)
            # endregion

            # region Move Orders
            for n in neighbors:
                action = loc + ' - ' + n
                action_list.append(action)
                action_loc_dict[action] = loc
                action_order_dict[action] = self._action_to_orders(action)
            # endregion

            # region Convoy Move Order
            if '/' not in loc and game.map.area_type(loc) == 'COAST':
                unchecked_waters = [loc]
                checked_waters = []
                destinations = []
                while len(unchecked_waters) > 0:
                    water = unchecked_waters.pop()
                    checked_waters.append(water)
                    for n in self._add_middle_split_coast_to_list(locs[water]):
                        if game.map.area_type(n) == 'WATER' and n not in unchecked_waters and n not in checked_waters:
                            unchecked_waters.append(n)
                        if game.map.area_type(
                                n) == 'COAST' and n not in destinations and n != loc and '/' not in n:
                            destinations.append(n)
                for d in destinations:
                    action = loc + ' - ' + d + ' VIA'
                    action_list.append(action)
                    action_loc_dict[action] = loc
                    action_order_dict[action] = self._action_to_orders(action)
                    convoys[d].append(loc)
            # endregion

            # region Convoy Order
            if game.map.area_type(loc) == 'WATER':
                unchecked_waters = [x for x in neighbors if game.map.area_type(x) == 'WATER']
                checked_waters = []
                destinations = [x for x in neighbors if game.map.area_type(x) == 'COAST']
                # include split-coasts middles in destinations
                for d in destinations:
                    if d in split_coasts.keys():
                        for split in split_coasts[d]:
                            if split not in destinations and '/' not in split:
                                destinations.append(split)
                destinations = [x for x in destinations if '/' not in x]
                # keep searching for new coasts and waters
                while len(unchecked_waters) > 0:
                    water = unchecked_waters.pop()
                    checked_waters.append(water)
                    for n in locs[water]:
                        if game.map.area_type(n) == 'WATER' \
                                and n not in unchecked_waters and n not in checked_waters and n != loc:
                            unchecked_waters.append(n)
                        if game.map.area_type(n) == 'COAST' and n not in destinations:
                            # add split-coasts middles
                            if '/' in n:
                                for x in split_coasts[n]:
                                    if '/' not in x and x not in destinations:
                                        destinations.append(x)
                            else:
                                destinations.append(n)
                # make all destinations into from-to pairs
                for d in destinations:
                    for dd in destinations:
                        if d != dd:
                            action = loc + ' C ' + dd + ' - ' + d
                            action_list.append(action)
                            action_loc_dict[action] = loc
                            action_order_dict[action] = self._action_to_orders(action)
            # endregion

        # region Support Orders
        for loc, neighbors in locs.items():
            # handle non-adjacent split-coast supporting
            supp_neighbors = self._add_split_coasts_to_list(neighbors)

            for n in supp_neighbors:
                # Support Hold Orders
                action = loc + ' S ' + n
                action_list.append(action)
                action_loc_dict[action] = loc
                action_order_dict[action] = self._action_to_orders(action)
                # Support Move Orders
                # include convoy supporting
                from_locs = [x for x in locs[n] if x != loc]
                if game.map.area_type(n) == 'COAST' and '/' not in n:
                    for c in convoys[n]:
                        if c not in from_locs and c != loc and c != n:
                            from_locs.append(c)
                for nn in from_locs:
                    action = loc + ' S ' + nn + ' - ' + n
                    action_list.append(action)
                    action_loc_dict[action] = loc
                    action_order_dict[action] = self._action_to_orders(action)
        # endregion
        return action_list, action_loc_dict, action_order_dict
