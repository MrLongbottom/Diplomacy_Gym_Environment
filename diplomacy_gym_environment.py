import re
from itertools import product
from numbers import Number
import diplomacy
import gym
import numpy as np
import collections
from gym import spaces
from diplomacy.utils.export import to_saved_game_format


class DiplomacyEnvironment(gym.Env):

    def __init__(self):
        super(DiplomacyEnvironment, self).__init__()
        self.game = diplomacy.Game()
        # TODO remove duplicates from action list involving convoys to / from spain
        self.action_list = self._action_list()
        self.reward_range = (-self.game.win, self.game.win)
        high = np.array([1.0 for _ in self.action_list])
        self.action_space = spaces.Box(high=high, low=-high, dtype=np.float32)
        self.observation_space = spaces.MultiBinary(len(self.observation()))

    def step(self, action_n):
        old_state = self.game.get_state()

        # diplomacy package string orders input
        if isinstance(list(action_n.values())[0][0], str):
            for power, orders in action_n.items():
                self.game.set_orders(power, orders)

        # nn list(numbers) input
        elif isinstance(list(action_n.values())[0][0], Number):
            # convert numbers to action_list indexes
            possible_orders = self.game.get_all_possible_orders()
            for power, numbers in action_n.items():
                power_orders = []
                for loc in self.game.get_orderable_locations(power):
                    # dict of action_list index and nn number for actions on current location
                    loc_numbers = {i: action_n[power][i] for i in range(len(self.action_list))
                                   if self.action_list[i][0] == loc}
                    # reduce to only possible orders
                    loc_numbers = {x: y for x, y in loc_numbers.items() if self._check_action_possibility(self.action_list[x])}
                    # check all possible orders are covered
                    actions = [self._action_conversion(self.action_list[x]) for x in loc_numbers.keys()]
                    if not set(actions) == set(possible_orders[loc]):
                        if len(actions) > len(possible_orders[loc]):
                            loc_numbers = {x: y for x, y in loc_numbers.items() if self.action_list[x] in possible_orders[loc]}
                        elif len(self.game.map.loc_coasts[loc]) > 1 and len(actions) == 0:
                            continue
                        else:
                            test = [item for item, count in collections.Counter(actions).items() if count > 1]
                            raise Exception('Not all possible orders are covered by converted actions')
                    max_action_index = [x for x, y in loc_numbers.items() if y == max(loc_numbers.values())]
                    if len(max_action_index) == 1:
                        max_action_index = max_action_index[0]
                    else:
                        raise Exception("Couldn't find best action")
                    power_orders.append(self._action_conversion(self.action_list[max_action_index]))
                self.game.set_orders(power, power_orders)
        else:
            raise Exception('wrong action imput. Either do dict(str,str) based on the diplomacy package '
                            'or dict(str,list(numbers.Number)), with list having len = len(self.action_list)')
        self.game.process()
        # update state and observation
        new_state = self.game.get_state()
        obs = self.observation()
        # Check to see if all possible orders are contained within action list
        self._check_for_unaccounted_possible_actions()

        reward_n = [len(new_state['centers'][power]) - len(old_state['centers'][power]) for power in action_n.keys()]
        done = self.game.is_game_done
        return [obs for _ in action_n], reward_n, [done for _ in action_n], [{} for _ in action_n]

    def reset(self):
        self.game = diplomacy.Game()
        return self.observation()

    def render(self, mode='human', path=None):
        if path:
            return self.game.render(output_path=path)
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

    def _action_conversion(self, action):
        # TODO make into a saved dictionary rather than calculating every time
        # TODO actually check game.map.units rather than test combinations with possible orders
        action_variations = []
        possible_actions = self.game.get_all_possible_orders()[action[0]]
        if self.game.phase_type == 'M':
            # add two F/A to convoy and support actions
            if ' S ' in action[1]:
                parts = action[1].split(' S ')
                action_variations.append('F ' + parts[0] + ' S F ' + parts[1])
                action_variations.append('A ' + parts[0] + ' S A ' + parts[1])
                action_variations.append('F ' + parts[0] + ' S A ' + parts[1])
                action_variations.append('A ' + parts[0] + ' S F ' + parts[1])
            elif ' C ' in action[1]:
                parts = action[1].split(' C ')
                action_variations.append('F ' + parts[0] + ' S F ' + parts[1])
                action_variations.append('A ' + parts[0] + ' S A ' + parts[1])
                action_variations.append('F ' + parts[0] + ' S A ' + parts[1])
                action_variations.append('A ' + parts[0] + ' S F ' + parts[1])
            # add one F/A to hold and move actions
            elif ' - ' in action[1] or action[1][-2:] == ' H':
                action_variations.append('F ' + action[1])
                action_variations.append('A ' + action[1])
            else:
                return None
        elif self.game.phase_type == 'A':
            if action[1][:2] == 'W ':
                action_variations.append('WAIVE')
            elif action[1][-2:] == ' D':
                action_variations.append('F ' + action[1])
                action_variations.append('A ' + action[1])
            elif action[1][-2:] == ' B':
                action_variations.append(action[1])
            else:
                return None
        elif self.game.phase_type == 'R':
            if ' - ' in action[1] and ' C ' not in action[1] and ' S ' not in action[1]:
                action_variations.append('F ' + action[1].replace(' - ', ' R '))
                action_variations.append('A ' + action[1].replace(' - ', ' R '))
            else:
                return None
        else:
            raise Exception('Unknown phase: ' + self.game.phase_type)

        possible = [x for x in action_variations if x in possible_actions]
        if len(possible) > 1:
            raise Exception('Multiple possible orders from action')
        elif len(possible) == 1:
            return possible[0]
        else:
            return None

    def _check_action_possibility(self, action):
        return not self._action_conversion(action) is None

    def _check_for_unaccounted_possible_actions(self):
        for loc, orders in self.game.get_all_possible_orders().items():
            for order in orders:
                neworder = order
                if self.game.phase_type == 'M':
                    neworder = neworder[2:]
                    neworder = neworder.replace(' F ', ' ').replace(' A ', ' ')
                elif self.game.phase_type == 'A':
                    if neworder == 'WAIVE':
                        neworder = 'W ' + loc + ' B'
                    if neworder[-2:] == ' D':
                        neworder = neworder[2:]
                elif self.game.phase_type == 'R':
                    neworder = neworder[2:]
                    neworder = neworder.replace(' R ', ' - ')
                if neworder not in [y for x,y in self.action_list]:
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
            action_list.append((loc, loc + ' H'))
            action_list.append((loc, loc + ' D'))

            if loc in [z for c in [y for x in game.map.homes.values() for y in x] for z in game.map.loc_coasts[c]]:
                action_list.append((loc, 'A ' + loc + ' B'))
                action_list.append((loc, 'F ' + loc + ' B'))
                action_list.append((loc, 'W ' + loc + ' B'))

            # region Move Orders
            for n in neighbors:
                action_list.append((loc, loc + ' - ' + n))
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
                    action_list.append((loc, loc + ' - ' + d + ' VIA'))
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
                            action_list.append((loc, loc + ' C ' + dd + ' - ' + d))
            # endregion

        # region Support Orders
        for loc, neighbors in locs.items():
            # handle non-adjacent split-coast supporting
            supp_neighbors = self._add_split_coasts_to_list(neighbors)

            for n in supp_neighbors:
                # Support Hold Orders
                action_list.append((loc, loc + ' S ' + n))
                # Support Move Orders
                # include convoy supporting
                from_locs = [x for x in locs[n] if x != loc]
                if game.map.area_type(n) == 'COAST' and '/' not in n:
                    for c in convoys[n]:
                        if c not in from_locs and c != loc and c != n:
                            from_locs.append(c)
                for nn in from_locs:
                    action_list.append((loc, loc + ' S ' + nn + ' - ' + n))
        # endregion
        return action_list
