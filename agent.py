from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.factory import Factory
from lux.unit import Unit
import numpy as np
import sys
#new below
from scipy.ndimage import distance_transform_cdt

class Hoard:
    def __init__(self, ship, hoard, score):
        self.target = ship
        self.hoard = hoard
        self.score = score
class My_Ship:
    def __init__(self, ship):
        self.ship = ship
        self.intentions = [0.0,0.0,0.0,0.0,0.0]
        self.run_away = False
        self.going_back = False
        self.target = None
        self.target_priority = 0




class Setting:
    def __init__(self):
        self.ore_multiplier_1 = 0.8
        self.second_factory_mult = 1/3
class Reference:
    def __init__(self):
        self.bid = None


class Agent():

    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.setting: Setting = Setting()
        self.reference: Reference = Reference()

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):

        def manhattan_distance(binary_mask):
            # Get the distance map from every pixel to the nearest positive pixel
            distance_map = distance_transform_cdt(binary_mask, metric='taxicab')
            return distance_map

        def find_good_spawns(obs): #only using proximity to nearest single ice and ore
            ice = obs["board"]["ice"]
            ore = obs["board"]["ore"]
            dist_ice = manhattan_distance(1 - ice)
            dist_ore = manhattan_distance(1 - ore)
            score = dist_ice + dist_ore * 0.9 # *0.9 just to make it tie breaker, 2 from ice and 3 from ore beats 3 from ice and 2 from ore
            valid_spawns = obs["board"]["valid_spawns_mask"] #not masking out any zeros because there are none; can't be distance zero to both ice and ore
            valid_spawns = np.ones(valid_spawns.shape) * valid_spawns + 0.0001
            score = score / valid_spawns
            sorted_indexes = np.argsort(score, axis=None)

            less = 100
            topN =  [np.unravel_index(index, (48, 48)) for index in sorted_indexes[:less]]
            dist_ice_2, dist_ore_2 = [], []
            for x,y in topN:
                closest_ice = min(
                    dist_ice[x-1,y-1],
                    dist_ice[x-1,y],
                    dist_ice[x-1,y+1],
                    dist_ice[x,y-1],
                    #dist_ice[x,y],
                    dist_ice[x,y+1],
                    dist_ice[x+1,y-1],
                    dist_ice[x+1,y],
                    dist_ice[x+1,y+1]
                )
                dist_ice_2.append(closest_ice)
                closest_ore = min(
                    dist_ore[x-1,y-1],
                    dist_ore[x-1,y],
                    dist_ore[x-1,y+1],
                    dist_ore[x,y-1],
                    #dist_ore[x,y],
                    dist_ore[x,y+1],
                    dist_ore[x+1,y-1],
                    dist_ore[x+1,y],
                    dist_ore[x+1,y+1]
                )
                dist_ore_2.append(closest_ore)
            scores = dist_ice_2 + np.array(dist_ore_2) * self.setting.ore_multiplier_1
            return topN, scores

        if step == 0:
            topN, scores = find_good_spawns(obs)
            sorted_indexes = np.argsort(scores)

            no_overlap = [] #list of locations sorted by score and non overlapping
            corresponding_scores = [] #list of corresponding scores
            for i in sorted_indexes:
                loc = np.array(topN[i])
                score = scores[i]
                if len(no_overlap) == 0:
                    no_overlap.append(loc)
                    corresponding_scores.append(score)
                    continue
                if any([np.sum(np.abs(loc-placed)) < 6 for placed in no_overlap]):
                    continue
                no_overlap.append(loc)
                corresponding_scores.append(score)
            _scores = corresponding_scores.copy()
            primitive_1st_vs_2nd = (_scores[0] - _scores[1]) + self.setting.second_factory_mult * (_scores[2] - _scores[3])
            self.reference.bid = round(primitive_1st_vs_2nd*-1) * 10
            return dict(faction="AlphaStrike", bid = self.reference.bid)
        
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            
            # initial factories
            n_factories = game_state.board.factories_per_team

            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                topN, scores = find_good_spawns(obs)
                sorted_indexes = np.argsort(scores)
                spawn_loc = np.array(topN[sorted_indexes[0]])
                if factories_to_place == n_factories: #this is our first factory
                    #the idea here is our first factory will have good proximity to ice/ore so will be
                    #   able to replenish resources easier
                    return dict(spawn=spawn_loc, metal=metal_left%150, water=water_left%150)
                else:
                    return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]



        game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                # previous ice mining code
                if unit.cargo.ice < 40:
                    ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= 40:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        return actions
