from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.factory import Factory
from lux.unit import Unit
import numpy as np
import sys
#new below
from scipy.ndimage import distance_transform_cdt
import itertools

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
        self.ore1_vs_ice1_mult = 1/2
        self.second_factory_mult = 1/3
        # commented out below line because territory metric should take care of this when neccessary
        #factory_border_buffer = 3
        self.N_second_ice_cutoff = 10 #more than N nsquares away from factory center and we don't care about the 2nd ice anymore
        self.second_ice_default_penalty = 1.3
        self.ice2_vs_ice1_mult = 1/(self.N_second_ice_cutoff * self.second_ice_default_penalty + 1) #this way if a factory doesnt have a cutoff and receives the default penalty then it still is only a tie breaker since <1class Reference:
        self.bid_normalizer = 2

class Agent():

    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.setting: Setting = Setting()
        self.bid: int = 0

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):

        def manhattan_distance(binary_mask):
            # Get the distance map from every pixel to the nearest positive pixel
            distance_map = distance_transform_cdt(binary_mask, metric='taxicab')
            return distance_map
        def generate_nsquares(point, n):
            x, y = point
            nsquares = []
            for i in range(-n, n+1):
                for j in range(-n, n+1):
                    if abs(i) == n or abs(j) == n:
                        nsquares.append((x+i, y+j))
            return nsquares
        def generate_pairings(lst):
            pairings = []
            for i, j in itertools.combinations(lst, 2):
                pairings.append((i,j))
            return pairings

        def score_spawn_locations(obs): #only using proximity to nearest single ice and ore
            board_ice = obs["board"]["ice"]
            board_ore = obs["board"]["ore"]
            dist_ice = manhattan_distance(1 - board_ice)
            dist_ore = manhattan_distance(1 - board_ore)
            valid_spawns = obs["board"]["valid_spawns_mask"] 
            factory_dist_ice, factory_dist_ore = [], []
            for x in range(48):
                for y in range(48):
                    if not valid_spawns[x][y]:
                        factory_dist_ice.append(999)
                        factory_dist_ore.append(999)
                        continue
                    closest_ice = min([dist_ice[_x, _y] for _x, _y in generate_nsquares([x,y], 1)])
                    factory_dist_ice.append(closest_ice)

                    closest_ore = min([dist_ore[_x, _y] for _x, _y in generate_nsquares([x,y], 1)])
                    factory_dist_ore.append(closest_ore)    

            scores = factory_dist_ice + np.array(factory_dist_ore) * self.setting.ore1_vs_ice1_mult
            #print(48*48, len(scores))
            sorted_indexes_1 = np.argsort(scores)

            less = 80 #number of locations we will do computations for 2nd closest ice
            factory_dist_ice2 = []
            for i in sorted_indexes_1[:less]:
                x,y = np.unravel_index(i, (48, 48))
                dist_ice1 = dist_ice[x][y]
                dist_ice2 = None
                ltoet_n_away = [] # ltoet <=
                for n in range(0, self.setting.N_second_ice_cutoff): #if second ice is more than N nsquares away it kinda doesnt matter, can just add some constant penalty
                    if n < 2: #these are factory tiles
                        continue
                    for _x,_y in generate_nsquares([x,y], n):
                        if (_x < 0 or _y < 0) or (_x >= 48 or _y >= 48): 
                            continue
                        if board_ice[_x][_y] == 1:
                            ltoet_n_away.append((_x,_y))
                    if len(ltoet_n_away) >= 2:
                        for ice1, ice2 in generate_pairings(ltoet_n_away):
                            if np.sum(np.abs(np.array(ice1)-np.array(ice2))) > 2:
                                if np.sum(np.abs(np.array(ice1)-np.array([x,y]))) == dist_ice1: #our first ice must be the closest ice(s)
                                    dist_ice2 = np.sum(np.abs( np.array(ice2)-np.array([x,y]) ))
                                    assert dist_ice1 <= dist_ice2
                if not dist_ice2:
                    dist_ice2 = self.setting.N_second_ice_cutoff * self.setting.second_ice_default_penalty
                factory_dist_ice2.append(dist_ice2)
            for i in sorted_indexes_1[less:]:
                dist_ice2 = self.setting.N_second_ice_cutoff * self.setting.second_ice_default_penalty
                factory_dist_ice2.append(dist_ice2)
            assert len(factory_dist_ice2) == len(scores)

            for __ in range(48*48):
                i = sorted_indexes_1[__] #because we indexed factory_dist_ice2 differently so we dont have to compute for all 48*48
                #print(scores[i], factory_dist_ice2[_])
                scores[i] += factory_dist_ice2[__] * self.setting.ice2_vs_ice1_mult
            return scores

        if step == 0:
            scores = score_spawn_locations(obs)
            sorted_indexes_2 = np.argsort(scores)
            no_overlap, nol_scores = [], [] 
            #corresponding_si1_index = [] #corresponding to sorted_indexes_1
            count = -1
            #while len(no_overlap) < 2*n_factories:
            while len(no_overlap) < 4:
                count += 1
                i = sorted_indexes_2[count]
                loc = np.unravel_index(i, (48, 48))
                score = scores[i]
                if len(no_overlap) == 0:
                    no_overlap.append(loc)
                    nol_scores.append(score)
                    flattened_i = loc[0]*48 + loc[1]
                    #corresponding_si1_index.append(list(sorted_indexes_1).index(flattened_i))
                    continue
                if any([np.sum(np.abs(np.array(loc)-np.array(factory))) < 6 for factory in no_overlap]):
                    scores[i] = 999 #this is only here to show results in the print loop below, once factories start getting placed can mess things up
                    continue
                no_overlap.append(loc)
                nol_scores.append(score)
                flattened_i = loc[0]*48 + loc[1]
                #corresponding_si1_index.append(list(sorted_indexes_1).index(flattened_i))

            _1st_vs_2nd = (nol_scores[0] - nol_scores[1]) + self.setting.second_factory_mult * (nol_scores[2] - nol_scores[3])
            _1st_vs_2nd *= -1 * self.setting.bid_normalizer
            bid = None
            if abs(_1st_vs_2nd) < 0.5:
                bid = 0
            elif 0.5 <= abs(_1st_vs_2nd) and abs(_1st_vs_2nd) < 1:
                bid = round(_1st_vs_2nd*10) - 4 #bid 1 if 0.5, bid 6 if 0.99
            elif 1 <= abs(_1st_vs_2nd) and abs(_1st_vs_2nd) < 1.6: #we bid non-ten values up to sixteen, then its just 20, 30, 40
                bid = round(_1st_vs_2nd*10)
            elif 1.6 <= abs(_1st_vs_2nd):
                bid = round(_1st_vs_2nd) * 10
            self.bid = bid
            return dict(faction="AlphaStrike", bid = bid)
        
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
                scores = score_spawn_locations(obs)
                sorted_indexes_2 = np.argsort(scores)
                spawn_loc = np.unravel_index(sorted_indexes_2[0], (48, 48))
                if factories_to_place == n_factories: #this is our first factory
                    #the idea here is our first factory will have good proximity to ice/ore so will be
                    #   able to replenish resources easier
                    if self.bid == 0:
                        return dict(spawn=spawn_loc, metal=150, water=150)
                    else:
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
