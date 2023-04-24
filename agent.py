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

        #related to spawn locations valuation
        self.ore1_vs_ice1_mult = 1/2
        self.N_second_ice_cutoff = 10 #more than N nsquares away from factory center and we don't care about the 2nd ice anymore
        self.ice2_vs_ice1_mult = self.ore1_vs_ice1_mult * 1/(self.N_second_ice_cutoff + 1) #this way if a factory doesnt have a cutoff and receives the default penalty then it still is only a tie breaker since <1class Reference:
        self.territory_vs_iceore_mult = 1
        self.successive_factory_mult = 2/3

        #bid stuff
        self.discount_greed = 5
        self.max_encountered_bid = 40
        self.n_factories_importance_exp = 0.5
        self.magic_mult = 30
        
def board_manhattan_distance(binary_mask):
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
def bipartite_pairings(L, R):
    pairings = set()
    for l in L:
        for r in R:
            pairings.add((l, r))
    return pairings
def manh(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
def nearest_point(point, points):
    nearest = points[0]
    nearest_distance = manh(point, nearest)
    for p in points[1:]:
        distance = manh(point, p)
        if distance < nearest_distance:
            nearest = p
            nearest_distance = distance
    return nearest
def n_dist_to_nearest(A, n):
    B = set()
    for x in range(min(p[0] for p in A) - n, max(p[0] for p in A) + n+1):
        for y in range(min(p[1] for p in A) - n, max(p[1] for p in A) + n+1):
            point = (x, y)
            nearest = nearest_point(point, A)
            if manh(point, nearest) == n:
                B.add(point)
    return B

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.setting: Setting = Setting()
        self.n_factories_initial = None
        self.bid: int = 0

    #if getting runtime error in factory placement, decrease the 'less' param
    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        self.n_factories_initial = game_state.board.factories_per_team
        
        if len(game_state.teams) == 0:
            self.all_factories_to_place = self.n_factories_initial*2
        else:
            self.all_factories_to_place = game_state.teams[self.player].factories_to_place + game_state.teams[self.opp_player].factories_to_place
        
        def ice1ore1_scores(obs):
            board_ice = obs["board"]["ice"]
            board_ore = obs["board"]["ore"]
            dist_ice = board_manhattan_distance(1 - board_ice)
            dist_ore = board_manhattan_distance(1 - board_ore)
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
            sorted_indexes_1 = np.argsort(scores)
            return scores, sorted_indexes_1
        def with_ice2_scores(obs, scores, sorted_indexes_1, less = 80):
            board_ice = obs["board"]["ice"]
            board_ore = obs["board"]["ore"]
            dist_ice = board_manhattan_distance(1 - board_ice)
            dist_ore = board_manhattan_distance(1 - board_ore)
            factory_dist_ice2 = []
            for i in sorted_indexes_1[:less]:
                x,y = np.unravel_index(i, (48, 48))
                dist_ice1 = dist_ice[x][y]
                dist_ice2 = None
                factory_tiles = generate_nsquares([x,y], 1)
                less_than_n_away = []
                for n in range(0, self.setting.N_second_ice_cutoff): #if second ice is more than N nsquares away it kinda doesnt matter, can just add some constant penalty
                    if dist_ice2:
                        break
                    n_away = []
                    if n < 2: #these are factory tiles
                        continue
                    if n == 2: #edge case where ltoet_n_away is empty
                        for _x,_y in n_dist_to_nearest(factory_tiles, n):
                            if (_x < 0 or _y < 0) or (_x >= 48 or _y >= 48): 
                                continue
                            if board_ice[_x][_y] == 1:
                                n_away.append((_x,_y))  
                        if len(n_away) >= 2:
                            for ice1, ice2 in generate_pairings(n_away):
                                if manh(ice1, ice2) > 2:
                                    dist_ice2 = manh(ice2, (x,y))
                        for loc in n_away:
                            less_than_n_away.append(loc)
                    else:            
                        for _x,_y in n_dist_to_nearest(factory_tiles, n):
                            if (_x < 0 or _y < 0) or (_x >= 48 or _y >= 48): 
                                continue
                            if board_ice[_x][_y] == 1:
                                n_away.append((_x,_y))  
                            if len(n_away) >= 1:
                                for ice1, ice2 in bipartite_pairings(less_than_n_away, n_away):
                                    if manh(ice1, ice2) > 2:
                                        dist_ice2 = manh(ice2, (x,y))
                            for loc in n_away:
                                less_than_n_away.append(loc)   
                #finished going from 0 to cutoff
                if not dist_ice2:
                    dist_ice2 = self.setting.N_second_ice_cutoff+1
                factory_dist_ice2.append(dist_ice2)
            for i in sorted_indexes_1[less:]:
                dist_ice2 = self.setting.N_second_ice_cutoff+1
                factory_dist_ice2.append(dist_ice2)
            assert len(factory_dist_ice2) == len(scores)
            #print(factory_dist_ice[:20])
            #print(factory_dist_ice2[:20])

            #update scores to reflect distances to ice2
            for __ in range(48*48):
                #break
                i = sorted_indexes_1[__] #because we indexed factory_dist_ice2 differently so we dont have to compute for all 48*48
                #print(scores[i], factory_dist_ice2[__])
                scores[i] += factory_dist_ice2[__] * self.setting.ice2_vs_ice1_mult
            #sorted_indexes_2 to reflect new scores accounting for ice2
            sorted_indexes_2 = np.argsort(scores) 
            return scores, sorted_indexes_2
        def no_overlap(scores, sorted_indexes_2, sorted_indexes_1, topN):
            nol_locations, nol_scores = [], [] 
            corresponding_si1_index = [] #corresponding to sorted_indexes_1, just for bookeeping, used to see that ~80 is safe for 'less' parameter above
            count = -1
            while len(nol_locations) < topN:
                count += 1
                i = sorted_indexes_2[count]
                loc = np.unravel_index(i, (48, 48))
                score = scores[i]
                if len(nol_locations) == 0:
                    nol_locations.append(loc)
                    nol_scores.append(score)
                    flattened_i = loc[0]*48 + loc[1]
                    corresponding_si1_index.append(list(sorted_indexes_1).index(flattened_i))
                    continue
                if any([np.sum(np.abs(np.array(loc)-np.array(factory))) < 6 for factory in nol_locations]):
                    scores[i] = 999 #this is only here to show results in the print loop below, once factories start getting placed can mess things up
                    continue
                nol_locations.append(loc)
                nol_scores.append(score)
                flattened_i = loc[0]*48 + loc[1]
                corresponding_si1_index.append(list(sorted_indexes_1).index(flattened_i))
            for i in sorted_indexes_2[:count+1]:
                #print(np.unravel_index(i, (48, 48)), factory_dist_ice[i], factory_dist_ore[i], scores[i])
                break
            return nol_locations, nol_scores, corresponding_si1_index
        def spawn_locations(obs): #only using proximity to nearest single ice and ore
            scores, sorted_indexes_1 = ice1ore1_scores(obs)
            scores, sorted_indexes_2 = with_ice2_scores(obs, scores, sorted_indexes_1, 200)
            nol_locations, nol_scores, corresponding_si1_index = no_overlap(scores, sorted_indexes_2, sorted_indexes_1, self.all_factories_to_place)
            return nol_locations, nol_scores
        def nol_to_1stvs2nd(nol_scores): #only called at step == 0
            assert len(nol_scores) == self.n_factories_initial * 2
            _1st_vs_2nd = 0
            for i in range(int(len(nol_scores)/2)):
                _1st_vs_2nd += (nol_scores[i] - nol_scores[i+1]) * self.setting.successive_factory_mult**i
            return _1st_vs_2nd #more negative = more of an advantage to go 1st



        if step == 0:
            #the fair bid value would be the amount F of water+metal where starting with F less water
            #       (and knowing we have N factories)
            #   and F less metal, and ceil(F/10) less initial light robots is exactly compensated by the 
            #   difference in going first vs second. this happens when probability of winning match with F 
            #   less etc and going first is the same as probability of winning match without paying any bid,
            #   but going second. however, we would never want to actually bid our idea of fair bid value,
            #   because this move has ev zero. instead we want to bid less than our fair value estimation
            #   and hope that we get to play first at a discount. however if we are too greedy then the
            #   opponent maybe be able to play first still at a discount, just at less of a discount. 
            #   so we don't want to be too greedy. a practical approach without doing extensive tuning
            #   may be bidding where F%10 > 0: F - (F % 10) + max( (F % 10) - 5, 0 )
            #                  where F%10 = 0: F - 10 + 5
            #   in this situation if we think fair value is 19, we'll bid 14, if we think is 20 we'll bid 15
            #   if we think is 13 we'll bid 10. if we think is 8 we'll bid 3.
            #   in the last case, we win if opponent bids > 8, we lose if opponent bids 4-7, and we win if
            #   opponent bids <3. if opponent bids 8 or 3 we can say its a tie

            #another thing: of course we don't have this magic fair value function, so have to do a very
            #rough estimate of it using _1st_vs_2nd as the input param

            nol_locations, nol_scores = spawn_locations(obs)
            _1st_vs_2nd = nol_to_1stvs2nd(nol_scores)

            def magic_f(_1st_vs_2nd): #turns _1st_vs_2nd difference into our estimate of fair value. initially we will just scale _1st_vs_2nd s.t. our ratio of games where we bid zero or bid high values seems reasonable
                return _1st_vs_2nd * self.setting.magic_mult * self.n_factories_initial**self.setting.n_factories_importance_exp
            def fairval_to_bid(fair_val): #returns bid amount given a fair value amount
                bid = 0
                if fair_val%10 > 0:
                    bid = fair_val - fair_val%10 + max(0, fair_val%10 - self.setting.discount_greed)
                elif fair_val%10 == 0:
                    bid = max(0, fair_val - self.setting.discount_greed)
                else:
                    None
                return min(self.setting.max_encountered_bid + 1, round(bid))

            fair_val = magic_f(_1st_vs_2nd)
            self.bid = fairval_to_bid(fair_val)
            return dict(faction="AlphaStrike", bid = self.bid)
        
        else:

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            my_factories_to_place = game_state.teams[self.player].factories_to_place

            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if my_factories_to_place > 0 and my_turn_to_place:
                scores, sorted_indexes_1 = ice1ore1_scores(obs)
                scores, sorted_indexes_2 = with_ice2_scores(obs, scores, sorted_indexes_1, 200)
                spawn_loc = np.unravel_index(sorted_indexes_2[0], (48, 48))
                if my_factories_to_place == self.n_factories_initial: #this is our first factory
                    #the idea here is our first factory will have good proximity to ice/ore so will be able to replenish resources easier
                    if self.bid == 0: #so we don't spawn first factory with zero metal and water
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
