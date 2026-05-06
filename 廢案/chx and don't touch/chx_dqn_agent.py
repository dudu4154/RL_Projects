import numpy as np
from pysc2.lib import actions, features, units

# =================================================================
# 1. 定義動作空間 (Action Space)
# =================================================================
ACTION_MAP = {
    0:  actions.FUNCTIONS.Build_SupplyDepot_screen,
    1:  actions.FUNCTIONS.Build_Barracks_screen,
    2:  actions.FUNCTIONS.Build_Factory_screen,
    3:  actions.FUNCTIONS.Build_Starport_screen,
    4:  actions.FUNCTIONS.Build_FusionCore_screen,
    5:  actions.FUNCTIONS.Build_CommandCenter_screen,
    6:  actions.FUNCTIONS.Build_EngineeringBay_screen,
    7:  actions.FUNCTIONS.Build_SensorTower_screen,
    8:  actions.FUNCTIONS.Build_GhostAcademy_screen,
    9:  actions.FUNCTIONS.Build_Armory_screen,
    10: actions.FUNCTIONS.Build_Refinery_screen,
    11: actions.FUNCTIONS.Build_MissileTurret_screen,
    12: actions.FUNCTIONS.Build_Bunker_screen,
    13: actions.FUNCTIONS.Train_SCV_quick,
    14: actions.FUNCTIONS.no_op, # 呼叫礦騾先關閉
    15: actions.FUNCTIONS.Train_Marine_quick,
    16: actions.FUNCTIONS.Train_Reaper_quick,
    17: actions.FUNCTIONS.Train_Marauder_quick,
    18: actions.FUNCTIONS.Train_Ghost_quick,
    19: actions.FUNCTIONS.Train_Hellion_quick,
    20: actions.FUNCTIONS.no_op, # 戰狼先關閉
    21: actions.FUNCTIONS.Train_WidowMine_quick,
    22: actions.FUNCTIONS.Train_SiegeTank_quick,
    23: actions.FUNCTIONS.Train_Cyclone_quick,
    24: actions.FUNCTIONS.Train_Thor_quick,
    25: actions.FUNCTIONS.Train_VikingFighter_quick,
    26: actions.FUNCTIONS.Train_Medivac_quick,
    27: actions.FUNCTIONS.Train_Liberator_quick,
    28: actions.FUNCTIONS.Train_Raven_quick,
    29: actions.FUNCTIONS.Train_Battlecruiser_quick,
    30: actions.FUNCTIONS.Train_Banshee_quick,
    31: actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick, # 補給站下降 (快速版)
    32: actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick, # 補給站上升 (快速版)
}
N_ACTIONS = len(ACTION_MAP)

# =================================================================
# 2. 定義狀態特徵 (State Features)
# =================================================================
def get_state(obs):
    player = obs.observation.player
    minerals = player.minerals / 1000.0
    vespene = player.vespene / 1000.0
    food_used = player.food_used / 200.0
    food_cap = player.food_cap / 200.0
    
    unit_type = obs.observation.feature_screen.unit_type
    
    # 建築
    cc_count = np.sum(unit_type == units.Terran.CommandCenter) / 5.0
    refinery_count = np.sum(unit_type == units.Terran.Refinery) / 10.0
    scv_count = np.sum(unit_type == units.Terran.SCV) / 50.0
    depot_count = np.sum(unit_type == units.Terran.SupplyDepot) / 20.0
    barracks_count = np.sum(unit_type == units.Terran.Barracks) / 10.0
    factory_count = np.sum(unit_type == units.Terran.Factory) / 5.0
    starport_count = np.sum(unit_type == units.Terran.Starport) / 5.0
    eng_bay_count = np.sum(unit_type == units.Terran.EngineeringBay) / 2.0
    armory_count = np.sum(unit_type == units.Terran.Armory) / 2.0
    fusion_core_count = np.sum(unit_type == units.Terran.FusionCore) / 1.0
    bunker_count = np.sum(unit_type == units.Terran.Bunker) / 5.0
    turret_count = np.sum(unit_type == units.Terran.MissileTurret) / 10.0
    sensor_tower_count = np.sum(unit_type == units.Terran.SensorTower) / 2.0
    ghost_academy_count = np.sum(unit_type == units.Terran.GhostAcademy) / 1.0

    # 兵力
    marine_count = np.sum(unit_type == units.Terran.Marine) / 50.0
    marauder_count = np.sum(unit_type == units.Terran.Marauder) / 20.0
    reaper_count = np.sum(unit_type == units.Terran.Reaper) / 10.0
    ghost_count = np.sum(unit_type == units.Terran.Ghost) / 10.0
    hellion_count = np.sum(unit_type == units.Terran.Hellion) / 20.0
    tank_count = np.sum(unit_type == units.Terran.SiegeTank) / 10.0
    thor_count = np.sum(unit_type == units.Terran.Thor) / 5.0
    viking_count = np.sum(unit_type == units.Terran.VikingFighter) / 10.0
    medivac_count = np.sum(unit_type == units.Terran.Medivac) / 10.0
    banshee_count = np.sum(unit_type == units.Terran.Banshee) / 10.0
    raven_count = np.sum(unit_type == units.Terran.Raven) / 5.0
    bc_count = np.sum(unit_type == units.Terran.Battlecruiser) / 5.0

    current_state = np.array([
        minerals, vespene, food_used, food_cap,
        cc_count, refinery_count, scv_count, depot_count, barracks_count,
        factory_count, starport_count, eng_bay_count, armory_count, fusion_core_count,
        bunker_count, turret_count, sensor_tower_count, ghost_academy_count,
        marine_count, marauder_count, reaper_count, ghost_count, hellion_count,
        tank_count, thor_count, viking_count, medivac_count, banshee_count,
        raven_count, bc_count
    ])
    return current_state

# =================================================================
# 3. 動作遮罩 (Action Masking)
# =================================================================
def check_valid_actions(obs, action_map):
    valid_actions = np.zeros(len(action_map))
    
    player = obs.observation.player
    minerals = player.minerals
    vespene = player.vespene
    food_left = player.food_cap - player.food_used
    
    unit_type = obs.observation.feature_screen.unit_type
    has_cc = np.sum(unit_type == units.Terran.CommandCenter) > 0
    has_depot = np.sum(unit_type == units.Terran.SupplyDepot) > 0
    has_barracks = np.sum(unit_type == units.Terran.Barracks) > 0
    has_factory = np.sum(unit_type == units.Terran.Factory) > 0
    has_starport = np.sum(unit_type == units.Terran.Starport) > 0
    has_fusion_core = np.sum(unit_type == units.Terran.FusionCore) > 0
    
    for action_id, func_id in action_map.items():
        if func_id.id not in obs.observation.available_actions:
            continue
            
        if action_id == 0: # 造補給站
            if minerals >= 100: valid_actions[action_id] = 1
        elif action_id == 1: # 造兵營
            if minerals >= 150 and has_depot: valid_actions[action_id] = 1
        elif action_id == 2: # 造軍工廠
            if minerals >= 150 and vespene >= 100 and has_barracks: valid_actions[action_id] = 1
        elif action_id == 3: # 造星際港
            if minerals >= 150 and vespene >= 100 and has_factory: valid_actions[action_id] = 1
        elif action_id == 4: # 造核融合核心
            if minerals >= 150 and vespene >= 150 and has_starport: valid_actions[action_id] = 1
        elif action_id == 5: # 造指揮中心
            if minerals >= 400: valid_actions[action_id] = 1
        elif action_id == 10: # 造瓦斯
            if minerals >= 75: valid_actions[action_id] = 1
        elif action_id == 13: # 造工兵
            if minerals >= 50 and food_left >= 1 and has_cc: valid_actions[action_id] = 1
        elif action_id == 15: # 造陸戰隊
            if minerals >= 50 and food_left >= 1 and has_barracks: valid_actions[action_id] = 1
        elif action_id == 22: # 造坦克
            if minerals >= 150 and vespene >= 125 and food_left >= 3 and has_factory: valid_actions[action_id] = 1
        elif action_id == 29: # 造戰巡艦
            if minerals >= 400 and vespene >= 300 and food_left >= 6 and has_starport and has_fusion_core: valid_actions[action_id] = 1
        else:
             if func_id.id in obs.observation.available_actions:
                 valid_actions[action_id] = 1

    return valid_actions