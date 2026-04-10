import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

class ZergEnemy96Bot(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.base_top_left = None

    def _get_units(self, obs, unit_type, alliance=1):
        return [u for u in obs.observation.raw_units if u.unit_type == unit_type and u.alliance == alliance]

    def _get_build_pos(self, h_unit, minerals, offset=7):
        """
        核心邏輯：計算礦區的反方向座標
        h_unit: 主堡物件
        minerals: 中立礦石列表
        offset: 偏移距離 (6~8 適合開局菌毯)
        """
        if not minerals:
            return (h_unit.x + offset, h_unit.y + offset)
        
        # 1. 計算所有礦石的中心點
        mx = np.mean([m.x for m in minerals])
        my = np.mean([m.y for m in minerals])
        
        # 2. 計算從 礦區中心 指向 主堡 的向量
        vec_x = h_unit.x - mx
        vec_y = h_unit.y - my
        
        # 3. 正規化向量並加上偏移量
        dist = np.sqrt(vec_x**2 + vec_y**2)
        if dist == 0: return (h_unit.x + offset, h_unit.y)
        
        # 往礦區的反方向推 offset 距離
        tx = h_unit.x + (vec_x / dist) * offset
        ty = h_unit.y + (vec_y / dist) * offset
        
        return (tx, ty)

    def step(self, obs):
        super().step(obs)
        obs_data = obs.observation
        player = obs_data.player
        
        # 0. 基礎資訊
        hatcheries = self._get_units(obs, units.Zerg.Hatchery)
        if not hatcheries: return actions.RAW_FUNCTIONS.no_op()
        main_h = hatcheries[0]
        
        # 獲取附近礦石用來計算建築位置
        all_minerals = self._get_units(obs, units.Neutral.MineralField, alliance=0)
        # 只取靠近我方基地的礦石 (距離 < 15)
        nearby_minerals = [m for m in all_minerals if np.sqrt((m.x-main_h.x)**2 + (m.y-main_h.y)**2) < 15]

        # 掃描單位狀態
        drones = self._get_units(obs, units.Zerg.Drone)
        larvae = self._get_units(obs, units.Zerg.Larva)
        roaches = self._get_units(obs, units.Zerg.Roach)
        pool = self._get_units(obs, units.Zerg.SpawningPool)
        warren = self._get_units(obs, units.Zerg.RoachWarren)
        extractors = self._get_units(obs, units.Zerg.Extractor)
        overlords = self._get_units(obs, units.Zerg.Overlord)

        drone_count = len(drones)
        # 正在孵化中的工蜂
        pending_drones = sum(1 for u in larvae if u.order_id_0 == 482)
        total_drones = drone_count + pending_drones

        # ---------------------------------------------------------
        # 優先級 1：進攻 (8 隻蟑螂)
        # ---------------------------------------------------------
        if len(roaches) >= 8:
            # 簡單判定敵方位置：朝地圖對角線 A 過去
            target = (160, 160) if main_h.x < 100 else (40, 40)
            return actions.RAW_FUNCTIONS.Attack_pt("now", [u.tag for u in roaches], target)

        # ---------------------------------------------------------
        # 優先級 2：8 蟑螂流程 (嚴格人口對齊)
        # ---------------------------------------------------------
        
        # [13 人口] 補王蟲
        if total_drones >= 13 and len(overlords) < 2:
            if player.minerals >= 100 and larvae:
                return actions.RAW_FUNCTIONS.Train_Overlord_quick("now", larvae[0].tag)

        # [13 人口] 蓋水池 - 使用「反向偏移」邏輯
        if total_drones >= 13 and not pool and player.minerals >= 200:
            pos = self._get_build_pos(main_h, nearby_minerals, offset=7.5)
            return actions.RAW_FUNCTIONS.Build_SpawningPool_pt("now", drones[0].tag, pos)

        # [12 人口] 蓋瓦斯
        if pool and not extractors and player.minerals >= 25:
            geysers = self._get_units(obs, units.Neutral.VespeneGeyser, alliance=0)
            if geysers:
                # 找離主堡最近的噴泉
                target_g = geysers[np.argmin([np.sqrt((g.x-main_h.x)**2 + (g.y-main_h.y)**2) for g in geysers])]
                return actions.RAW_FUNCTIONS.Build_Extractor_pt("now", drones[0].tag, target_g.tag)

        # [持續補農] 到 14 隻
        if total_drones < 14 and player.minerals >= 50 and larvae:
            return actions.RAW_FUNCTIONS.Train_Drone_quick("now", larvae[0].tag)

        # [14 人口] 蓋蟑螂巢 - 稍微加大偏移避開水池
        if drone_count >= 14 and pool and not warren and player.minerals >= 150:
            pos = self._get_build_pos(main_h, nearby_minerals, offset=9.0)
            return actions.RAW_FUNCTIONS.Build_RoachWarren_pt("now", drones[0].tag, pos)

        # [14 人口] 補第二隻王蟲
        if warren and len(overlords) < 3 and player.minerals >= 100 and larvae:
            return actions.RAW_FUNCTIONS.Train_Overlord_quick("now", larvae[0].tag)

        # [採氣分配] 3 隻工蜂
        if extractors and any(e.build_progress == 100 for e in extractors):
            if extractors[0].assigned_harvesters < 3:
                mining_drones = [u for u in drones if u.order_id_0 == 359]
                if mining_drones:
                    return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", mining_drones[0].tag, extractors[0].tag)

        # [爆蟑螂]
        pool_done = any(p.build_progress == 100 for p in pool)
        warren_done = any(w.build_progress == 100 for w in warren)
        if pool_done and warren_done and larvae:
            if player.minerals >= 75 and player.vespene >= 25 and len(roaches) < 8:
                return actions.RAW_FUNCTIONS.Train_Roach_quick("now", larvae[0].tag)

        return actions.RAW_FUNCTIONS.no_op()