# zerg_enemy96_bot.py
# python-sc2 → PySC2 改寫版
# 框架：pysc2.agents.base_agent.BaseAgent

import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

FUNCTIONS = actions.FUNCTIONS

# ── 單位 ID ──────────────────────────────────────
HATCHERY_ID     = units.Zerg.Hatchery
LAIR_ID         = units.Zerg.Lair
DRONE_ID        = units.Zerg.Drone
OVERLORD_ID     = units.Zerg.Overlord
ZERGLING_ID     = units.Zerg.Zergling
ROACH_ID        = units.Zerg.Roach
QUEEN_ID        = units.Zerg.Queen
SPAWNINGPOOL_ID = units.Zerg.SpawningPool
ROACHWARREN_ID  = units.Zerg.RoachWarren
EXTRACTOR_ID    = units.Zerg.Extractor
GEYSER_ID       = units.Neutral.VespeneGeyser
RICH_GEYSER_ID  = units.Neutral.RichVespeneGeyser

# ── 攻擊門檻 ─────────────────────────────────────
ZERGLING_ATTACK_THRESH = 12
ROACH_ATTACK_THRESH    = 6
ARMY_ATTACK_THRESH     = 12

# ── 工蜂目標 ─────────────────────────────────────
DRONE_TARGET_BASE   = 28
DRONE_TARGET_EXPAND = 40

# ── 蟑螂目標 ─────────────────────────────────────
ROACH_TARGET_BASE   = 18
ROACH_TARGET_EXPAND = 30

# ── 小狗目標 ─────────────────────────────────────
ZERGLING_TARGET_BASE   = 12
ZERGLING_TARGET_WARREN = 16


class ZergEnemy96Bot(base_agent.BaseAgent):
    """
    PySC2 版蟲族對手 Bot
    對應 python-sc2 版的所有邏輯：
      distribute_workers / build_overlords / build_drones /
      build_spawning_pool / build_extractor / build_roach_warren /
      expand_if_needed / train_zerglings / train_roaches /
      defend_near_base / attack_logic
    """

    def __init__(self):
        super().__init__()
        self.base_top_left  = None
        self.attack_started = False

        # 選取旗標（PySC2 需先選取再下指令）
        self._select_hatch   = False
        self._select_warren  = False
        self._select_drone   = False
        self._select_army    = False
        self._pending_build  = None   # (FUNCTION, coord) 待執行的建造指令
        self._pending_attack = None   # minimap 座標，待執行的攻擊指令

    def reset(self):
        super().reset()
        self.base_top_left  = None
        self.attack_started = False
        self._select_hatch  = False
        self._select_warren = False
        self._select_drone  = False
        self._select_army   = False
        self._pending_build  = None
        self._pending_attack = None

    # ════════════════════════════════════════════
    #  主步驟
    # ════════════════════════════════════════════
    def step(self, obs):
        super().step(obs)
        available = obs.observation.available_actions

        # 初始化基地方位
        if self.base_top_left is None:
            self.base_top_left = self._detect_base_side(obs)

        # ── 選取等待狀態（優先執行）──
        if self._select_hatch:
            return self._do_select_hatch(obs)
        if self._select_warren:
            return self._do_select_warren(obs)
        if self._select_drone:
            return self._do_select_drone(obs)
        if self._select_army:
            return self._do_select_army(obs)

        # ── 執行待定建造指令 ──
        if self._pending_build is not None:
            fn, coord = self._pending_build
            self._pending_build = None
            if fn.id in available:
                return fn("now", coord)

        # ── 執行待定攻擊指令 ──
        if self._pending_attack is not None:
            coord = self._pending_attack
            self._pending_attack = None
            if FUNCTIONS.Attack_minimap.id in available:
                return FUNCTIONS.Attack_minimap("now", coord)

        # ── 自動補人口（最高優先）──
        if self._supply_left(obs) <= 4:
            if FUNCTIONS.Train_Overlord_quick.id in available:
                return FUNCTIONS.Train_Overlord_quick("now")

        # ── 各模組依序嘗試，第一個有回傳值的就執行 ──
        for module in [
            self._build_spawning_pool,
            self._build_extractor,
            self._build_roach_warren,
            self._expand_if_needed,
            self._build_drones,
            self._train_zerglings,
            self._train_roaches,
            self._defend_near_base,
            self._attack_logic,
        ]:
            action = module(obs)
            if action is not None:
                return action

        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  選取動作
    # ════════════════════════════════════════════
    def _do_select_hatch(self, obs):
        self._select_hatch = False
        u = self._find_unit(obs, LAIR_ID)
        if u is None:
            u = self._find_unit(obs, HATCHERY_ID)
        if u is not None:
            return FUNCTIONS.select_point("select", (int(u.x), int(u.y)))
        return FUNCTIONS.no_op()

    def _do_select_warren(self, obs):
        self._select_warren = False
        u = self._find_unit(obs, ROACHWARREN_ID)
        if u is not None:
            return FUNCTIONS.select_point("select", (int(u.x), int(u.y)))
        return FUNCTIONS.no_op()

    def _do_select_drone(self, obs):
        self._select_drone = False
        u = self._find_unit(obs, DRONE_ID)
        if u is not None:
            return FUNCTIONS.select_point("select", (int(u.x), int(u.y)))
        return FUNCTIONS.no_op()

    def _do_select_army(self, obs):
        available = obs.observation.available_actions
        self._select_army = False
        if FUNCTIONS.select_army.id in available:
            return FUNCTIONS.select_army("select")
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  build_spawning_pool
    # ════════════════════════════════════════════
    def _build_spawning_pool(self, obs):
        available = obs.observation.available_actions
        if self._count_units(obs, SPAWNINGPOOL_ID) > 0:
            return None
        if self._supply_used(obs) < 13:
            return None
        if obs.observation.player.minerals < 200:
            return None

        pos = self._pool_pos()
        if FUNCTIONS.Build_SpawningPool_minimap.id in available:
            return FUNCTIONS.Build_SpawningPool_minimap("now", pos)

        self._select_drone  = True
        self._pending_build = (FUNCTIONS.Build_SpawningPool_minimap, pos)
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  build_extractor（最多 2 座）
    # ════════════════════════════════════════════
    def _build_extractor(self, obs):
        available = obs.observation.available_actions
        if self._count_units(obs, SPAWNINGPOOL_ID) == 0:
            return None

        hatch_total = (self._count_units(obs, HATCHERY_ID) +
                       self._count_units(obs, LAIR_ID))
        max_ext = 2 if hatch_total >= 2 else 1
        if self._count_units(obs, EXTRACTOR_ID) >= max_ext:
            return None
        if obs.observation.player.minerals < 25:
            return None

        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        gy, gx = ((unit_type == GEYSER_ID) | (unit_type == RICH_GEYSER_ID)).nonzero()
        if not gx.size:
            return None

        target = (int(gx[0]), int(gy[0]))
        if FUNCTIONS.Build_Extractor_screen.id in available:
            return FUNCTIONS.Build_Extractor_screen("now", target)

        self._select_drone  = True
        self._pending_build = (FUNCTIONS.Build_Extractor_screen, target)
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  build_roach_warren
    # ════════════════════════════════════════════
    def _build_roach_warren(self, obs):
        available = obs.observation.available_actions
        if self._count_units(obs, SPAWNINGPOOL_ID) == 0:
            return None
        if self._count_units(obs, ROACHWARREN_ID) > 0:
            return None
        if obs.observation.player.minerals < 150:
            return None

        pos = self._warren_pos()
        if FUNCTIONS.Build_RoachWarren_minimap.id in available:
            return FUNCTIONS.Build_RoachWarren_minimap("now", pos)

        self._select_drone  = True
        self._pending_build = (FUNCTIONS.Build_RoachWarren_minimap, pos)
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  expand_if_needed（二礦）
    # ════════════════════════════════════════════
    def _expand_if_needed(self, obs):
        available = obs.observation.available_actions
        hatch_total = (self._count_units(obs, HATCHERY_ID) +
                       self._count_units(obs, LAIR_ID))
        if hatch_total >= 2:
            return None
        if self._count_units(obs, DRONE_ID) < 24:
            return None
        if obs.observation.player.minerals < 300:
            return None

        pos = self._natural_pos()
        if FUNCTIONS.Build_Hatchery_minimap.id in available:
            return FUNCTIONS.Build_Hatchery_minimap("now", pos)

        self._select_drone  = True
        self._pending_build = (FUNCTIONS.Build_Hatchery_minimap, pos)
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  build_drones
    # ════════════════════════════════════════════
    def _build_drones(self, obs):
        available = obs.observation.available_actions
        hatch_total = (self._count_units(obs, HATCHERY_ID) +
                       self._count_units(obs, LAIR_ID))
        target = DRONE_TARGET_EXPAND if hatch_total >= 2 else DRONE_TARGET_BASE

        if self._count_units(obs, DRONE_ID) >= target:
            return None
        if self._supply_left(obs) <= 0:
            return None
        if obs.observation.player.minerals < 50:
            return None

        if FUNCTIONS.Train_Drone_quick.id in available:
            return FUNCTIONS.Train_Drone_quick("now")

        self._select_hatch = True
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  train_zerglings
    # ════════════════════════════════════════════
    def _train_zerglings(self, obs):
        available = obs.observation.available_actions
        if self._count_units(obs, SPAWNINGPOOL_ID) == 0:
            return None

        target = (ZERGLING_TARGET_WARREN
                  if self._count_units(obs, ROACHWARREN_ID) > 0
                  else ZERGLING_TARGET_BASE)

        if self._count_units(obs, ZERGLING_ID) >= target:
            return None
        if self._supply_left(obs) < 1:
            return None
        if obs.observation.player.minerals < 50:
            return None

        if FUNCTIONS.Train_Zergling_quick.id in available:
            return FUNCTIONS.Train_Zergling_quick("now")

        self._select_hatch = True
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  train_roaches
    # ════════════════════════════════════════════
    def _train_roaches(self, obs):
        available = obs.observation.available_actions
        if self._count_units(obs, ROACHWARREN_ID) == 0:
            return None

        hatch_total = (self._count_units(obs, HATCHERY_ID) +
                       self._count_units(obs, LAIR_ID))
        target = ROACH_TARGET_EXPAND if hatch_total >= 2 else ROACH_TARGET_BASE

        if self._count_units(obs, ROACH_ID) >= target:
            return None
        if self._supply_left(obs) < 2:
            return None
        if obs.observation.player.minerals < 75:
            return None
        if obs.observation.player.vespene < 25:
            return None

        if FUNCTIONS.Train_Roach_quick.id in available:
            return FUNCTIONS.Train_Roach_quick("now")

        self._select_hatch = True
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  defend_near_base
    # ════════════════════════════════════════════
    def _defend_near_base(self, obs):
        available = obs.observation.available_actions
        player_rel = obs.observation.feature_screen[
            features.SCREEN_FEATURES.player_relative.index]
        unit_type = obs.observation.feature_screen[
            features.SCREEN_FEATURES.unit_type.index]

        ey, ex = (player_rel == features.PlayerRelative.ENEMY).nonzero()
        if not ex.size:
            return None

        hy, hx = ((unit_type == HATCHERY_ID) | (unit_type == LAIR_ID)).nonzero()
        if not hx.size:
            return None

        hcx, hcy = int(hx.mean()), int(hy.mean())
        dists = np.sqrt((ex - hcx)**2 + (ey - hcy)**2)
        if dists.min() > 20:
            return None

        idx    = int(np.argmin(dists))
        target = (int(ex[idx]), int(ey[idx]))

        if FUNCTIONS.Attack_screen.id in available:
            return FUNCTIONS.Attack_screen("now", target)

        self._select_army   = True
        self._pending_attack = None  # screen 攻擊需另行處理
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  attack_logic
    # ════════════════════════════════════════════
    def _attack_logic(self, obs):
        available = obs.observation.available_actions

        z = self._count_units(obs, ZERGLING_ID)
        r = self._count_units(obs, ROACH_ID)

        if not (z >= ZERGLING_ATTACK_THRESH or
                r >= ROACH_ATTACK_THRESH or
                z + r >= ARMY_ATTACK_THRESH):
            return None

        target = (80, 80) if self.base_top_left else (15, 15)

        if FUNCTIONS.Attack_minimap.id in available:
            self.attack_started = True
            return FUNCTIONS.Attack_minimap("now", target)

        self._select_army    = True
        self._pending_attack = target
        return FUNCTIONS.no_op()

    # ════════════════════════════════════════════
    #  工具函式
    # ════════════════════════════════════════════
    def _detect_base_side(self, obs):
        unit_type = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        hy, hx = ((unit_type == HATCHERY_ID) | (unit_type == LAIR_ID)).nonzero()
        if hx.size:
            return int(hx.mean()) < 42
        mini_rel = obs.observation.feature_minimap[
            features.MINIMAP_FEATURES.player_relative.index]
        my, mx = (mini_rel == features.PlayerRelative.SELF).nonzero()
        if mx.size:
            return int(mx.mean()) < 48
        return True

    def _count_units(self, obs, unit_id) -> int:
        unit_type  = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        player_rel = obs.observation.feature_screen[features.SCREEN_FEATURES.player_relative.index]
        return int(((unit_type == unit_id) &
                    (player_rel == features.PlayerRelative.SELF)).sum())

    def _find_unit(self, obs, unit_id):
        """回傳螢幕上第一個指定類型的 feature_unit，找不到回傳 None。"""
        if not hasattr(obs.observation, "feature_units"):
            return None
        for u in obs.observation.feature_units:
            if int(u.unit_type) == int(unit_id) and int(u.alliance) == 1:
                return u
        return None

    def _supply_used(self, obs) -> int:
        return obs.observation.player.food_used

    def _supply_left(self, obs) -> int:
        p = obs.observation.player
        return p.food_cap - p.food_used

    def _mirror(self, x, y):
        if self.base_top_left is None or self.base_top_left:
            return (x, y)
        return (96 - x, 96 - y)

    def _pool_pos(self):
        return self._mirror(22, 22)

    def _warren_pos(self):
        return self._mirror(28, 22)

    def _natural_pos(self):
        if self.base_top_left is None or self.base_top_left:
            return (24, 36)
        return (72, 60)