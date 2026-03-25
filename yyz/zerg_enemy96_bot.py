from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

RAW_FUNCTIONS = actions.RAW_FUNCTIONS


class ZergEnemy96Bot(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()

        self.attack_sent = False
        self.debug_last = -1

        # build order flags
        self.bo_13_overlord_done = False
        self.bo_16_hatch_done = False
        self.bo_17_pool_done = False
        self.bo_18_gas_done = False
        self.bo_19_overlord_done = False
        self.bo_2queen_done = False
        self.bo_ling_speed_done = False
        self.bo_3ling_done = False
        self.bo_29_overlord_done = False
        self.bo_lair_done = False
        self.bo_def_queen_done = False
        self.bo_evo_done = False
        self.bo_roach_warren_done = False
        self.bo_missile1_done = False
        self.bo_40_overlord_done = False
        self.bo_2extra_gas_done = False
        self.bo_roach_speed_done = False
        self.bo_drone_41_done = False
        self.bo_48_overlord2_done = False
        self.bo_roach_push_done = False

    # =========================================================
    # 基本工具
    # =========================================================
    def step(self, obs):
        super().step(obs)

        if obs.first():
            print("ZergEnemy96Bot 啟動成功（PySC2 RAW 版）")

        if obs.last():
            print("對局結束")
            return RAW_FUNCTIONS.no_op()

        self.obs = obs
        self.time_sec = obs.observation.game_loop / 22.4

        self.raw_units = obs.observation.raw_units
        self.my_units = [u for u in self.raw_units if u.alliance == features.PlayerRelative.SELF]
        self.enemy_units = [u for u in self.raw_units if u.alliance == features.PlayerRelative.ENEMY]

        self.hatcheries = self.get_units_by_type(units.Zerg.Hatchery)
        self.lairs = self.get_units_by_type(units.Zerg.Lair)
        self.hives = self.get_units_by_type(units.Zerg.Hive)
        self.townhalls = self.hatcheries + self.lairs + self.hives

        self.larvae = self.get_units_by_type(units.Zerg.Larva)
        self.drones = self.get_units_by_type(units.Zerg.Drone)
        self.overlords = self.get_units_by_type(units.Zerg.Overlord)
        self.queens = self.get_units_by_type(units.Zerg.Queen)
        self.zerglings = self.get_units_by_type(units.Zerg.Zergling)
        self.roaches = self.get_units_by_type(units.Zerg.Roach)

        self.spawning_pools = self.get_units_by_type(units.Zerg.SpawningPool)
        self.extractors = self.get_units_by_type(units.Zerg.Extractor)
        self.roach_warrens = self.get_units_by_type(units.Zerg.RoachWarren)
        self.evos = self.get_units_by_type(units.Zerg.EvolutionChamber)

        self.minerals = obs.observation.player.minerals
        self.vespene = obs.observation.player.vespene
        self.food_used = obs.observation.player.food_used
        self.food_cap = obs.observation.player.food_cap
        self.food_left = self.food_cap - self.food_used

        if not self.townhalls:
            return RAW_FUNCTIONS.no_op()

        # 定期印狀態
        now_t = int(self.time_sec)
        if now_t != self.debug_last and now_t % 20 == 0:
            self.debug_last = now_t
            print(
                f"[{now_t:>3}s] 礦={self.minerals} 氣={self.vespene} "
                f"人口={self.food_used}/{self.food_cap} "
                f"工蜂={len(self.drones)} 后蟲={len(self.queens)} "
                f"狗={len(self.zerglings)} 蟑螂={len(self.roaches)} "
                f"基地={len(self.townhalls)}"
            )

        # 0. 緊急防卡人口
        act = self.emergency_overlord()
        if act:
            return act

        # 1. 依照你給的 build order 順序執行
        act = self.execute_build_order()
        if act:
            return act

        # 2. 補后蟲 / inject
        act = self.inject_larva()
        if act:
            return act

        # 3. 補工蜂到 41
        act = self.produce_drones()
        if act:
            return act

        # 4. 補兵：先狗少量，再蟑螂主力
        act = self.produce_army()
        if act:
            return act

        # 5. 防守
        act = self.defend_home()
        if act:
            return act

        # 6. 4:50 出門
        act = self.timing_attack()
        if act:
            return act

        return RAW_FUNCTIONS.no_op()

    def get_units_by_type(self, unit_type):
        return [u for u in self.my_units if u.unit_type == unit_type]

    def get_enemy_by_type(self, unit_type):
        return [u for u in self.enemy_units if u.unit_type == unit_type]

    def is_idle(self, unit_obj):
        return getattr(unit_obj, "order_length", 0) == 0

    def get_idle_drones(self):
        return [u for u in self.drones if self.is_idle(u)]

    def get_idle_larva(self):
        return [u for u in self.larvae if self.is_idle(u)]

    def get_idle_townhall(self):
        return [u for u in self.townhalls if self.is_idle(u)]

    def get_idle_queens(self):
        return [u for u in self.queens if self.is_idle(u)]

    def distance(self, a, b):
        ax, ay = self.pos_of(a)
        bx, by = self.pos_of(b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def pos_of(self, obj):
        if hasattr(obj, "x") and hasattr(obj, "y"):
            return (float(obj.x), float(obj.y))
        if isinstance(obj, tuple):
            return (float(obj[0]), float(obj[1]))
        return (0.0, 0.0)

    def can_afford(self, minerals=0, gas=0, food=0):
        return self.minerals >= minerals and self.vespene >= gas and self.food_left >= food

    def get_main_base(self):
        return min(self.townhalls, key=lambda h: (h.x + h.y))

    def get_natural_pos(self):
        main = self.get_main_base()
        # Simple96 粗略二礦點
        if main.x < 48:
            return (min(90, int(main.x + 18)), min(90, int(main.y + 6)))
        return (max(6, int(main.x - 18)), max(6, int(main.y - 6)))

    def get_enemy_start_pos(self):
        main = self.get_main_base()
        if main.x < 48:
            return (86, 86)
        return (10, 10)

    def get_build_pos(self, name):
        main = self.get_main_base()

        if main.x < 48:
            slots = {
                "pool": (int(main.x + 6), int(main.y + 0)),
                "evo": (int(main.x + 8), int(main.y + 4)),
                "roach": (int(main.x + 10), int(main.y - 2)),
            }
        else:
            slots = {
                "pool": (int(main.x - 6), int(main.y + 0)),
                "evo": (int(main.x - 8), int(main.y - 4)),
                "roach": (int(main.x - 10), int(main.y + 2)),
            }
        return slots[name]

    def find_closest(self, src, unit_list):
        if not unit_list:
            return None
        return min(unit_list, key=lambda u: self.distance(src, u))

    def get_geysers_near_base(self, base, radius=12):
        geyser_types = {
            units.Neutral.VespeneGeyser,
            units.Neutral.RichVespeneGeyser,
            units.Neutral.SpacePlatformGeyser,
            units.Neutral.ProtossVespeneGeyser,
            units.Neutral.PurifierVespeneGeyser,
            units.Neutral.ShakurasVespeneGeyser,
        }

        geysers = [
            u for u in self.raw_units
            if u.alliance == features.PlayerRelative.NEUTRAL and u.unit_type in geyser_types
        ]
        return [g for g in geysers if self.distance(base, g) <= radius]

    def townhall_count(self):
        return len(self.townhalls)

    def lair_or_higher_exists(self):
        return len(self.lairs) + len(self.hives) > 0

    def army_units(self):
        return self.zerglings + self.roaches

    def army_tags(self):
        return [u.tag for u in self.army_units()]

    def count_completed(self, unit_list):
        return sum(1 for u in unit_list if getattr(u, "build_progress", 1.0) >= 1.0)

    def has_building(self, unit_list):
        return len(unit_list) > 0

    def any_enemy_near_base(self, radius=22):
        for base in self.townhalls:
            for e in self.enemy_units:
                if self.distance(base, e) <= radius:
                    return e
        return None

    def raw_func(self, name):
        return getattr(RAW_FUNCTIONS, name, None)

    def call_first_existing_func(self, names, *args):
        for n in names:
            f = self.raw_func(n)
            if f is not None:
                try:
                    return f(*args)
                except Exception:
                    pass
        return None

    # =========================================================
    # 核心：build order
    # =========================================================
    def execute_build_order(self):
        # 13 王蟲
        if self.food_used >= 13 and not self.bo_13_overlord_done:
            act = self.train_overlord()
            if act:
                self.bo_13_overlord_done = True
                print("BO: 13 王蟲")
                return act

        # 16 二礦
        if self.food_used >= 16 and not self.bo_16_hatch_done:
            act = self.build_hatchery()
            if act:
                self.bo_16_hatch_done = True
                print("BO: 16 二礦")
                return act

        # 17 狗池
        if self.food_used >= 17 and not self.bo_17_pool_done:
            act = self.build_spawning_pool()
            if act:
                self.bo_17_pool_done = True
                print("BO: 17 狗池")
                return act

        # 18 瓦斯
        if self.food_used >= 18 and not self.bo_18_gas_done:
            act = self.build_extractor(limit_total=1)
            if act:
                self.bo_18_gas_done = True
                print("BO: 18 瓦斯")
                return act

        # 19 王蟲
        if self.food_used >= 19 and not self.bo_19_overlord_done:
            act = self.train_overlord()
            if act:
                self.bo_19_overlord_done = True
                print("BO: 19 王蟲")
                return act

        # 2:00 后蟲 x2
        if self.time_sec >= 120 and not self.bo_2queen_done:
            if len(self.queens) < 2:
                act = self.train_queen()
                if act:
                    print("BO: 2:00 補后蟲")
                    return act
            else:
                self.bo_2queen_done = True
                print("BO: 2:00 后蟲 x2 完成")

        # 23 狗速
        if self.food_used >= 23 and not self.bo_ling_speed_done:
            act = self.research_ling_speed()
            if act:
                self.bo_ling_speed_done = True
                print("BO: 23 狗速")
                return act

        # 小狗 x3（Zergling 一次出 2 隻，這裡做成至少 6 隻）
        if self.food_used >= 23 and not self.bo_3ling_done:
            if len(self.zerglings) < 6:
                act = self.train_zergling()
                if act:
                    return act
            else:
                self.bo_3ling_done = True
                print("BO: 小狗補足")

        # 29 王蟲
        if self.food_used >= 29 and not self.bo_29_overlord_done:
            act = self.train_overlord()
            if act:
                self.bo_29_overlord_done = True
                print("BO: 29 王蟲")
                return act

        # 30 二本
        if self.food_used >= 30 and not self.bo_lair_done:
            act = self.morph_lair()
            if act:
                self.bo_lair_done = True
                print("BO: 30 蟲穴（二本）")
                return act

        # 30 防守后蟲
        if self.food_used >= 30 and not self.bo_def_queen_done:
            if len(self.queens) < 3:
                act = self.train_queen()
                if act:
                    print("BO: 30 防守后蟲")
                    return act
            else:
                self.bo_def_queen_done = True
                print("BO: 防守后蟲完成")

        # 35 進化室
        if self.food_used >= 35 and not self.bo_evo_done:
            act = self.build_evolution_chamber()
            if act:
                self.bo_evo_done = True
                print("BO: 35 進化室")
                return act

        # 35 蟑螂繁殖場
        if self.food_used >= 35 and not self.bo_roach_warren_done:
            act = self.build_roach_warren()
            if act:
                self.bo_roach_warren_done = True
                print("BO: 35 蟑螂繁殖場")
                return act

        # 38 遠攻 +1
        if self.food_used >= 38 and not self.bo_missile1_done:
            act = self.research_missile_attack_1()
            if act:
                self.bo_missile1_done = True
                print("BO: 38 遠攻 +1")
                return act

        # 40 王蟲
        if self.food_used >= 40 and not self.bo_40_overlord_done:
            act = self.train_overlord()
            if act:
                self.bo_40_overlord_done = True
                print("BO: 40 王蟲")
                return act

        # 42 瓦斯 *2（總共到 3 個）
        if self.food_used >= 42 and not self.bo_2extra_gas_done:
            if len(self.extractors) < 3:
                act = self.build_extractor(limit_total=3)
                if act:
                    print("BO: 42 補額外瓦斯")
                    return act
            else:
                self.bo_2extra_gas_done = True
                print("BO: 42 瓦斯 *2 完成")

        # 44 蟑螂速度
        if self.food_used >= 44 and not self.bo_roach_speed_done:
            act = self.research_roach_speed()
            if act:
                self.bo_roach_speed_done = True
                print("BO: 44 蟑螂速度")
                return act

        # 工兵補到 41
        if not self.bo_drone_41_done:
            if len(self.drones) < 41:
                act = self.train_drone()
                if act:
                    return act
            else:
                self.bo_drone_41_done = True
                print("BO: 工兵補到 41 完成")

        # 48 王蟲 x2
        if self.food_used >= 48 and not self.bo_48_overlord2_done:
            if len(self.overlords) < 6:
                act = self.train_overlord()
                if act:
                    return act
            else:
                self.bo_48_overlord2_done = True
                print("BO: 48 王蟲 x2 完成")

        # 48 蟑螂 x8~10
        if self.food_used >= 48 and not self.bo_roach_push_done:
            if len(self.roaches) < 8:
                act = self.train_roach()
                if act:
                    return act
            else:
                self.bo_roach_push_done = True
                print("BO: 48 蟑螂主力完成")

        return None

    # =========================================================
    # 生產 / 建造
    # =========================================================
    def emergency_overlord(self):
        if self.food_cap >= 200:
            return None
        if self.food_left <= 2 and self.larvae and self.minerals >= 100:
            return self.train_overlord()
        return None

    def train_overlord(self):
        larva = self.get_first_trainable_larva()
        if larva is None or self.minerals < 100:
            return None
        func = self.raw_func("Train_Overlord_quick")
        if func is None:
            return None
        try:
            return func("now", larva.tag)
        except Exception:
            return None

    def train_drone(self):
        larva = self.get_first_trainable_larva()
        if larva is None or not self.can_afford(minerals=50, food=1):
            return None
        func = self.raw_func("Train_Drone_quick")
        if func is None:
            return None
        try:
            return func("now", larva.tag)
        except Exception:
            return None

    def train_zergling(self):
        larva = self.get_first_trainable_larva()
        if larva is None or not self.can_afford(minerals=50, food=1):
            return None
        if not self.spawning_pools:
            return None
        func = self.raw_func("Train_Zergling_quick")
        if func is None:
            return None
        try:
            return func("now", larva.tag)
        except Exception:
            return None

    def train_roach(self):
        larva = self.get_first_trainable_larva()
        if larva is None or not self.can_afford(minerals=75, gas=25, food=2):
            return None
        if not self.roach_warrens:
            return None
        func = self.raw_func("Train_Roach_quick")
        if func is None:
            return None
        try:
            return func("now", larva.tag)
        except Exception:
            return None

    def train_queen(self):
        if not self.spawning_pools:
            return None
        if self.minerals < 150:
            return None

        bases = sorted(self.townhalls, key=lambda x: getattr(x, "order_length", 0))
        func = self.raw_func("Train_Queen_quick")
        if func is None:
            return None

        for base in bases:
            if self.is_idle(base):
                try:
                    return func("now", base.tag)
                except Exception:
                    continue
        return None

    def build_hatchery(self):
        if self.townhall_count() >= 2 or self.minerals < 300:
            return None

        drones = self.get_idle_drones() or self.drones
        if not drones:
            return None

        drone = drones[0]
        pos = self.get_natural_pos()
        func = self.raw_func("Build_Hatchery_pt")
        if func is None:
            return None

        try:
            return func("now", drone.tag, pos)
        except Exception:
            return None

    def build_spawning_pool(self):
        if self.has_building(self.spawning_pools) or self.minerals < 200:
            return None

        drones = self.get_idle_drones() or self.drones
        if not drones:
            return None

        drone = drones[0]
        pos = self.get_build_pos("pool")
        func = self.raw_func("Build_SpawningPool_pt")
        if func is None:
            return None

        try:
            return func("now", drone.tag, pos)
        except Exception:
            return None

    def build_evolution_chamber(self):
        if self.has_building(self.evos) or self.minerals < 75:
            return None

        drones = self.get_idle_drones() or self.drones
        if not drones:
            return None

        drone = drones[0]
        pos = self.get_build_pos("evo")
        func = self.raw_func("Build_EvolutionChamber_pt")
        if func is None:
            return None

        try:
            return func("now", drone.tag, pos)
        except Exception:
            return None

    def build_roach_warren(self):
        if self.has_building(self.roach_warrens) or self.minerals < 150:
            return None
        if not self.spawning_pools:
            return None

        drones = self.get_idle_drones() or self.drones
        if not drones:
            return None

        drone = drones[0]
        pos = self.get_build_pos("roach")
        func = self.raw_func("Build_RoachWarren_pt")
        if func is None:
            return None

        try:
            return func("now", drone.tag, pos)
        except Exception:
            return None

    def build_extractor(self, limit_total=1):
        if len(self.extractors) >= limit_total or self.minerals < 25:
            return None
        if not self.townhalls or not self.drones:
            return None

        drone = self.get_idle_drones()[0] if self.get_idle_drones() else self.drones[0]

        candidate_geysers = []
        for base in self.townhalls:
            candidate_geysers.extend(self.get_geysers_near_base(base, radius=12))

        # 去掉已經蓋了 Extractor 的氣礦
        free_geysers = []
        for g in candidate_geysers:
            occupied = False
            for ex in self.extractors:
                if self.distance(g, ex) <= 1.5:
                    occupied = True
                    break
            if not occupied:
                free_geysers.append(g)

        if not free_geysers:
            return None

        target = self.find_closest(drone, free_geysers)
        func = self.raw_func("Build_Extractor_unit")
        if func is None:
            return None

        try:
            return func("now", drone.tag, target.tag)
        except Exception:
            return None

    def morph_lair(self):
        if self.lair_or_higher_exists():
            return None
        if not self.spawning_pools:
            return None
        if self.minerals < 150 or self.vespene < 100:
            return None
        if not self.hatcheries:
            return None

        func = self.raw_func("Morph_Lair_quick")
        if func is None:
            return None

        for hatch in self.hatcheries:
            if self.is_idle(hatch):
                try:
                    return func("now", hatch.tag)
                except Exception:
                    continue
        return None

    # =========================================================
    # 升級
    # =========================================================
    def research_ling_speed(self):
        if not self.spawning_pools or self.vespene < 100 or self.minerals < 100:
            return None

        pool = self.spawning_pools[0]

        return self.call_first_existing_func(
            [
                "Research_ZerglingMetabolicBoost_quick",
                "Research_MetabolicBoost_quick",
            ],
            "now",
            pool.tag,
        )

    def research_missile_attack_1(self):
        if not self.evos or self.minerals < 100 or self.vespene < 100:
            return None

        evo = self.evos[0]

        return self.call_first_existing_func(
            [
                "Research_ZergMissileWeapons_quick",
                "Research_ZergMissileWeaponsLevel1_quick",
            ],
            "now",
            evo.tag,
        )

    def research_roach_speed(self):
        if not self.roach_warrens or self.minerals < 100 or self.vespene < 100:
            return None
        if not self.lair_or_higher_exists():
            return None

        rw = self.roach_warrens[0]

        return self.call_first_existing_func(
            [
                "Research_GlialReconstitution_quick",
                "Research_RoachMovementSpeed_quick",
            ],
            "now",
            rw.tag,
        )

    # =========================================================
    # inject / 經濟 / 出兵
    # =========================================================
    def inject_larva(self):
        if not self.queens or not self.townhalls:
            return None
        if self.vespene < 25:
            return None

        queen = None
        for q in self.queens:
            if self.is_idle(q) and getattr(q, "energy", 0) >= 25:
                queen = q
                break

        if queen is None:
            return None

        target_base = self.find_closest(queen, self.townhalls)

        return self.call_first_existing_func(
            [
                "Effect_InjectLarva_unit",
                "Effect_InjectLarva_unit",
            ],
            "now",
            queen.tag,
            target_base.tag,
        )

    def produce_drones(self):
        if len(self.drones) >= 41:
            return None
        if not self.can_afford(minerals=50, food=1):
            return None
        return self.train_drone()

    def produce_army(self):
        # 前期先少量狗
        if len(self.zerglings) < 6 and self.spawning_pools and self.can_afford(minerals=50, food=1):
            return self.train_zergling()

        # 中後期主力蟑螂
        if self.roach_warrens and self.can_afford(minerals=75, gas=25, food=2):
            return self.train_roach()

        # 沒蟑螂巢前，礦很多就補狗
        if self.spawning_pools and self.minerals >= 200 and self.food_left >= 1:
            return self.train_zergling()

        return None

    # =========================================================
    # 戰鬥
    # =========================================================
    def defend_home(self):
        enemy = self.any_enemy_near_base(radius=20)
        if enemy is None:
            return None

        tags = self.army_tags()
        if not tags:
            return None

        func = self.raw_func("Attack_unit")
        if func is not None:
            try:
                return func("now", tags, enemy.tag)
            except Exception:
                pass

        func = self.raw_func("Attack_pt")
        if func is not None:
            try:
                return func("now", tags, (int(enemy.x), int(enemy.y)))
            except Exception:
                pass

        return None

    def timing_attack(self):
        # 4:50 = 290 秒
        if self.time_sec < 290:
            return None
        if self.attack_sent:
            return None

        tags = self.army_tags()
        if len(self.roaches) < 8 or not tags:
            return None

        enemy_pos = self.get_enemy_start_pos()

        func = self.raw_func("Attack_pt")
        if func is None:
            return None

        try:
            self.attack_sent = True
            print("BO: 4:50 出門進攻")
            return func("now", tags, enemy_pos)
        except Exception:
            return None

    # =========================================================
    # 幫助函式
    # =========================================================
    def get_first_trainable_larva(self):
        if not self.larvae:
            return None
        idle = self.get_idle_larva()
        if idle:
            return idle[0]
        return self.larvae[0]