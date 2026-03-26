from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import random

RAW_FUNCTIONS = actions.RAW_FUNCTIONS

class ZergStrictBOBot(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.bo_step = 0
        self.attack_sent = False
        self.last_emergency_overlord_time = -999
        self.step_command_sent = False
        self.step_command_time = 0
        self.step_order_count = 0
        
        # ✨ 新增：鎖定一礦座標
        self.start_x = -1
        self.start_y = -1

    def step(self, obs):
        super().step(obs)

        if obs.first():
            print("=== ZergStrictBOBot (強制等待建築版) 啟動 ===")

        if obs.last():
            print("對局結束")
            return RAW_FUNCTIONS.no_op()

        self.obs = obs
        self.time_sec = obs.observation.game_loop / 22.4
        self.raw_units = obs.observation.raw_units
        self.my_units = [u for u in self.raw_units if u.alliance == features.PlayerRelative.SELF]
        self.enemy_units = [u for u in self.raw_units if u.alliance == features.PlayerRelative.ENEMY]

        # 單位分類
        self.hatcheries = self.get_units_by_type(units.Zerg.Hatchery)
        self.lairs = self.get_units_by_type(units.Zerg.Lair)
        self.townhalls = self.hatcheries + self.lairs
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

        # ✨ 新增：遊戲開始時，把第一個基地的座標永遠記下來當作「一礦」
        if self.start_x == -1 and self.townhalls:
            self.start_x = self.townhalls[0].x
            self.start_y = self.townhalls[0].y

        if not self.townhalls:
            return RAW_FUNCTIONS.no_op()

        # 0. 基礎營運
        act = self.emergency_overlord()
        if act: return act
        act = self.distribute_workers()
        if act: return act
        act = self.inject_larva()
        if act: return act

        # 1. 嚴格執行 Build Order
        act = self.execute_strict_build_order()
        if act: return act

        # 2. 4:50 出門攻擊
        act = self.timing_attack()
        if act: return act

        # 3. 戰鬥防守
        act = self.defend_home()
        if act: return act

        return RAW_FUNCTIONS.no_op()

    def advance_step(self, step_name):
        print(f"[{int(self.time_sec):>3}s] ✅ 完成 BO: {step_name} (準備進入 Step {self.bo_step + 1})")
        self.bo_step += 1
        # 清除追蹤狀態，準備給下一個階段使用
        self.step_command_sent = False
        self.step_command_time = 0
        self.step_order_count = 0

    # =========================================================
    # 核心：嚴格階段狀態機 (Strict Build Order State Machine)
    # =========================================================
    def execute_strict_build_order(self):
        step = self.bo_step

        # [0] 13 王蟲
        if step == 0:
            if self.food_used < 13: return self.train_drone() or RAW_FUNCTIONS.no_op()
            act = self.train_overlord()
            if act: self.advance_step("13 王蟲"); return act
            return RAW_FUNCTIONS.no_op() 

        # ✨ [1] 16 二礦 (加入快速重試)
        elif step == 1:
            if self.food_used < 16: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if len(self.townhalls) >= 2: 
                self.advance_step("16 二礦")
                return RAW_FUNCTIONS.no_op()
            
            # ✨ 修正：超時時間從 15 秒縮短為 8 秒！
            # 這樣工蜂如果點無效發呆被抓走，我們能馬上換個新點再派一隻！
            if not self.step_command_sent:
                act = self.build_hatchery()
                if act: 
                    self.step_command_sent = True
                    self.step_command_time = self.time_sec
                    return act
            else:
                if self.time_sec - self.step_command_time > 8:
                    self.step_command_sent = False
            return RAW_FUNCTIONS.no_op()

        # ✨ [2] 17 狗池 (加入強制等待)
        elif step == 2:
            if self.food_used < 17: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if len(self.spawning_pools) >= 1:
                self.advance_step("17 狗池")
                return RAW_FUNCTIONS.no_op()
                
            if not self.step_command_sent:
                act = self.build_spawning_pool()
                if act: 
                    self.step_command_sent = True
                    self.step_command_time = self.time_sec
                    return act
            else:
                if self.time_sec - self.step_command_time > 15: self.step_command_sent = False
            return RAW_FUNCTIONS.no_op()

        # ✨ [3] 18 瓦斯 (加入強制等待)
        elif step == 3:
            if self.food_used < 18: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if len(self.extractors) >= 1:
                self.advance_step("18 瓦斯")
                return RAW_FUNCTIONS.no_op()
                
            if not self.step_command_sent:
                act = self.build_extractor(limit_total=1)
                if act: 
                    self.step_command_sent = True
                    self.step_command_time = self.time_sec
                    return act
            else:
                if self.time_sec - self.step_command_time > 15: self.step_command_sent = False
            return RAW_FUNCTIONS.no_op()

        # [4] 19 王蟲
        elif step == 4:
            if self.food_used < 19: return self.train_drone() or RAW_FUNCTIONS.no_op()
            act = self.train_overlord()
            if act: self.advance_step("19 王蟲"); return act
            return RAW_FUNCTIONS.no_op()

        # ✨ [5] 2:00 后蟲*2 (透過追蹤下單次數來確保數量)
        elif step == 5:
            if not self.count_completed(self.spawning_pools): return RAW_FUNCTIONS.no_op()
            if self.step_order_count >= 2:
                self.advance_step("2:00 后蟲*2")
                return RAW_FUNCTIONS.no_op()
                
            act = self.train_queen()
            if act:
                self.step_order_count += 1
                return act
            return RAW_FUNCTIONS.no_op()

        # [6] 23 狗速
        elif step == 6:
            if self.food_used < 23: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if self.vespene >= 100 and self.minerals >= 100:
                act = self.research_ling_speed()
                if act: self.advance_step("23 狗速"); return act
            return RAW_FUNCTIONS.no_op()

        # [7] 23 小狗*3 (6隻)
        elif step == 7:
            if self.step_order_count >= 3:
                self.advance_step("23 小狗*3")
                return RAW_FUNCTIONS.no_op()
            act = self.train_zergling()
            if act:
                self.step_order_count += 1
                return act
            return RAW_FUNCTIONS.no_op()

        # [8] 29 王蟲
        elif step == 8:
            if self.food_used < 29: return self.train_drone() or RAW_FUNCTIONS.no_op()
            act = self.train_overlord()
            if act: self.advance_step("29 王蟲"); return act
            return RAW_FUNCTIONS.no_op()

        # [9] 30 蟲穴(二本)
        elif step == 9:
            if self.food_used < 30: return self.train_drone() or RAW_FUNCTIONS.no_op()
            act = self.morph_lair()
            if act: self.advance_step("30 蟲穴(二本)"); return act
            return RAW_FUNCTIONS.no_op()

        # [10] 30 后蟲(防守用)
        elif step == 10:
            if self.step_order_count >= 1:
                self.advance_step("30 后蟲(防守用)")
                return RAW_FUNCTIONS.no_op()
            act = self.train_queen()
            if act:
                self.step_order_count += 1
                return act
            return RAW_FUNCTIONS.no_op()

        # ✨ [11] 35 3:00 進化室 (加入強制等待)
        elif step == 11:
            if self.food_used < 35: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if len(self.evos) >= 1:
                self.advance_step("35 進化室")
                return RAW_FUNCTIONS.no_op()
                
            if not self.step_command_sent:
                act = self.build_evolution_chamber()
                if act: 
                    self.step_command_sent = True
                    self.step_command_time = self.time_sec
                    return act
            else:
                if self.time_sec - self.step_command_time > 15: self.step_command_sent = False
            return RAW_FUNCTIONS.no_op()

        # ✨ [12] 35 蟑螂繁殖場 (加入強制等待)
        elif step == 12:
            if len(self.roach_warrens) >= 1:
                self.advance_step("35 蟑螂繁殖場")
                return RAW_FUNCTIONS.no_op()
                
            if not self.step_command_sent:
                act = self.build_roach_warren()
                if act: 
                    self.step_command_sent = True
                    self.step_command_time = self.time_sec
                    return act
            else:
                if self.time_sec - self.step_command_time > 15: self.step_command_sent = False
            return RAW_FUNCTIONS.no_op()

        # [13] 38 遠攻+1
        elif step == 13:
            if self.food_used < 38: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if not self.count_completed(self.evos): return RAW_FUNCTIONS.no_op()
            act = self.research_missile_attack_1()
            if act: self.advance_step("38 遠攻+1"); return act
            return RAW_FUNCTIONS.no_op()

        # [14] 40 王蟲
        elif step == 14:
            if self.food_used < 40: return self.train_drone() or RAW_FUNCTIONS.no_op()
            act = self.train_overlord()
            if act: self.advance_step("40 王蟲"); return act
            return RAW_FUNCTIONS.no_op()

        # ✨ [15] 42 瓦斯*2 (特殊等待處理：需蓋兩個瓦斯)
        elif step == 15:
            if self.food_used < 42: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if len(self.extractors) >= 3:
                self.advance_step("42 瓦斯*2")
                return RAW_FUNCTIONS.no_op()
            
            # 每 2 秒下一次瓦斯指令，直到湊滿 3 個為止 (防止工蜂同時搶目標)
            if self.time_sec - getattr(self, "last_gas_order_time", 0) > 2.0:
                act = self.build_extractor(limit_total=3)
                if act:
                    self.last_gas_order_time = self.time_sec
                    return act
            return RAW_FUNCTIONS.no_op()

        # [16] 44 蟑螂速度
        elif step == 16:
            if self.food_used < 44: return self.train_drone() or RAW_FUNCTIONS.no_op()
            if not self.count_completed(self.roach_warrens): return RAW_FUNCTIONS.no_op()
            act = self.research_roach_speed()
            if act: self.advance_step("44 蟑螂速度"); return act
            return RAW_FUNCTIONS.no_op()

        # [17] 工兵補到41
        elif step == 17:
            if len(self.drones) >= 41:
                self.advance_step("工兵補到41")
                return RAW_FUNCTIONS.no_op()
            act = self.train_drone()
            return act if act else RAW_FUNCTIONS.no_op()

        # [18] 48 王蟲x2 (確保補給空間)
        elif step == 18:
            if self.food_left < 16:
                act = self.train_overlord()
                return act if act else RAW_FUNCTIONS.no_op()
            else:
                self.advance_step("48 王蟲x2")
            return RAW_FUNCTIONS.no_op()

        # [19] 48 蟑螂x8-10
        elif step == 19:
            if self.step_order_count >= 8:
                self.advance_step("48 蟑螂x8-10 完成！進入待命等待出門")
                return RAW_FUNCTIONS.no_op()
            act = self.train_roach()
            if act:
                self.step_order_count += 1
                return act
            return RAW_FUNCTIONS.no_op()

        return None

    # =========================================================
    # 巨集營運機制 (Macro)
    # =========================================================
    def emergency_overlord(self):
        if self.food_cap >= 200: return None
        if self.time_sec - self.last_emergency_overlord_time < 15: return None
        if self.food_left <= 2 and self.larvae and self.minerals >= 100:
            act = self.train_overlord()
            if act:
                self.last_emergency_overlord_time = self.time_sec
                print(f"[{int(self.time_sec):>3}s] ⚠️ 緊急防卡人口：造王蟲！")
                return act
        return None

    def distribute_workers(self):
        # 1. 瓦斯分配 (我們蓋的瓦斯廠一定在基地旁，這部分沒問題)
        completed_extractors = [ex for ex in self.extractors if getattr(ex, "build_progress", 1.0) >= 1.0]
        for ex in completed_extractors:
            if getattr(ex, "assigned_harvesters", 0) < getattr(ex, "ideal_harvesters", 3):
                idle = self.get_idle_drones()
                drones = idle if idle else self.drones
                if drones:
                    func = self.raw_func("Harvest_Gather_unit")
                    if func: return func("now", random.choice(drones).tag, ex.tag)

        # ✨ 2. 礦物分配 (加入基地距離限制)
        idle_drones = self.get_idle_drones()
        if idle_drones:
            # 抓出全圖的礦
            all_minerals = [u for u in self.raw_units if u.unit_type in {units.Neutral.MineralField, units.Neutral.MineralField750}]
            
            # 建立一個「安全礦區」清單
            safe_minerals = []
            for m in all_minerals:
                # 檢查這顆礦是否在我們任何一個基地的 12 格範圍內
                for base in self.townhalls:
                    if self.distance(base, m) <= 12:
                        safe_minerals.append(m)
                        break # 這顆礦合格了，換檢查下一顆
            
            # 如果有安全的礦，才隨機派工蜂去採
            if safe_minerals:
                func = self.raw_func("Harvest_Gather_unit")
                if func: 
                    return func("now", idle_drones[0].tag, random.choice(safe_minerals).tag)
                    
        return None

    # =========================================================
    # 戰鬥與移動
    # =========================================================
    def timing_attack(self):
        if self.time_sec < 290 or self.attack_sent: return None
        tags = [u.tag for u in self.roaches + self.zerglings]
        if not tags: return None
        
        enemy_pos = self.get_enemy_start_pos()
        func = self.raw_func("Attack_pt")
        if func:
            try:
                self.attack_sent = True
                print(f"[{int(self.time_sec):>3}s] ⚔️ 4:50 時間到！全軍出擊！")
                return func("now", tags, enemy_pos)
            except Exception: pass
        return None

    def defend_home(self):
        enemy = self.any_enemy_near_base(radius=25)
        if enemy is None: return None
        
        tags = [u.tag for u in self.roaches + self.zerglings]
        if not tags: return None
        func = self.raw_func("Attack_pt")
        if func: return func("now", tags, (int(enemy.x), int(enemy.y)))
        return None

    # =========================================================
    # 動作執行器 (Action Wrappers)
    # =========================================================
    def train_overlord(self): return self._train_unit("Train_Overlord_quick", 100, 0, 0)
    def train_drone(self): return self._train_unit("Train_Drone_quick", 50, 0, 1)
    def train_zergling(self): return self._train_unit("Train_Zergling_quick", 50, 0, 1)
    def train_roach(self): return self._train_unit("Train_Roach_quick", 75, 25, 2)
    
    def _train_unit(self, func_name, min_cost, gas_cost, food_cost):
        if not self.can_afford(minerals=min_cost, gas=gas_cost, food=food_cost): return None
        larva = self.get_first_trainable_larva()
        if larva is None: return None
        func = self.raw_func(func_name)
        if func:
            try: return func("now", larva.tag)
            except Exception: pass
        return None

    def train_queen(self):
        if self.minerals < 150 or not self.spawning_pools: return None
        func = self.raw_func("Train_Queen_quick")
        if not func: return None

        # ✨ 終極防呆 1：加入 2 秒冷卻時間，防止瞬間在同一個基地狂點兩隻后蟲
        if self.time_sec - getattr(self, "last_queen_order_time", 0) < 2.0:
            return None

        # ✨ 終極防呆 2：基地必須「已經完全蓋好 (build_progress == 1.0)」且「正在發呆 (is_idle)」
        valid_bases = [b for b in self.townhalls if getattr(b, "build_progress", 1.0) >= 1.0 and self.is_idle(b)]
        if not valid_bases: return None

        # ✨ 終極防呆 3：優先挑選「半徑 10 格內沒有后蟲」的基地，強迫分散生產！
        for base in valid_bases:
            queens_nearby = [q for q in self.queens if self.distance(base, q) < 10]
            if not queens_nearby:
                try: 
                    self.last_queen_order_time = self.time_sec
                    return func("now", base.tag)
                except Exception: pass
                
        # 如果每個基地都有后蟲了 (例如 Step 10 要補第三隻)，就挑第一個有空的基地造
        try:
            self.last_queen_order_time = self.time_sec
            return func("now", valid_bases[0].tag)
        except Exception: pass
            
        return None

    def inject_larva(self):
        if not self.queens or not self.townhalls: return None
        idle_queens = [q for q in self.queens if self.is_idle(q) and getattr(q, "energy", 0) >= 25]
        if not idle_queens: return None
        
        q = idle_queens[0]
        
        # ✨ 修正：后蟲只能對「已經完全蓋好」的基地噴卵，防止牠跑去還沒完工的基地發呆
        valid_bases = [b for b in self.townhalls if getattr(b, "build_progress", 1.0) >= 1.0]
        if not valid_bases: return None
        
        target = self.find_closest(q, valid_bases)
        func = self.raw_func("Effect_InjectLarva_unit")
        if func: return func("now", q.tag, target.tag)
        return None

    def build_hatchery(self):
        if self.minerals < 300: return None
        drones = self.get_idle_drones() or self.drones
        if not drones: return None
        pos = self.get_natural_pos()
        func = self.raw_func("Build_Hatchery_pt")
        if func:
            try: return func("now", drones[0].tag, pos)
            except Exception: pass
        return None

    def build_spawning_pool(self): return self._build_structure("pool", "Build_SpawningPool_pt", 200)
    def build_evolution_chamber(self): return self._build_structure("evo", "Build_EvolutionChamber_pt", 75)
    def build_roach_warren(self): return self._build_structure("roach", "Build_RoachWarren_pt", 150)

    def _build_structure(self, pos_name, func_name, min_cost):
        if self.minerals < min_cost: return None
        drones = self.get_idle_drones() or self.drones
        if not drones: return None
        pos = self.get_build_pos(pos_name)
        func = self.raw_func(func_name)
        if func:
            try: return func("now", drones[0].tag, pos)
            except Exception: pass
        return None

    def build_extractor(self, limit_total):
        if len(self.extractors) >= limit_total or self.minerals < 25: return None
        drones = self.get_idle_drones() or self.drones
        if not drones or not self.townhalls: return None
        
        candidate_geysers = []
        for base in self.townhalls:
            geysers = [u for u in self.raw_units if u.unit_type in {units.Neutral.VespeneGeyser} and self.distance(base, u) <= 12]
            candidate_geysers.extend(geysers)
            
        free_geysers = [g for g in candidate_geysers if not any(self.distance(g, ex) <= 1.5 for ex in self.extractors)]
        if not free_geysers: return None
        
        target = self.find_closest(drones[0], free_geysers)
        func = self.raw_func("Build_Extractor_unit")
        if func:
            try: return func("now", drones[0].tag, target.tag)
            except Exception: pass
        return None

    def morph_lair(self):
        if self.minerals < 150 or self.vespene < 100 or not self.hatcheries: return None
        func = self.raw_func("Morph_Lair_quick")
        if func:
            for hatch in self.hatcheries:
                if self.is_idle(hatch):
                    try: return func("now", hatch.tag)
                    except Exception: pass
        return None

    def research_ling_speed(self): 
        act = self._research(self.spawning_pools, "Research_ZerglingMetabolicBoost_quick")
        return act if act else self._research(self.spawning_pools, "Research_MetabolicBoost_quick")

    def research_roach_speed(self): 
        act = self._research(self.roach_warrens, "Research_GlialReconstitution_quick")
        return act if act else self._research(self.roach_warrens, "Research_RoachMovementSpeed_quick")

    def research_missile_attack_1(self): 
        act = self._research(self.evos, "Research_ZergMissileWeaponsLevel1_quick")
        return act if act else self._research(self.evos, "Research_ZergMissileWeapons_quick")
    def _research(self, building_list, func_name):
        if not building_list: return None
        func = self.raw_func(func_name)
        if func:
            try: return func("now", building_list[0].tag)
            except Exception: pass
        return None

    # =========================================================
    # 輔助函式與座標計算
    # =========================================================
    def get_units_by_type(self, unit_type): return [u for u in self.my_units if u.unit_type == unit_type]
    def is_idle(self, unit_obj): return getattr(unit_obj, "order_length", 0) == 0
    def get_idle_drones(self): return [u for u in self.drones if self.is_idle(u)]
    def get_idle_larva(self): return [u for u in self.larvae if self.is_idle(u)]
    def distance(self, a, b): return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5
    def can_afford(self, minerals=0, gas=0, food=0): return self.minerals >= minerals and self.vespene >= gas and self.food_left >= food
    def count_completed(self, unit_list): return sum(1 for u in unit_list if getattr(u, "build_progress", 1.0) >= 1.0)

    def get_main_base(self):
        if not self.townhalls: return None
        # 改成：永遠回傳距離「開局一礦點」最近的那個基地
        return min(self.townhalls, key=lambda h: (h.x - self.start_x)**2 + (h.y - self.start_y)**2)

    def get_build_pos(self, name):
        # ✨ 終極修正：不論二礦在哪，永遠以「開局一礦 (start_x, start_y)」為中心蓋建築
        dir_x = 1 if self.start_x < 48 else -1
        dir_y = 1 if self.start_y < 48 else -1
        
        slots = {
            "pool":  (int(self.start_x + dir_x * 7), int(self.start_y + dir_y * 5)),
            "evo":   (int(self.start_x + dir_x * 4), int(self.start_y + dir_y * 8)),
            "roach": (int(self.start_x + dir_x * 8), int(self.start_y + dir_y * 2))
        }
        return slots[name]
    
    
    
    def get_first_trainable_larva(self):
        idle = self.get_idle_larva()
        return idle[0] if idle else (self.larvae[0] if self.larvae else None)
    
    def find_closest(self, src, unit_list): return min(unit_list, key=lambda u: self.distance(src, u))
    
    def raw_func(self, name):
        try:
            return getattr(RAW_FUNCTIONS, name)
        except (AttributeError, KeyError):
            return None
        
    def any_enemy_near_base(self, radius):
        for base in self.townhalls:
            for e in self.enemy_units:
                if self.distance(base, e) <= radius: return e
        return None

    def get_natural_pos(self):
        main = self.get_main_base()
        
        minerals = [u for u in self.raw_units if u.unit_type in {units.Neutral.MineralField, units.Neutral.MineralField750}]
        distant_minerals = [m for m in minerals if self.distance(main, m) > 15]
        
        if not distant_minerals: return (int(main.x), int(main.y))
            
        closest_mineral = min(distant_minerals, key=lambda m: self.distance(main, m))
        
        cluster = [m for m in distant_minerals if self.distance(closest_mineral, m) < 10]
        avg_x = sum(m.x for m in cluster) / len(cluster)
        avg_y = sum(m.y for m in cluster) / len(cluster)
        
        import random
        jitter_x = random.uniform(-1.0, 1.0)
        jitter_y = random.uniform(-1.0, 1.0)
        
        if main.x < 48:
            # 左上角出生：二礦在右方偏下，避開礦脈
            target_x = main.x + 24
            target_y = main.y + 10 
        else:
            # 右下角出生：二礦在左方偏上，避開礦脈
            target_x = main.x - 24
            target_y = main.y - 10
            
        return (int(target_x), int(target_y))

    

    def get_enemy_start_pos(self): return (86, 86) if self.get_main_base().x < 48 else (10, 10)