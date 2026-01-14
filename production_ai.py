import csv
import time
import os
import sc2
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId

# 設定遊戲路徑 (請確認這是你的路徑)
os.environ["SC2PATH"] = r"D:\StarCraft II"

# =========================================================
# 📊 模組 1: 數據收集器 (DataCollector)
# 用途: 記錄每一刻的資源與決策，這是 AI 專題的精隨
# =========================================================
class DataCollector:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        self.filename = f"logs/marauder_log_{int(time.time())}.csv"
        
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Time", "Minerals", "Vespene", 
                "Supply_Used", "Marauder_Count", 
                "Decision_Type", "Decision_Target"
            ])

    def log_step(self, time, minerals, vespene, supply, count, decision):
        d_type = decision[0] if decision else "None"
        d_target = decision[1] if decision else "None"

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                round(time, 2), minerals, vespene, 
                supply, count, d_type, d_target
            ])

# =========================================================
# 🧠 模組 2: 生產大腦 (ProductionAI) - 這是你負責的核心 A 部分
# 用途: 判斷缺什麼，發出指令
# =========================================================
class ProductionAI:
    def __init__(self, bot):
        self.bot = bot
        self.collector = DataCollector()
        
        # 初始化目標
        self.target_units = {}
        self.target_structures = {}

    def set_goals(self, units, structures):
        self.target_units = units
        self.target_structures = structures

    def make_decision(self):
        """ A 部分的核心邏輯：優先級決策樹 """
        decision = None

        # 1. 生存優先 (Supply)
        if (self.bot.supply_left < 5 and self.bot.supply_cap < 200 
            and self.bot.structures(UnitTypeId.SUPPLYDEPOT).not_ready.amount == 0):
            if self.bot.can_afford(UnitTypeId.SUPPLYDEPOT):
                decision = ("BUILD", UnitTypeId.SUPPLYDEPOT)

        # 2. 建築優先 (Structure) - 包含 兵營、瓦斯廠、科技實驗室
        if not decision:
            for s_id, goal in self.target_structures.items():
                # 這裡的 amount 會計算 (已完成 + 建造中) 的數量
                if self.bot.structures(s_id).amount < goal:
                    if self.bot.can_afford(s_id):
                        decision = ("BUILD", s_id)
                        break

        # 3. 單位優先 (Unit) - 這裡就是造掠奪者
        if not decision:
            for u_id, goal in self.target_units.items():
                if self.bot.units(u_id).amount < goal:
                    if self.bot.can_afford(u_id):
                        decision = ("TRAIN", u_id)
                        break

        # 4. 記錄數據 (Log)
        self.collector.log_step(
            time=self.bot.time,
            minerals=self.bot.minerals,
            vespene=self.bot.vespene,
            supply=self.bot.supply_used,
            count=self.bot.units(UnitTypeId.MARAUDER).amount, # 記錄掠奪者數量
            decision=decision
        )

        return decision

# =========================================================
# 🤖 主程式: 掠奪者專題機器人 (MarauderBot)
# 用途: 設定目標，並模擬 B 部分的執行
# =========================================================
class MarauderBot(BotAI):
    def __init__(self):
        self.brain = ProductionAI(self)

    async def on_step(self, iteration):
        # 0. 基礎運作：工兵自動挖礦
        await self.distribute_workers()

        # ==========================================
        # 🎯 [專題目標設定]
        # 這裡告訴 A 大腦：我要 5 隻掠奪者，你需要準備什麼設施
        # ==========================================
        self.brain.set_goals(
            # 目標單位
            units={
                UnitTypeId.MARAUDER: 5 
            },
            # 目標設施 (掠奪者需要：兵營 -> 瓦斯 -> 科技實驗室)
            structures={
                UnitTypeId.BARRACKS: 2,         # 2 座兵營
                UnitTypeId.REFINERY: 1,         # 1 座瓦斯廠 (一定要有，不然沒瓦斯)
                UnitTypeId.BARRACKSTECHLAB: 2   # 2 個科技掛件 (一定要有，不然不能造)
            }
        )

        # 1. 呼叫 A 大腦做決策
        decision = self.brain.make_decision()

        # 2. 執行決策 (模擬 B 部分)
        if decision:
            action, target = decision
            
            # 在終端機印出指令，讓你確認 A 是不是正常運作
            print(f"[{self.time:.1f}s] A發出指令: {action} -> {target}")

            if action == "BUILD":
                # --- 蓋建築邏輯 ---
                if target == UnitTypeId.SUPPLYDEPOT:
                    await self.build(target, near=self.townhalls.first)
                
                elif target == UnitTypeId.BARRACKS:
                    await self.build(target, near=self.townhalls.first)
                
                elif target == UnitTypeId.REFINERY:
                    # 找離家最近的瓦斯泉蓋
                    for vg in self.vespene_geyser.closer_than(10, self.townhalls.first):
                        if not self.structures(UnitTypeId.REFINERY).closer_than(1, vg).exists:
                            await self.build(target, vg)
                            break
                            
                elif target == UnitTypeId.BARRACKSTECHLAB:
                    # 找一個「沒有掛件」的兵營來蓋實驗室
                    for b in self.structures(UnitTypeId.BARRACKS).ready:
                        if b.add_on_tag == 0:
                            b.build(target)
                            break

            elif action == "TRAIN":
                # --- 造兵邏輯 ---
                if target == UnitTypeId.MARAUDER:
                    # 找一個「有掛科技實驗室」且「閒置」的兵營來生產
                    producers = self.structures(UnitTypeId.BARRACKS).ready.idle
                    for b in producers:
                        if b.has_techlab: 
                            b.train(target)
                            break

if __name__ == "__main__":
    run_game(
        maps.get("Simple64"),
        [Bot(Race.Terran, MarauderBot()), Computer(Race.Zerg, Difficulty.Easy)],
        realtime=True
    )