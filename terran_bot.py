import sc2
import os
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
# ---------------------------------------------------
# 【手動補丁區：手動定義遊戲內的 ID】
# 因為你的環境缺少某些檔案，我們這裡手動告訴 Python 這些技能的編號
# 這些是星海爭霸遊戲引擎內部的代碼
from enum import Enum

class AbilityId(Enum):
    CALLDOWNMULE_CALLDOWNMULE = 171        # 技能：星軌指揮部丟礦騾 (增加收入)
    UPGRADETOORBITAL_ORBITALCOMMAND = 1516 # 技能：主堡升級成星軌指揮部
    RESEARCH_STIMPACK = 3652               # 科技：研發興奮劑 (消耗血量換攻速)
    RESEARCH_COMBATSHIELD = 3654           # 科技：研發戰鬥盾牌 (陸戰隊血量 +10)
    EFFECT_STIM_MARINE = 380               # 效果：施放興奮劑 (讓槍兵打針)

# ---------------------------------------------------
# 【環境設定】
# 設定遊戲路徑，確保 Python 找得到遊戲執行檔
os.environ["SC2PATH"] = r"D:\StarCraft II"
# ---------------------------------------------------

class AdvancedTerranBot(BotAI):
    # 【主迴圈】
    # 這是機器人的大腦，遊戲中每一幀 (Frame) 都會執行一次這個函式
    async def on_step(self, iteration):
        # 1. 工兵管理：讓閒置的工兵自動去挖礦
        await self.distribute_workers()
        
        # 2. 經濟發展：生產工兵、使用礦騾加速挖礦
        await self.manage_economy()
        
        # 3. 補給站管理：避免卡人口 (Supply Block)
        await self.build_supply_depots()
        
        # 4. 瓦斯採集：為了研發高科技和製造醫療機，需要瓦斯
        await self.build_refineries()
        
        # 5. 生產建築鏈：兵營 -> 重工廠 -> 星際港 (科技樹順序)
        await self.build_production()
        
        # 6. 科技研發：讓兵營掛上實驗室，並研發興奮劑與盾牌
        await self.research_tech()
        
        # 7. 軍隊生產：訓練陸戰隊與醫療運輸機
        await self.train_army()
        
        # 8. 戰鬥邏輯：包含「防守家園」與「集結進攻」
        await self.attack_and_defend()

    # --- 以下是各個功能的詳細實作與戰術解說 ---

    async def manage_economy(self):
        # 遍歷所有的指揮中心 (包含普通主堡與星軌指揮部)
        for cc in self.townhalls:
            # 【生產 SCV】
            # 條件：主堡沒事做 + 錢夠 (50礦) + 工兵還沒滿 22 隻
            # 戰術：22 隻是單礦區的飽和運作數 (16採礦+6採氣)，超過效率會變低
            if cc.is_idle and self.can_afford(UnitTypeId.SCV) and self.workers.amount < 22:
                cc.train(UnitTypeId.SCV)

            # 【丟礦騾 (MULE)】
            # 條件：必須是升級過的「星軌指揮部」 + 能量 >= 50
            # 戰術：礦騾採礦速度是 SCV 的好幾倍，這是人族經濟爆發的關鍵
            if cc.type_id == UnitTypeId.ORBITALCOMMAND and cc.energy >= 50:
                # 尋找附近的礦區
                minerals = self.mineral_field.closer_than(10, cc)
                if minerals:
                    # 對著最大塊的礦丟下去
                    cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, minerals.largest)

    async def build_supply_depots(self):
        # 【智慧蓋補給站】
        # 條件：剩餘人口 < 5 (快卡了) 且 總人口 < 200 (遊戲上限)
        if self.supply_left < 5 and self.supply_cap < 200:
            # 限制：場上沒有「正在建造中」的補給站才准蓋新的
            # 目的：防止機器人因為恐慌一次蓋下 10 個補給站把錢花光
            if self.structures(UnitTypeId.SUPPLYDEPOT).not_ready.amount == 0:
                if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                    # 在主堡附近找位置蓋
                    await self.build(UnitTypeId.SUPPLYDEPOT, near=self.townhalls.first)

    async def build_refineries(self):
        # 【瓦斯採集策略】
        # 條件：當我們擁有 2 座以上的兵營時，代表要進入科技期，需要大量瓦斯
        if self.structures(UnitTypeId.BARRACKS).amount >= 2:
            for cc in self.townhalls:
                # 找出離這個主堡最近的瓦斯泉
                geysers = self.vespene_geyser.closer_than(10, cc)
                for geyser in geysers:
                    # 檢查這座瓦斯泉上面是不是已經蓋了工廠
                    if not self.structures(UnitTypeId.REFINERY).closer_than(1.0, geyser).exists:
                        if self.can_afford(UnitTypeId.REFINERY):
                            await self.build(UnitTypeId.REFINERY, geyser)

    async def build_production(self):
        # 【人族科技樹 (Tech Tree) 建造邏輯】
        
        # 1. 升級星軌指揮部 (Orbital Command)
        # 條件：有兵營 (解鎖條件) + 有 150 礦 (手動檢查資源)
        if self.structures(UnitTypeId.BARRACKS).ready.exists:
            if self.minerals >= 150:
                # 找還沒升級的普通主堡
                for cc in self.townhalls.idle:
                    if cc.type_id == UnitTypeId.COMMANDCENTER:
                        cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

        # 2. 建造兵營 (Barracks) - 基礎兵種生產
        # 條件：有補給站 + 目前兵營少於 3 座
        if self.structures(UnitTypeId.SUPPLYDEPOT).ready.exists:
            if self.structures(UnitTypeId.BARRACKS).amount < 3:
                # 避免卡錢，一次只蓋一座
                if self.can_afford(UnitTypeId.BARRACKS) and self.structures(UnitTypeId.BARRACKS).not_ready.amount == 0:
                    await self.build(UnitTypeId.BARRACKS, near=self.townhalls.first)

        # 3. 建造重工廠 (Factory) - 進階兵種與機場的前置條件
        # 戰術：我們不出坦克，蓋重工廠純粹是為了能蓋星際港 (Starport)
        # 條件：有 2 座兵營 + 瓦斯足夠
        if self.structures(UnitTypeId.BARRACKS).amount >= 2 and self.vespene >= 100:
            if self.structures(UnitTypeId.FACTORY).amount < 1:
                # 確保一次只蓋一座
                if self.can_afford(UnitTypeId.FACTORY) and self.structures(UnitTypeId.FACTORY).not_ready.amount == 0:
                    await self.build(UnitTypeId.FACTORY, near=self.townhalls.first)

        # 4. 建造星際港 (Starport) - 生產醫療運輸機
        # 條件：重工廠已經蓋好了 (Ready)
        if self.structures(UnitTypeId.FACTORY).ready.exists:
            if self.structures(UnitTypeId.STARPORT).amount < 1:
                if self.can_afford(UnitTypeId.STARPORT) and self.structures(UnitTypeId.STARPORT).not_ready.amount == 0:
                    await self.build(UnitTypeId.STARPORT, near=self.townhalls.first)

    async def research_tech(self):
        # 【科技研發邏輯】
        
        # 1. 幫兵營掛上「科技實驗室 (Tech Lab)」
        # 只有掛了這個，才能研發興奮劑
        for barrack in self.structures(UnitTypeId.BARRACKS).ready:
            # add_on_tag == 0 代表這個兵營旁邊空空的，沒有掛件
            if barrack.add_on_tag == 0:
                if self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                    barrack.build(UnitTypeId.BARRACKSTECHLAB)
    
        # 2. 研發升級 (手動檢查資源，避免報錯)
        if self.minerals >= 100 and self.vespene >= 100:
            for tech_lab in self.structures(UnitTypeId.BARRACKSTECHLAB).ready:
                if tech_lab.is_idle: # 實驗室閒著才下指令
                    # 優先順序 A: 研發興奮劑 (Stimpack)
                    # 這是陸戰隊的靈魂科技
                    tech_lab(AbilityId.RESEARCH_STIMPACK)
                    
                    # 優先順序 B: 研發戰鬥盾牌 (Combat Shield)
                    # 如果興奮劑已經研發過 (或正在研發)，這行指令會變成有效
                    # 增加 10 點血量，防止被坦克一砲炸死
                    tech_lab(AbilityId.RESEARCH_COMBATSHIELD)

    async def train_army(self):
        # 【部隊生產邏輯】
        
        # 1. 訓練陸戰隊 (Marine)
        for barrack in self.structures(UnitTypeId.BARRACKS).ready:
            # 重要：如果兵營正在研發科技，千萬別生兵插隊，不然科技會暫停！
            if barrack.has_add_on and not barrack.is_idle:
                continue
            
            # 正常生兵
            if barrack.is_idle and self.can_afford(UnitTypeId.MARINE):
                barrack.train(UnitTypeId.MARINE)

        # 2. 訓練醫療運輸機 (Medivac)
        # 條件：星際港蓋好了
        for starport in self.structures(UnitTypeId.STARPORT).ready:
            # 只要錢夠就造，醫療機不嫌多
            if starport.is_idle and self.can_afford(UnitTypeId.MEDIVAC):
                starport.train(UnitTypeId.MEDIVAC)

    async def attack_and_defend(self):
        # 取得我方所有部隊
        marines = self.units(UnitTypeId.MARINE)
        medivacs = self.units(UnitTypeId.MEDIVAC)
        army = marines + medivacs # 混合編隊

        if not army: return # 如果沒兵，就不用執行下面邏輯

        # --- A. 防守邏輯 (Defend) ---
        # 偵測：是否有敵軍進入我方主堡 30 格範圍內
        enemy_near_base = self.enemy_units.closer_than(30, self.townhalls.first)
        
        if enemy_near_base.exists:
            # 緊急狀況：全軍回防！
            for unit in army:
                # 攻擊離自己最近的入侵者
                unit.attack(enemy_near_base.closest_to(unit))
            return # 防守優先，執行完防守就跳出，不執行下面的進攻

        # --- B. 進攻邏輯 (Attack) ---
        # 條件：陸戰隊超過 15 隻 且 至少有 1 台醫療機
        # 戰術：只有槍兵太脆，必須等醫療機出來才出門
        if marines.amount > 15 and medivacs.amount >= 1:
            for unit in army:
                # 特殊操作：陸戰隊打針 (Stimpack)
                if unit.type_id == UnitTypeId.MARINE:
                    # 條件 1: 血量健康 (>80%)
                    # 條件 2: 正在開火 (weapon_cooldown > 0，代表遇到敵人了)
                    # 條件 3: 身上沒有興奮劑效果 (has_buff 檢查，避免重複打針自殺)
                    if unit.health_percentage > 0.8 and unit.weapon_cooldown > 0 and not unit.has_buff(AbilityId.EFFECT_STIM_MARINE):
                        unit(AbilityId.EFFECT_STIM_MARINE)

                # 全軍攻擊敵人出生點
                if self.enemy_start_locations:
                    unit.attack(self.enemy_start_locations[0])

# --- 程式進入點 ---
if __name__ == "__main__":
    print("生化部隊 AI (全註解教學版) 啟動中...")
    run_game(
        maps.get("Simple64"),
        [
            Bot(Race.Terran, AdvancedTerranBot()),
            Computer(Race.Zerg, Difficulty.Medium) # 挑戰中等難度蟲族
        ],
        realtime=True
    )