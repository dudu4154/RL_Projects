from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps


class ZergEnemy96Bot(BotAI):
    async def on_start(self):
        print("ZergEnemy96Bot 啟動成功")

    async def on_step(self, iteration: int):
        await self.distribute_workers()

        if not self.townhalls:
            return

        await self.build_overlords()
        await self.build_drones()
        await self.build_spawning_pool()
        await self.build_extractor()
        await self.build_roach_warren()
        await self.expand_if_needed()
        await self.train_zerglings()
        await self.train_roaches()
        await self.defend_near_base()
        await self.attack_logic()

    async def build_overlords(self):
        if self.supply_cap >= 200:
            return

        overlord_pending = self.already_pending(UnitTypeId.OVERLORD)

        # 更早補王蟲，避免卡人口
        if self.supply_left <= 4 and overlord_pending == 0:
            if self.can_afford(UnitTypeId.OVERLORD) and self.larva:
                self.larva.first.train(UnitTypeId.OVERLORD)

    async def build_drones(self):
        if not self.larva:
            return

        drone_target = 28
        if self.townhalls.amount >= 2:
            drone_target = 40

        # 快卡人口時先不要硬補工蜂
        if self.workers.amount < drone_target:
            if self.can_afford(UnitTypeId.DRONE) and self.supply_left > 0:
                self.larva.first.train(UnitTypeId.DRONE)

    async def build_spawning_pool(self):
        if self.structures(UnitTypeId.SPAWNINGPOOL).exists:
            return
        if self.already_pending(UnitTypeId.SPAWNINGPOOL) > 0:
            return
        if self.workers.amount < 13:
            return
        if not self.can_afford(UnitTypeId.SPAWNINGPOOL):
            return
        if not self.townhalls.ready:
            return

        hatch = self.townhalls.ready.first
        await self.build(
            UnitTypeId.SPAWNINGPOOL,
            near=hatch.position.towards(self.game_info.map_center, 6)
        )

    async def build_extractor(self):
        if not self.townhalls.ready:
            return

        if not self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.already_pending(UnitTypeId.SPAWNINGPOOL) == 0:
            return

        max_extractors = 1
        if self.townhalls.amount >= 2:
            max_extractors = 2

        current_extractors = self.gas_buildings.amount + self.already_pending(UnitTypeId.EXTRACTOR)
        if current_extractors >= max_extractors:
            return

        for hatch in self.townhalls.ready:
            geysers = self.vespene_geyser.closer_than(10, hatch)
            for geyser in geysers:
                if not self.can_afford(UnitTypeId.EXTRACTOR):
                    return

                if self.gas_buildings.closer_than(1.0, geyser).exists:
                    continue

                worker = self.select_build_worker(geyser.position)
                if worker:
                    worker.build(UnitTypeId.EXTRACTOR, geyser)
                    return

    async def build_roach_warren(self):
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            return
        if self.structures(UnitTypeId.ROACHWARREN).exists:
            return
        if self.already_pending(UnitTypeId.ROACHWARREN) > 0:
            return
        if not self.can_afford(UnitTypeId.ROACHWARREN):
            return
        if not self.townhalls.ready:
            return

        hatch = self.townhalls.ready.first
        await self.build(
            UnitTypeId.ROACHWARREN,
            near=hatch.position.towards(self.game_info.map_center, 8)
        )

    async def expand_if_needed(self):
        if self.townhalls.amount >= 2:
            return
        if self.already_pending(UnitTypeId.HATCHERY):
            return
        if self.workers.amount < 24:
            return
        if not self.can_afford(UnitTypeId.HATCHERY):
            return

        await self.expand_now()

    async def train_zerglings(self):
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            return
        if not self.larva:
            return

        zergling_target = 12
        if self.structures(UnitTypeId.ROACHWARREN).ready.exists:
            zergling_target = 16

        if self.units(UnitTypeId.ZERGLING).amount < zergling_target:
            if self.can_afford(UnitTypeId.ZERGLING) and self.supply_left >= 1:
                self.larva.first.train(UnitTypeId.ZERGLING)

    async def train_roaches(self):
        if not self.structures(UnitTypeId.ROACHWARREN).ready.exists:
            return
        if not self.larva:
            return

        roach_target = 18
        if self.townhalls.amount >= 2:
            roach_target = 30

        if self.units(UnitTypeId.ROACH).amount < roach_target:
            if self.can_afford(UnitTypeId.ROACH) and self.supply_left >= 2:
                self.larva.first.train(UnitTypeId.ROACH)

    async def defend_near_base(self):
        army = self.units(UnitTypeId.ZERGLING) | self.units(UnitTypeId.ROACH)
        if not army:
            return

        for hatch in self.townhalls.ready:
            enemies = self.enemy_units.closer_than(20, hatch)
            if enemies:
                target = enemies.closest_to(hatch)
                for unit in army.idle:
                    unit.attack(target)
                return

    async def attack_logic(self):
        army = self.units(UnitTypeId.ZERGLING) | self.units(UnitTypeId.ROACH)
        if not army:
            return

        should_attack = (
            self.units(UnitTypeId.ZERGLING).amount >= 12
            or self.units(UnitTypeId.ROACH).amount >= 6
            or army.amount >= 12
        )

        if not should_attack:
            return

        if self.enemy_units:
            target = self.enemy_units.closest_to(army.center)
            for unit in army.idle:
                unit.attack(target)
            return

        if self.enemy_structures:
            target = self.enemy_structures.closest_to(army.center)
            for unit in army.idle:
                unit.attack(target)
            return

        enemy_start = self.enemy_start_locations[0]
        for unit in army.idle:
            unit.attack(enemy_start)


def get_working_map():
    candidate_maps = [
        "Simple64_v96",
        "Simple96",
        "AbyssalReefLE",
        "AbyssalReef",
        "Simple64",
    ]

    for map_name in candidate_maps:
        try:
            selected_map = maps.get(map_name)
            print(f"使用地圖: {map_name}")
            return selected_map
        except Exception:
            pass

    raise RuntimeError(
        "找不到可用地圖。請先確認你的 StarCraft II Maps 資料夾內有 Simple64_v96 或其他可用地圖。"
    )


if __name__ == "__main__":
    selected_map = get_working_map()

    run_game(
        selected_map,
        [
            Bot(Race.Zerg, ZergEnemy96Bot()),
            Computer(Race.Terran, Difficulty.Medium),
        ],
        realtime=False,
    )