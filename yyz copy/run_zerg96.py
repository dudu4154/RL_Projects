from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2 import maps

from zerg_enemy96_bot import ZergEnemy96Bot


def get_map():
    for name in ["Simple64_v96", "Simple96", "Simple64"]:
        try:
            print(f"使用地圖: {name}")
            return maps.get(name)
        except:
            pass
    raise RuntimeError("找不到地圖")


if __name__ == "__main__":
    run_game(
        get_map(),
        [
            Bot(Race.Zerg, ZergEnemy96Bot()),
            Computer(Race.Protoss, Difficulty.Medium),
        ],
        realtime=False,
    )