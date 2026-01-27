#!/usr/bin/env python3
"""
測試新的獎勵系統實現
這個腳本測試新的獎勵系統是否正確實現了用戶要求的功能：
1. 新的分數計算方式
2. 上限機制（補給站只有前3個給分）
3. 歷史最大值比較邏輯
4. 無效動作懲罰
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入必要的模組
from chx_production_AI_learning import RewardSystem, SUPPLY_DEPOT_ID, BARRACKS_ID, BARRACKS_TECHLAB_ID, REFINERY_ID, SCV_ID, MARAUDER_ID

def create_mock_obs(minerals=500, vespene=500, supply_depots=0, barracks=0, techlabs=0, refineries=0, scvs=0, marauders=0):
    """創建一個模擬的觀察對象用於測試"""
    class MockPlayer:
        def __init__(self, minerals, vespene):
            self.minerals = minerals
            self.vespene = vespene
            self.food_workers = scvs
            self.food_used = scvs + marauders * 2  # SCV用1供應，掠奪者用2供應
            self.food_cap = 15 + supply_depots * 8  # 基本15供應 + 每個補給站+8供應

    class MockFeatures:
        """模擬 features 模組"""
        class SCREEN_FEATURES:
            unit_type = type('obj', (object,), {'index': 6})()  # 正確的 unit_type index

    class MockObservation:
        def __init__(self, player):
            self.player = player
            # 創建一個模擬的unit_type數組
            self.feature_screen = [None] * 10  # 創建足夠的槽位
            # 創建一個模擬的unit_type數組，包含指定數量的建築和單位
            unit_type_array = np.zeros((84, 84), dtype=np.int32)

            # 添加補給站
            for i in range(supply_depots):
                if i < len(unit_type_array):
                    unit_type_array[i, 0] = SUPPLY_DEPOT_ID

            # 添加兵營
            for i in range(barracks):
                if i < len(unit_type_array):
                    unit_type_array[i, 1] = BARRACKS_ID

            # 添加科技實驗室
            for i in range(techlabs):
                if i < len(unit_type_array):
                    unit_type_array[i, 2] = BARRACKS_TECHLAB_ID

            # 添加瓦斯廠
            for i in range(refineries):
                if i < len(unit_type_array):
                    unit_type_array[i, 3] = REFINERY_ID

            # 添加SCV
            for i in range(scvs):
                if i < len(unit_type_array):
                    unit_type_array[i, 4] = SCV_ID

            # 添加掠奪者
            for i in range(marauders):
                if i < len(unit_type_array):
                    unit_type_array[i, 5] = MARAUDER_ID

            self.feature_screen[6] = unit_type_array  # 使用正確的 index

    class MockObs:
        def __init__(self, minerals, vespene, supply_depots, barracks, techlabs, refineries, scvs, marauders):
            self.observation = MockObservation(MockPlayer(minerals, vespene))

    return MockObs(minerals, vespene, supply_depots, barracks, techlabs, refineries, scvs, marauders)

def test_scoring_system():
    """測試新的獎勵系統"""
    print("🧪 測試新的獎勵系統實現")
    print("=" * 50)

    reward_system = RewardSystem()

    # 測試1：造出一隻掠奪者 +50 (大獎)
    print("測試1：造出一隻掠奪者 +50 (大獎)")
    # 首先沒有掠奪者
    obs = create_mock_obs(marauders=0)
    reward_system.calculate_reward(obs, 0, 0)
    # 然後有掠奪者了
    obs = create_mock_obs(marauders=1)
    reward = reward_system.calculate_reward(obs, 0, 1)
    print(f"✅ 造出1隻掠奪者，獎勵: {reward}")
    # 由於可能有其他建築也觸發獎勵，我們主要檢查是否包含掠奪者的大獎
    assert reward >= 49.9, f"期望獎勵至少為49.9（包含50大獎-0.1時間懲罰），實際: {reward}"

    # 測試2：蓋出兵營 +10 (中獎)
    print("\n測試2：蓋出兵營 +10 (中獎)")
    reward_system.reset()
    # 首先沒有兵營
    obs = create_mock_obs(barracks=0)
    reward_system.calculate_reward(obs, 0, 0)
    # 然後有兵營了
    obs = create_mock_obs(barracks=1)
    reward = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 蓋出兵營，獎勵: {reward}")
    assert reward == 9.9, f"期望獎勵約為9.9（10-0.1時間懲罰），實際: {reward}"

    # 測試3：蓋出科技實驗室 +10 (中獎)
    print("\n測試3：蓋出科技實驗室 +10 (中獎)")
    reward_system.reset()
    # 首先沒有科技實驗室
    obs = create_mock_obs(techlabs=0)
    reward_system.calculate_reward(obs, 0, 0)
    # 然後有科技實驗室了
    obs = create_mock_obs(techlabs=1)
    reward = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 蓋出科技實驗室，獎勵: {reward}")
    assert reward == 9.9, f"期望獎勵約為9.9（10-0.1時間懲罰），實際: {reward}"

    # 測試4：蓋出瓦斯廠 +5 (小獎)
    print("\n測試4：蓋出瓦斯廠 +5 (小獎)")
    reward_system.reset()
    # 首先沒有瓦斯廠
    obs = create_mock_obs(refineries=0)
    reward_system.calculate_reward(obs, 0, 0)
    # 然後有瓦斯廠了
    obs = create_mock_obs(refineries=1)
    reward = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 蓋出瓦斯廠，獎勵: {reward}")
    assert reward == 4.9, f"期望獎勵約為4.9（5-0.1時間懲罰），實際: {reward}"

    # 測試5：蓋出補給站 +2 (小獎) - 上限機制測試
    print("\n測試5：蓋出補給站 +2 (小獎) - 上限機制測試")
    reward_system.reset()

    # 測試前3個補給站
    for i in range(1, 4):
        obs = create_mock_obs(supply_depots=i)
        reward = reward_system.calculate_reward(obs, 0, 0)
        print(f"✅ 蓋出第{i}個補給站，獎勵: {reward}")
        expected = 1.9 if i <= 3 else -0.1
        assert reward == expected, f"第{i}個補給站期望獎勵約為{expected}（2-0.1時間懲罰），實際: {reward}"

    # 測試第4個補給站（應該不給分）
    obs = create_mock_obs(supply_depots=4)
    reward = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 蓋出第4個補給站，獎勵: {reward}")
    assert reward == -0.1, f"第4個補給站期望獎勵為-0.1（只有時間懲罰），實際: {reward}"

    # 測試6：造出一隻工兵 (SCV) +1 (小小獎)
    print("\n測試6：造出一隻工兵 (SCV) +1 (小小獎)")
    reward_system.reset()
    # 首先沒有SCV
    obs = create_mock_obs(scvs=0)
    reward_system.calculate_reward(obs, 0, 0)
    # 然後有SCV了
    obs = create_mock_obs(scvs=1)
    reward = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 造出1隻SCV，獎勵: {reward}")
    assert reward == 0.9, f"期望獎勵約為0.9（1-0.1時間懲罰），實際: {reward}"

    # 測試7：無效動作 (錢不夠亂按) -1 (懲罰)
    print("\n測試7：無效動作 (錢不夠亂按) -1 (懲罰)")
    reward_system.reset()
    # 測試資源不足的情況
    obs = create_mock_obs(minerals=10, vespene=10)  # 很少資源
    reward = reward_system.calculate_reward(obs, 1, 0)  # 試圖訓練SCV（需要50礦物）
    print(f"✅ 資源不足試圖訓練SCV，獎勵: {reward}")
    assert reward == -1.1, f"期望獎勵約為-1.1（-1無效動作懲罰-0.1時間懲罰），實際: {reward}"

    # 測試8：歷史最大值比較邏輯
    print("\n測試8：歷史最大值比較邏輯")
    reward_system.reset()

    # 首先建造1個兵營
    obs = create_mock_obs(barracks=1)
    reward1 = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 第一次建造兵營，獎勵: {reward1}")

    # 再次報告1個兵營（不應該再給分，因為沒有增加）
    obs = create_mock_obs(barracks=1)
    reward2 = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 第二次報告1個兵營（無變化），獎勵: {reward2}")

    # 建造第2個兵營（應該再給分）
    obs = create_mock_obs(barracks=2)
    reward3 = reward_system.calculate_reward(obs, 0, 0)
    print(f"✅ 建造第2個兵營，獎勵: {reward3}")

    # 驗證邏輯：只有增加時才給分
    assert reward1 == 9.9, f"第一次建造兵營應該給分，實際: {reward1}"
    assert reward2 == -0.1, f"沒有變化不應該給分，實際: {reward2}"
    assert reward3 == 9.9, f"第二次建造兵營應該再給分，實際: {reward3}"

    print("\n🎉 所有測試通過！新的獎勵系統實現正確。")
    print("\n新的獎勵系統功能總結：")
    print("1. ✅ 造出一隻掠奪者 +50 (大獎)")
    print("2. ✅ 蓋出兵營 +10 (中獎)")
    print("3. ✅ 蓋出科技實驗室 +10 (中獎)")
    print("4. ✅ 蓋出瓦斯廠 +5 (小獎)")
    print("5. ✅ 蓋出補給站 +2 (小獎) - 只有前3個給分")
    print("6. ✅ 造出一隻工兵 (SCV) +1 (小小獎)")
    print("7. ✅ 無效動作 (錢不夠亂按) -1 (懲罰)")
    print("8. ✅ 歷史最大值比較邏輯 - 只有增加時才給分")

if __name__ == "__main__":
    test_scoring_system()
