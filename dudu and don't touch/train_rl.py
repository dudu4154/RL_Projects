import os
from stable_baselines3 import PPO
from sc2_env_wrapper import SC2MarauderEnv

def train():
    # 1. 初始化環境
    print("正在啟動星海爭霸 II 環境...")
    env = SC2MarauderEnv()

    # 2. 定義模型
    # PPO 是處理這類離散動作空間最穩定的演算法
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, # 學習率
        tensorboard_log="./ppo_sc2_logs/" # 訓練日誌，可用 Tensorboard 查看
    )

    # 3. 開始訓練
    total_steps = 100000  # 先設定 10 萬步
    print(f"訓練開始，預計執行 {total_steps} 步...")
    
    try:
        model.learn(total_timesteps=total_steps)
        # 4. 存檔
        model.save("sc2_marauder_model_v1")
        print("訓練完成，模型已儲存為 sc2_marauder_model_v1")
    except KeyboardInterrupt:
        print("偵測到手動中斷，正在儲存目前的模型...")
        model.save("sc2_marauder_model_interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    train()