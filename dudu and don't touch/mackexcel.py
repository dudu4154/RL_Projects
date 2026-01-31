import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 這裡是你提供的數據資料
csv_data = """Episode,Total_Reward,Marauders,End_Loop,Reason,Is_Bottom_Right
959,0.383,481,0,13440,Timeout,True
960,0.383,31,0,13440,Timeout,False
961,0.383,111,0,13440,Timeout,True
962,0.382,1362,5,11472,Target_Reached,False
963,0.382,371,2,13440,Timeout,True
964,0.382,1284,5,9408,Target_Reached,True
965,0.381,221,1,13440,Timeout,False
966,0.381,671,3,13440,Timeout,False
967,0.380,671,4,13440,Timeout,True
968,0.380,1281,5,13136,Target_Reached,False
969,0.380,631,2,13440,Timeout,True
970,0.379,31,0,13440,Timeout,True
971,0.379,-8,0,13440,Timeout,False
972,0.379,331,3,13440,Timeout,True
973,0.378,521,3,13440,Timeout,False
974,0.378,1282,5,11840,Target_Reached,True
975,0.377,181,1,13440,Timeout,False
976,0.377,521,2,13440,Timeout,False
977,0.377,331,0,13440,Timeout,True
978,0.376,631,4,13440,Timeout,False
979,0.376,481,3,13440,Timeout,False
980,0.376,331,0,13440,Timeout,True
981,0.375,751,4,13440,Timeout,False
982,0.375,481,3,13440,Timeout,True
983,0.374,561,3,13440,Timeout,True
984,0.374,411,0,13440,Timeout,False
985,0.374,711,3,13440,Timeout,True
986,0.373,-8,0,13440,Timeout,False
987,0.373,1281,5,12912,Target_Reached,False
988,0.373,1281,5,13184,Target_Reached,True
989,0.372,1323,5,11120,Target_Reached,False
990,0.372,31,0,13440,Timeout,True
991,0.371,331,0,13440,Timeout,False
992,0.371,1362,5,12128,Target_Reached,False
993,0.371,1282,5,11632,Target_Reached,False
994,0.370,371,0,13440,Timeout,False
995,0.370,1282,5,11984,Target_Reached,True
996,0.370,521,2,13440,Timeout,False
997,0.369,31,0,13440,Timeout,True
998,0.369,1282,5,12288,Target_Reached,False
999,0.368,-8,0,13440,Timeout,True
1000,0.368,1322,5,11536,Target_Reached,False
"""

# 讀取數據
df = pd.read_csv(io.StringIO(csv_data))

# 設定繪圖風格
sns.set_theme(style="darkgrid")

# 建立一個 3x1 的畫布
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# --- 圖表 1: 學習曲線與結果分佈 ---
sns.scatterplot(ax=axes[0], data=df, x="Episode", y="Total_Reward", hue="Reason", palette={"Timeout": "salmon", "Target_Reached": "green"}, alpha=0.6)
# 增加移動平均線來顯示趨勢 (窗口大小為 20)
df['Reward_MA'] = df['Total_Reward'].rolling(window=20).mean()
sns.lineplot(ax=axes[0], data=df, x="Episode", y="Reward_MA", color="blue", label="20-Episode Moving Average")
axes[0].set_title("Fig 1: Total Reward over Episodes (Learning Curve)", fontsize=14)
axes[0].set_ylabel("Total Reward")

# --- 圖表 2: 遊戲時長與獎勵關係圖 ---
sns.scatterplot(ax=axes[1], data=df, x="End_Loop", y="Total_Reward", hue="Reason", palette={"Timeout": "salmon", "Target_Reached": "green"}, s=60)
axes[1].set_title("Fig 2: Duration (End Loop) vs. Total Reward", fontsize=14)
axes[1].set_xlabel("End Loop (Game Steps)")
axes[1].set_ylabel("Total Reward")
# 標示出超時的界線
axes[1].axvline(x=13440, color='gray', linestyle='--', alpha=0.5, label="Timeout Limit (13440)")
axes[1].legend()

# --- 圖表 3: 掠奪者數量對獎勵的影響 ---
# 使用 Boxplot 顯示分佈，並疊加 stripplot 顯示實際數據點
sns.boxplot(ax=axes[2], data=df, x="Marauders", y="Total_Reward", palette="viridis", showfliers=False)
sns.stripplot(ax=axes[2], data=df, x="Marauders", y="Total_Reward", color="black", alpha=0.3, jitter=True)
axes[2].set_title("Fig 3: Impact of Marauder Count on Total Reward", fontsize=14)
axes[2].set_xlabel("Number of Marauders Created")
axes[2].set_ylabel("Total Reward")

# 調整佈局
plt.tight_layout()
plt.show()