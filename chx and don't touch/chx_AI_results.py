import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 讀取 CSV 成績單
csv_file = "training_performance.csv"

if not os.path.exists(csv_file):
    print("❌ 錯誤：找不到 training_performance.csv")
    print("請先執行 marauder_learner.py 進行訓練！")
    exit()

df = pd.read_csv(csv_file)
print(f"✅ 成功讀取數據，共 {len(df)} 場紀錄")

# 2. 轉存為 Excel 報表
excel_filename = "AI_Training_Report.xlsx"
df.to_excel(excel_filename, index=False)
print(f"📊 Excel 報表已生成: {excel_filename}")

# 3. 繪製學習曲線圖
plt.figure(figsize=(10, 6))

# 繪製「掠奪者數量」曲線
plt.plot(df["Episode"], df["Marauders_Created"], 
         marker='o', linestyle='-', color='blue', linewidth=2, label='Marauders Created')

plt.title("AI Training Learning Curve (Terran Marauders)", fontsize=16)
plt.xlabel("Episode (Game Round)", fontsize=12)
plt.ylabel("Number of Marauders", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 儲存圖片
plt.savefig("learning_curve.png")
print("📈 曲線圖已儲存: learning_curve.png")

# 顯示圖片
plt.show()