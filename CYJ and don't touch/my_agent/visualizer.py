import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import json

def load_data(log_folder="logs"):
    # 遞迴搜尋所有子資料夾內的 train_history.jsonl
    files = glob.glob(os.path.join(log_folder, "**/train_history.jsonl"), recursive=True)
    all_runs = []

    for f in files:
        run_data = []
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                # 只讀取每一局的總結 (summary)
                if '"type": "summary"' in line:
                    run_data.append(json.loads(line))
        
        if run_data:
            df = pd.DataFrame(run_data)
            # 用資料夾名稱當作 Run ID
            df['run_id'] = os.path.dirname(f).split(os.sep)[-1]
            all_runs.append(df)
    
    return pd.concat(all_runs, ignore_index=True) if all_runs else None

def plot_all(log_folder="logs"):
    df = load_data(log_folder)
    if df is None:
        print("❌ 沒找到任何 Log 資料，請確認路徑。")
        return

    # 確保資料依 Episode 排序
    df = df.sort_values('ep')

    # 計算平均與標準差
    stats = df.groupby('ep')['total_rew'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 6))

    # 1. 畫出 N 筆實驗的陰影區 (標準差)
    plt.fill_between(
        stats['ep'], 
        stats['mean'] - stats['std'], 
        stats['mean'] + stats['std'], 
        color='blue', alpha=0.15, label='Standard Deviation'
    )

    # 2. 畫出平均獎勵曲線
    plt.plot(stats['ep'], stats['mean'], color='blue', linewidth=2, label='Mean Reward')

    # 3. 圖表修飾
    plt.title(f'Multi-Run Training Analysis (N={len(df["run_id"].unique())})', fontsize=14)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    output_path = "training_analysis.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ 圖像化分析完成：{output_path}")
    plt.show()

if __name__ == "__main__":
    plot_all()