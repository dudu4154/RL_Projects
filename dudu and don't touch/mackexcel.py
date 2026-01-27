import pandas as pd
import glob
import os

# 1. 找到最新的一份訓練紀錄
log_files = glob.glob("log/dqn_training_log_*.csv")
if not log_files:
    print("找不到 log 檔案，請確認路徑。")
else:
    latest_log = max(log_files, key=os.path.getctime)
    df = pd.read_csv(latest_log)
    
    # 2. 建立 Excel 檔案與圖表引擎
    excel_file = "AI_Training_Progress.xlsx"
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='學習紀錄', index=False)
    
    workbook  = writer.book
    worksheet = writer.sheets['學習紀錄']
    max_row = len(df)

    # 3. 繪製「總獎勵趨勢圖」(折線圖)
    chart_reward = workbook.add_chart({'type': 'line'})
    chart_reward.add_series({
        'name':       '總獎勵 (Total Reward)',
        'categories': ['學習紀錄', 1, 0, max_row, 0], # Episode
        'values':     ['學習紀錄', 1, 1, max_row, 1], # Total_Reward
        'line':       {'color': '#1f77b4'},
    })
    chart_reward.set_title({'name': 'AI 進化趨勢：總獎勵'})
    chart_reward.set_x_axis({'name': 'Episode (回合)'})
    chart_reward.set_y_axis({'name': 'Reward (分數)'})
    worksheet.insert_chart('G2', chart_reward)

    # 4. 繪製「狩獵者產量圖」(長條圖)
    chart_marauders = workbook.add_chart({'type': 'column'})
    chart_marauders.add_series({
        'name':       '產出數量 (Marauders)',
        'categories': ['學習紀錄', 1, 0, max_row, 0],
        'values':     ['學習紀錄', 1, 2, max_row, 2],
        'fill':       {'color': '#ff7f0e'},
    })
    chart_marauders.set_title({'name': 'AI 生產力：狩獵者數量'})
    chart_marauders.set_x_axis({'name': 'Episode (回合)'})
    chart_marauders.set_y_axis({'name': 'Count (隻)'})
    worksheet.insert_chart('G18', chart_marauders)

    writer.close()
    print(f"✅ 圖表已產生：{excel_file}")