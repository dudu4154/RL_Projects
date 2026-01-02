@echo off
title SC2 Training Loop

set SC2PATH=D:\Game\StarCraft II

:start
echo ==========================================
echo Starting SC2 Training... 
echo Time: %time%
echo ==========================================

:: ==========================================
:: 2. 設定 Python 執行檔路徑
:: 從你的第二張截圖看，你的虛擬環境在 sc2_new_env
:: 所以我们要直接呼叫那個環境裡的 python.exe
:: ==========================================
"C:\RL_Projects\sc2_new_env\Scripts\python.exe" train_hybrid.py

echo ==========================================
echo WARNING: The script has crashed or finished.
echo Restarting in 10 seconds...
echo Press Ctrl+C to stop completely.
echo ==========================================
timeout /t 3

:: 回到開頭重新執行
goto start