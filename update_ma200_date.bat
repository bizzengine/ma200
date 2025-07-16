@echo off
chcp 65001
cd /d "C:\Users\canmanmo\Desktop\ma200_history"
python update_ma200_data.py
git add .
git commit -m "자동 갱신: 최신 jump 데이터" || exit /b 0
git push origin main