# update_ma200_data.py
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def calculate_jump_stats(ticker_symbol):
    """
    주어진 티커에 대해 200일선 점프 구간 수익률을 계산합니다. (점프 기간은 제외)
    """
    try:
        data = yf.download(ticker_symbol, start='2020-01-01', auto_adjust=False, progress=False)
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel('Ticker')
        
        # 데이터가 충분하지 않으면 (예: 200일 MA를 계산할 수 없음) 스킵
        if len(data) < 200:
            return None

        data['200MA'] = data['Close'].rolling(window=200, min_periods=1).mean()
        data = data.dropna(subset=['200MA']) # 200MA가 계산되지 않은 초기 부분 제거

        data['prev_above'] = data['Close'].shift(1) > data['200MA'].shift(1)
        data['now_above'] = data['Close'] > data['200MA']
        data['cross_up'] = (~data['prev_above']) & (data['now_above'])
        cross_ups = data[data['cross_up']].copy()

        returns = []
        # durations = [] # 점프 기간 계산은 더 이상 필요 없음

        for idx, cross_date in enumerate(cross_ups.index):
            if idx+1 < len(cross_ups):
                next_cross_date = cross_ups.index[idx+1]
                sub_data = data.loc[cross_date:next_cross_date]
            else:
                sub_data = data.loc[cross_date:]

            if sub_data.empty:
                continue

            sub_data = sub_data.copy()
            sub_data['prev_below'] = sub_data['Close'].shift(1) < sub_data['200MA'].shift(1)
            sub_data['now_below'] = sub_data['Close'] < sub_data['200MA'] # <-- 이 부분이 수정되었습니다.
            sub_data['cross_down'] = (~sub_data['prev_below']) & (sub_data['now_below'])
            cross_downs = sub_data[sub_data['cross_down']]

            if not cross_downs.empty:
                down_date = cross_downs.index[0]
                # 돌파 가격과 하향 이탈 가격 모두 유효한지 확인
                if cross_date in data.index and down_date in data.index:
                    jump_return = (data.loc[down_date, 'Close'] / data.loc[cross_date, 'Close'] - 1) * 100
                    returns.append(jump_return)
            else:
                # 아직 내려오지 않은 경우, 현재까지의 수익률 계산
                if cross_date in data.index and not data.empty:
                    curr_close = data['Close'].iloc[-1]
                    jump_return = (curr_close / data.loc[cross_date, 'Close'] - 1) * 100
                    returns.append(jump_return)

        avg_return = round(np.mean(returns), 2) if returns else None
        
        # avg_days는 더 이상 반환하지 않음
        return avg_return

    except Exception as e:
        print(f"Error calculating jump stats for {ticker_symbol}: {e}")
        return None

def update_ma200_jump_data():
    print(f"[{datetime.now()}] Starting MA200 jump data update...")
    
    # recent_ohlc.csv 파일 경로
    ohlc_file = 'recent_ohlc.csv'
    # tickers.json 파일 경로 (랭킹 정보)
    tickers_file = 'tickers.json'
    # 평균 점프 수익률 저장할 파일 경로
    output_file = 'ma200_jump_data.json'

    if not os.path.exists(ohlc_file):
        print(f"Error: {ohlc_file} not found. Cannot update jump data.")
        return

    df = pd.read_csv(ohlc_file)
    
    # tickers.json 불러와서 랭킹 정보 병합
    tickers_df = pd.DataFrame()
    if os.path.exists(tickers_file):
        try:
            with open(tickers_file, 'r', encoding='utf-8') as f:
                tickers_data = json.load(f)
            tickers_df = pd.DataFrame(tickers_data).set_index('symbol')
        except Exception as e:
            print(f"Warning: Could not load or parse {tickers_file}: {e}")
    
    if not tickers_df.empty:
        df = df.set_index('symbol').join(tickers_df['rank'], how='left')
        df = df.reset_index()
        df['rank'] = df['rank'].fillna(999999).astype(int) # 랭킹 없는 경우 큰 값 부여
    else:
        df['rank'] = 999999 # 랭킹 파일 없으면 큰 값 부여

    df['distance'] = ((df['today_close'] - df['today_200ma']) / df['today_200ma'] * 100).round(2)

    cross_ups_df = df[(df['today_close'] > df['today_200ma']) & (df['yesterday_close'] <= df['yesterday_200ma'])].copy()

    relevant_symbols = set()
    if not cross_ups_df.empty:
        cross_ups_df = cross_ups_df.sort_values(by=['distance', 'rank'], ascending=[False, True])
        relevant_symbols.update(cross_ups_df['symbol'].head(200).tolist())
    else:
        closest_stocks_df = df[df['today_close'] <= df['today_200ma']].copy()
        closest_stocks_df['abs_distance'] = abs(closest_stocks_df['distance'])
        closest_stocks_df = closest_stocks_df.sort_values(by=['abs_distance', 'rank'], ascending=[True, True]).head(20)
        relevant_symbols.update(closest_stocks_df['symbol'].tolist())

    jump_data = {}
    total_symbols = len(relevant_symbols)
    processed_count = 0

    print(f"[{datetime.now()}] Identified {total_symbols} relevant symbols for jump analysis.")

    for symbol in relevant_symbols:
        avg_return = calculate_jump_stats(symbol) # avg_days는 이제 반환되지 않음
        if avg_return is not None:
            jump_data[symbol] = {
                'avg_jump_return': avg_return
                # 'avg_jump_days': avg_days # 더 이상 저장하지 않음
            }
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"[{datetime.now()}] Processed {processed_count}/{total_symbols} symbols...")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jump_data, f, ensure_ascii=False, indent=4)
        print(f"[{datetime.now()}] Successfully updated {output_file} with {len(jump_data)} entries.")
    except Exception as e:
        print(f"[{datetime.now()}] Error saving {output_file}: {e}")

if __name__ == '__main__':
    update_ma200_jump_data()