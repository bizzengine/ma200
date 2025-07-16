# app.py
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime # Import datetime

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    table = []
    ticker = ""
    stats = {}
    error = None

    # GET 요청 (ma200 페이지에서 링크 클릭 시) 또는 POST 요청 (검색 폼 제출 시)
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper().strip()
    else: # GET 요청일 경우 (URL 쿼리 파라미터 확인)
        ticker = request.args.get('ticker', '').upper().strip()

    if ticker: # 티커가 존재할 경우에만 분석 실행
        try:
            data = yf.download(ticker, start='2020-01-01', auto_adjust=False)
            if data.empty:
                error = f"'{ticker}'에 대한 데이터를 찾을 수 없습니다."
                return render_template('index.html', table=table, ticker=ticker, error=error, stats=stats)

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel('Ticker')
            
            if len(data) < 200: # 200일선 계산을 위한 최소 데이터 길이
                error = f"'{ticker}'에 대한 데이터가 충분하지 않아 200일선을 계산할 수 없습니다."
                return render_template('index.html', table=table, ticker=ticker, error=error, stats=stats)


            data['200MA'] = data['Close'].rolling(window=200, min_periods=1).mean()
            data = data.dropna(subset=['200MA']) # 200MA가 계산되지 않은 초기 부분 제거

            # 200MA 상승 돌파 지점
            data['prev_above'] = data['Close'].shift(1) > data['200MA'].shift(1)
            data['now_above'] = data['Close'] > data['200MA']
            data['cross_up'] = (~data['prev_above']) & (data['now_above'])
            cross_ups = data[data['cross_up']].copy()

            results = []
            for idx, cross_date in enumerate(cross_ups.index):
                row = {}
                row['돌파날짜'] = cross_date.date()
                row['돌파종가'] = round(data.loc[cross_date, 'Close'], 2)
                row['돌파200MA'] = round(data.loc[cross_date, '200MA'], 2)

                # 돌파 후 다음 하락돌파 구간
                if idx+1 < len(cross_ups):
                    next_cross = cross_ups.index[idx+1]
                    sub_data = data.loc[cross_date:next_cross]
                else:
                    sub_data = data.loc[cross_date:]

                if sub_data.empty: # sub_data가 비어있으면 다음으로 넘어감
                    continue

                sub_data = sub_data.copy()
                sub_data['prev_below'] = sub_data['Close'].shift(1) < sub_data['200MA'].shift(1)
                sub_data['now_below'] = sub_data['Close'] < sub_data['200MA']
                sub_data['cross_down'] = (~sub_data['prev_below']) & (sub_data['now_below'])
                cross_downs = sub_data[sub_data['cross_down']]

                if not cross_downs.empty:
                    down_date = cross_downs.index[0]
                    if cross_date in data.index and down_date in data.index:
                        row['다시 내려온 날'] = down_date.date()
                        row['내려올 때 가격'] = round(data.loc[down_date, 'Close'], 2)
                        row['그날 200일선'] = round(data.loc[down_date, '200MA'], 2)
                        row['점프 구간 수익률'] = round((data.loc[down_date, 'Close'] / data.loc[cross_date, 'Close'] - 1) * 100, 2)
                        row['점프 기간'] = (data.index.get_loc(down_date) - data.index.get_loc(cross_date))
                    else: # 데이터 불완전
                        row['다시 내려온 날'] = ""
                        row['내려올 때 가격'] = ""
                        row['그날 200일선'] = ""
                        row['점프 구간 수익률'] = 0.0
                        row['점프 기간'] = 0
                else:
                    row['다시 내려온 날'] = ""
                    if cross_date in data.index and not data.empty:
                        curr_close = data['Close'].iloc[-1]
                        row['내려올 때 가격'] = round(curr_close, 2) # 현재 가격으로 설정
                        row['그날 200일선'] = round(data['200MA'].iloc[-1], 2) # 현재 200MA로 설정
                        row['점프 구간 수익률'] = round((curr_close / data.loc[cross_date, 'Close'] - 1) * 100, 2)
                        row['점프 기간'] = (data.index.get_loc(data.index[-1]) - data.index.get_loc(cross_date))
                    else: # 데이터 불완전
                        row['내려올 때 가격'] = ""
                        row['그날 200일선'] = ""
                        row['점프 구간 수익률'] = 0.0
                        row['점프 기간'] = 0
                results.append(row)
            table = sorted(results, key=lambda x: x['돌파날짜'], reverse=True)

            # ---- 요약 통계 6개 계산 ----
            returns = [row['점프 구간 수익률'] for row in table if isinstance(row['점프 구간 수익률'], (int, float))]
            durations = [row['점프 기간'] for row in table if isinstance(row['점프 기간'], (int, float))]
            current_close = float(data['Close'].iloc[-1])
            current_200ma = float(data['200MA'].iloc[-1])
            distance = round((current_close - current_200ma) / current_200ma * 100, 2) if current_200ma else None

            stats = {
                'avg_return': round(np.mean(returns), 2) if returns else None,
                'max_return': round(np.max(returns), 2) if returns else None,
                'min_return': round(np.min(returns), 2) if returns else None,
                'current_close': round(current_close, 2) if returns else None,  # 현재 종가 추가
                'distance': distance,
                'total_jumps': len(returns),
            }

        except Exception as e:
            error = f"분석 오류: {str(e)}"
    return render_template('index.html', table=table, ticker=ticker, error=error, stats=stats)

@app.route('/ma200')
def ma200():
    table = []
    error = None
    last_update = None
    try:
        # CSV에서 불러오기
        df = pd.read_csv('recent_ohlc.csv')

        # 'tickers.json' 파일에서 랭킹 정보 불러오기
        tickers_df = pd.DataFrame()
        try:
            with open('tickers.json', 'r', encoding='utf-8') as f:
                tickers_data = json.load(f)
            tickers_df = pd.DataFrame(tickers_data).set_index('symbol')
            # 기존 df와 랭킹 데이터를 symbol 기준으로 병합
            df = df.set_index('symbol').join(tickers_df['rank'], how='left')
            df = df.reset_index() # symbol을 다시 컬럼으로
            df['rank'] = df['rank'].fillna(999999).astype(int) # 랭킹이 없는 종목은 큰 값 부여

            # ma200_jump_data.json 에서 평균 점프 수익률 데이터 로드
            if os.path.exists('ma200_jump_data.json'):
                with open('ma200_jump_data.json', 'r', encoding='utf-8') as f:
                    jump_data = json.load(f)
                
                # Get modification time of ma200_jump_data.json
                timestamp = os.path.getmtime('ma200_jump_data.json')
                last_update = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                # df에 avg_jump_return 컬럼 추가
                df['avg_jump_return'] = df['symbol'].map(lambda s: jump_data.get(s, {}).get('avg_jump_return'))
            else:
                print("Warning: ma200_jump_data.json not found. Average jump return will not be available.")
                df['avg_jump_return'] = None # 파일 없으면 None으로 설정

        except FileNotFoundError:
            if not error:
                error = "tickers.json 파일을 찾을 수 없습니다. 시가총액 랭킹을 표시할 수 없습니다."
            else:
                error += ". tickers.json 파일을 찾을 수 없습니다. 시가총액 랭킹을 표시할 수 없습니다."
            df['rank'] = 999999 # 랭킹 파일을 찾지 못하면 임시로 큰 값 부여
            df['avg_jump_return'] = None # 파일 없으면 None으로 설정
        except Exception as e:
            if not error:
                error = f"데이터 로드 오류: {e}"
            else:
                error += f". 데이터 로드 오류: {e}"
            df['rank'] = 999999 # 오류 시 임시로 큰 값 부여
            df['avg_jump_return'] = None # 오류 시 None으로 설정


        if 'timestamp' in df.columns:
            # Removed the assignment from df['timestamp'] as per the request to use ma200_jump_data.json's modification time.
            pass 

        df['distance'] = ((df['today_close'] - df['today_200ma']) / df['today_200ma'] * 100).round(2)

        cross_ups_df = df[(df['today_close'] > df['today_200ma']) & (df['yesterday_close'] <= df['yesterday_200ma'])].copy()

        if not cross_ups_df.empty:
            cross_ups_df = cross_ups_df.sort_values(by=['distance', 'rank'], ascending=[False, True])
            # avg_jump_return 컬럼도 포함하여 전달
            table = cross_ups_df[['rank', 'symbol', 'name', 'today_close', 'today_200ma', 'distance', 'avg_jump_return']].to_dict(orient='records')
        else:
            closest_stocks_df = df[df['today_close'] <= df['today_200ma']].copy()
            closest_stocks_df['abs_distance'] = abs(closest_stocks_df['distance'])
            closest_stocks_df = closest_stocks_df.sort_values(by=['abs_distance', 'rank'], ascending=[True, True]).head(20)
            # avg_jump_return 컬럼도 포함하여 전달
            table = closest_stocks_df[['rank', 'symbol', 'name', 'today_close', 'today_200ma', 'distance', 'avg_jump_return']].to_dict(orient='records')
            
            if not error: # 이미 오류가 없으면 새로운 메시지 추가
                error = "오늘 200일선을 돌파한 종목이 없습니다. 대신 200일선에 가장 가까운 20개 종목을 보여드립니다."

    except Exception as e:
        if not error:
            error = f"MA200 조회 오류: {e}"
        else: # 기존 오류가 있으면 추가
            error += f". 전체 MA200 조회 오류: {e}"
    return render_template('ma200.html', table=table, error=error, last_update=last_update)

if __name__ == '__main__':
    app.run(debug=True)