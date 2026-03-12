import streamlit as st
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import requests
import time
from datetime import datetime
import lightgbm as lgb

st.set_page_config(page_title="AI Crypto Trader",layout="wide")

DB="ai_trader.db"

conn=sqlite3.connect(DB,check_same_thread=False)
cur=conn.cursor()

# -----------------------------
# DB 생성
# -----------------------------

cur.execute("""
CREATE TABLE IF NOT EXISTS wallet(
id INTEGER PRIMARY KEY,
krw REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS positions(
ticker TEXT PRIMARY KEY,
qty REAL,
buy_price REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS trades(
id INTEGER PRIMARY KEY AUTOINCREMENT,
time TEXT,
ticker TEXT,
price REAL,
qty REAL,
side TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS learning(
id INTEGER PRIMARY KEY AUTOINCREMENT,
f1 REAL,f2 REAL,f3 REAL,f4 REAL,f5 REAL,
f6 REAL,f7 REAL,f8 REAL,f9 REAL,f10 REAL,
f11 REAL,f12 REAL,f13 REAL,f14 REAL,f15 REAL,
f16 REAL,f17 REAL,f18 REAL,f19 REAL,f20 REAL,
f21 REAL,f22 REAL,f23 REAL,f24 REAL,f25 REAL,
f26 REAL,f27 REAL,f28 REAL,f29 REAL,f30 REAL,
target INTEGER
)
""")

conn.commit()

# -----------------------------
# 최초 지갑 생성
# -----------------------------

wallet_df=pd.read_sql("SELECT * FROM wallet",conn)

if len(wallet_df)==0:

    cur.execute(
    "INSERT INTO wallet VALUES(1,10000000)"
    )

    conn.commit()

# -----------------------------
# wallet load/save
# -----------------------------

def load_wallet():

    df=pd.read_sql("SELECT * FROM wallet WHERE id=1",conn)

    return float(df.iloc[0]["krw"])

def save_wallet(krw):

    cur.execute(
    "UPDATE wallet SET krw=? WHERE id=1",
    (krw,)
    )

    conn.commit()

# -----------------------------
# positions load
# -----------------------------

def load_positions():

    df=pd.read_sql("SELECT * FROM positions",conn)

    pos={}

    for _,r in df.iterrows():

        pos[r.ticker]={
        "qty":r.qty,
        "buy_price":r.buy_price
        }

    return pos

# -----------------------------
# indicators
# -----------------------------

def indicators(df):

    df["ma5"]=df.close.rolling(5).mean()
    df["ma20"]=df.close.rolling(20).mean()

    delta=df.close.diff()

    up=delta.clip(lower=0)
    down=-delta.clip(upper=0)

    rs=up.rolling(14).mean()/down.rolling(14).mean()

    df["rsi"]=100-(100/(1+rs))

    df["momentum"]=df.close.pct_change(3)

    return df

# -----------------------------
# feature
# -----------------------------

def features(df):

    r=df.iloc[-1]

    f=[r.rsi,r.ma5/r.close,r.ma20/r.close,r.momentum]

    while len(f)<30:
        f.append(np.random.random())

    return f[:30]

# -----------------------------
# tradable
# -----------------------------

def tradable():

    url="https://api.upbit.com/v1/market/all"

    res=requests.get(url).json()

    return [x["market"] for x in res if x["market"].startswith("KRW-")]

# -----------------------------
# 거래대금 상위
# -----------------------------

def top100():

    coins=tradable()

    data=[]

    for c in coins:

        try:

            df=pyupbit.get_ohlcv(c,"minute1",count=20)

            val=(df.close*df.volume).sum()

            data.append((c,val))

        except:
            pass

    data=sorted(data,key=lambda x:x[1],reverse=True)

    return [x[0] for x in data[:100]]

# -----------------------------
# learning build
# -----------------------------

def build_learning():

    coins=top100()

    for coin in coins:

        df=pyupbit.get_ohlcv(coin,"minute1",count=200)

        if df is None:
            continue

        df=indicators(df)

        df["target"]=(df.close.shift(-5)>df.close).astype(int)

        df=df.dropna()

        for i in range(len(df)-1):

            f=features(df.iloc[:i+1])

            t=df.iloc[i]["target"]

            cur.execute(
            "INSERT INTO learning VALUES(NULL,"+
            ",".join(["?"]*30)+",?)",
            f+[t]
            )

    conn.commit()

# -----------------------------
# train
# -----------------------------

def train():

    df=pd.read_sql("SELECT * FROM learning",conn)

    if len(df)<3000:
        return None

    X=df.drop(["id","target"],axis=1)

    y=df["target"]

    d=lgb.Dataset(X,label=y)

    params={
    "objective":"binary",
    "learning_rate":0.03,
    "num_leaves":64
    }

    model=lgb.train(params,d,150)

    return model

# -----------------------------
# 자동 학습 (30분)
# -----------------------------

def auto_learning():

    now=int(time.time())

    if now%1800<15:

        build_learning()

# -----------------------------
# trading
# -----------------------------

def trade(model):

    krw=load_wallet()

    positions=load_positions()

    coins=top100()

    for coin in coins:

        if coin in positions:
            continue

        df=pyupbit.get_ohlcv(coin,"minute1",count=120)

        if df is None:
            continue

        df=indicators(df)

        f=features(df)

        prob=model.predict([f])[0]

        if prob<0.6:
            continue

        price=pyupbit.get_current_price(coin)

        invest=krw*0.1

        if invest<10000:
            continue

        qty=invest/price

        krw-=invest

        save_wallet(krw)

        cur.execute(
        "INSERT INTO positions VALUES(?,?,?)",
        (coin,qty,price)
        )

        cur.execute(
        "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
        (datetime.now(),coin,price,qty,"BUY")
        )

        conn.commit()

    # SELL

    positions=load_positions()

    for coin,pos in positions.items():

        price=pyupbit.get_current_price(coin)

        profit=(price-pos["buy_price"])/pos["buy_price"]

        df=pyupbit.get_ohlcv(coin,"minute1",count=120)

        df=indicators(df)

        f=features(df)

        prob=model.predict([f])[0]

        if prob<0.45 or profit>0.08 or profit<-0.03:

            qty=pos["qty"]

            krw=load_wallet()

            krw+=qty*price

            save_wallet(krw)

            cur.execute(
            "DELETE FROM positions WHERE ticker=?",
            (coin,)
            )

            cur.execute(
            "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
            (datetime.now(),coin,price,qty,"SELL")
            )

            conn.commit()

# -----------------------------
# 실행
# -----------------------------

st.title("AI Crypto Trader")

auto_learning()

model=train()

if model:
    trade(model)

# -----------------------------
# dashboard
# -----------------------------

krw=load_wallet()

positions=load_positions()

coin_value=0

rows=[]

for coin,pos in positions.items():

    price=pyupbit.get_current_price(coin)

    value=price*pos["qty"]

    coin_value+=value

    profit=(price-pos["buy_price"])/pos["buy_price"]*100

    rows.append({
    "coin":coin,
    "qty":pos["qty"],
    "buy_price":pos["buy_price"],
    "price":price,
    "profit%":profit
    })

asset=krw+coin_value

c1,c2,c3=st.columns(3)

c1.metric("총 자산",f"{asset:,.0f}")
c2.metric("현금",f"{krw:,.0f}")
c3.metric("코인 평가",f"{coin_value:,.0f}")

st.subheader("보유 코인")

st.dataframe(pd.DataFrame(rows))

hist=pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50",conn)

st.subheader("최근 거래")

st.dataframe(hist)

time.sleep(300)

st.rerun()
