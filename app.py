import streamlit as st
import pyupbit
import pandas as pd
import numpy as np
import sqlite3
import requests
import time
from datetime import datetime
import lightgbm as lgb

st.set_page_config(page_title="Adaptive AI Crypto Trader",layout="wide")

DB="ai_trader.db"

conn=sqlite3.connect(DB,check_same_thread=False)
cur=conn.cursor()

# ---------------------------
# DB
# ---------------------------

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

cur.execute("""
CREATE TABLE IF NOT EXISTS strategy(
id INTEGER PRIMARY KEY,
buy_threshold REAL,
sell_threshold REAL,
max_risk REAL
)
""")

conn.commit()

# 초기 전략

if len(pd.read_sql("SELECT * FROM strategy",conn))==0:

    cur.execute(
    "INSERT INTO strategy VALUES(1,0.6,0.45,0.25)"
    )

    conn.commit()

# ---------------------------
# wallet
# ---------------------------

if "wallet" not in st.session_state:

    st.session_state.wallet={
        "krw":10000000.0,
        "positions":{}
    }

wallet=st.session_state.wallet

# ---------------------------
# indicators
# ---------------------------

def indicators(df):

    df["rsi"]=100-(100/(1+(df.close.diff().clip(lower=0).rolling(14).mean()/(-df.close.diff().clip(upper=0).rolling(14).mean()))))

    df["ema5"]=df.close.ewm(span=5).mean()
    df["ema20"]=df.close.ewm(span=20).mean()

    df["macd"]=df.ema5-df.ema20
    df["macd_signal"]=df.macd.ewm(span=9).mean()

    df["roc"]=df.close.pct_change(5)

    df["vol_mean"]=df.volume.rolling(20).mean()
    df["vol_ratio"]=df.volume/df.vol_mean

    df["momentum"]=df.close.pct_change(3)

    df["boll_mid"]=df.close.rolling(20).mean()
    df["boll_std"]=df.close.rolling(20).std()

    df["boll_gap"]=(df.close-df.boll_mid)/df.boll_std

    return df

# ---------------------------
# feature
# ---------------------------

def features(df):

    r=df.iloc[-1]

    f=[
        r.rsi,
        r.ema5/r.close,
        r.ema20/r.close,
        r.macd,
        r.macd_signal,
        r.roc,
        r.vol_ratio,
        r.momentum,
        r.boll_gap
    ]

    while len(f)<30:
        f.append(np.random.random())

    return f[:30]

# ---------------------------
# tradable coins
# ---------------------------

def tradable():

    url="https://api.upbit.com/v1/market/all"

    res=requests.get(url).json()

    return [x["market"] for x in res if x["market"].startswith("KRW-")]

# ---------------------------
# 거래대금 상위
# ---------------------------

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

# ---------------------------
# learning build
# ---------------------------

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

# ---------------------------
# train
# ---------------------------

def train():

    df=pd.read_sql("SELECT * FROM learning",conn)

    if len(df)<3000:
        return None

    X=df.drop(["id","target"],axis=1)
    y=df["target"]

    d=lgb.Dataset(X,label=y)

    params={
    "objective":"binary",
    "metric":"auc",
    "learning_rate":0.03,
    "num_leaves":64
    }

    model=lgb.train(params,d,150)

    return model

# ---------------------------
# 전략 자동 수정
# ---------------------------

def adapt_strategy():

    hist=pd.read_sql("SELECT * FROM trades",conn)

    if len(hist)<50:
        return

    hist["value"]=hist.price*hist.qty

    buy=hist[hist.side=="BUY"]["value"].sum()
    sell=hist[hist.side=="SELL"]["value"].sum()

    profit=sell-buy

    strat=pd.read_sql("SELECT * FROM strategy",conn).iloc[0]

    buy_th=strat.buy_threshold
    sell_th=strat.sell_threshold
    risk=strat.max_risk

    if profit<0:

        buy_th=min(0.7,buy_th+0.02)
        risk=max(0.05,risk-0.02)

    else:

        buy_th=max(0.55,buy_th-0.01)
        risk=min(0.30,risk+0.01)

    cur.execute(
    "UPDATE strategy SET buy_threshold=?,sell_threshold=?,max_risk=? WHERE id=1",
    (buy_th,sell_th,risk)
    )

    conn.commit()

# ---------------------------
# Kelly
# ---------------------------

def position_size(prob,risk):

    edge=(prob*2)-1

    if edge<=0:
        return 0

    k=edge*risk

    return min(k,risk)

# ---------------------------
# trading
# ---------------------------

def trade(model):

    strat=pd.read_sql("SELECT * FROM strategy",conn).iloc[0]

    buy_th=strat.buy_threshold
    sell_th=strat.sell_threshold
    risk=strat.max_risk

    coins=top100()

    for coin in coins:

        df=pyupbit.get_ohlcv(coin,"minute1",count=120)

        if df is None:
            continue

        df=indicators(df)

        f=features(df)

        prob=model.predict([f])[0]

        size=position_size(prob,risk)

        if prob>buy_th and size>0:

            price=pyupbit.get_current_price(coin)

            invest=wallet["krw"]*size

            if invest<10000:
                continue

            qty=invest/price

            wallet["krw"]-=invest

            wallet["positions"][coin]={"qty":qty,"buy_price":price}

            cur.execute(
            "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
            (datetime.now(),coin,price,qty,"BUY")
            )

            conn.commit()

    # sell

    for coin in list(wallet["positions"].keys()):

        pos=wallet["positions"][coin]

        price=pyupbit.get_current_price(coin)

        profit=(price-pos["buy_price"])/pos["buy_price"]

        df=pyupbit.get_ohlcv(coin,"minute1",count=120)

        df=indicators(df)

        f=features(df)

        prob=model.predict([f])[0]

        if prob<sell_th or profit>0.08 or profit<-0.03:

            qty=pos["qty"]

            wallet["krw"]+=qty*price

            del wallet["positions"][coin]

            cur.execute(
            "INSERT INTO trades VALUES(NULL,?,?,?,?,?)",
            (datetime.now(),coin,price,qty,"SELL")
            )

            conn.commit()

# ---------------------------
# 자동 학습 (30분)
# ---------------------------

def auto_learning():

    now=int(time.time())

    if now%1800<15:

        build_learning()

        adapt_strategy()

# ---------------------------
# dashboard
# ---------------------------

st.title("Adaptive AI Crypto Trader")

auto_learning()

model=train()

if model:
    trade(model)

# asset

coin_value=0

rows=[]

for coin,pos in wallet["positions"].items():

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

asset=wallet["krw"]+coin_value

c1,c2,c3=st.columns(3)

c1.metric("총 자산",f"{asset:,.0f}")
c2.metric("현금",f"{wallet['krw']:,.0f}")
c3.metric("코인 평가",f"{coin_value:,.0f}")

st.dataframe(pd.DataFrame(rows))

hist=pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50",conn)

st.subheader("최근 거래")

st.dataframe(hist)

time.sleep(300)

st.rerun()
