# -----------------------------
# 대시보드 표시 (매수/매도 금액, 이윤, 이윤%) 추가
# -----------------------------
krw = load_wallet()
positions = load_positions()
coin_value = 0
rows=[]
for coin,pos in positions.items():
    price = pyupbit.get_current_price(coin)
    value = price * pos["qty"]
    coin_value += value
    profit = (price - pos["buy_price"]) / pos["buy_price"] * 100
    rows.append({
        "coin": coin,
        "qty": pos["qty"],
        "buy_price": pos["buy_price"],
        "price": price,
        "profit%": profit
    })

asset = krw + coin_value
c1,c2,c3 = st.columns(3)
c1.metric("총 자산", f"{asset:,.0f}")
c2.metric("현금", f"{krw:,.0f}")
c3.metric("코인 평가", f"{coin_value:,.0f}")

st.subheader("보유 코인")
st.dataframe(pd.DataFrame(rows))

# 최근 거래 기록 불러오기
hist = pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT 50", conn)

# 매수/매도 금액, 이윤, 이윤% 계산
enhanced_hist = []
for _, row in hist.iterrows():
    if row['side'] == "BUY":
        enhanced_hist.append({
            "시간": row['time'],
            "코인": row['ticker'],
            "사/매도": row['side'],
            "수량": row['qty'],
            "가격": row['price'],
            "매수금액": row['qty']*row['price'],
            "매도금액": None,
            "이윤": None,
            "이윤%": None
        })
    elif row['side'] == "SELL":
        # 매도 시점에서 매수 기록 불러오기
        buy_row = cur.execute("SELECT price, qty FROM trades WHERE ticker=? AND side='BUY' ORDER BY id DESC LIMIT 1", (row['ticker'],)).fetchone()
        if buy_row:
            buy_price, buy_qty = buy_row
            sell_price, sell_qty = row['price'], row['qty']
            profit = (sell_price - buy_price) * sell_qty
            profit_percent = (sell_price - buy_price) / buy_price * 100
        else:
            profit = None
            profit_percent = None
        enhanced_hist.append({
            "시간": row['time'],
            "코인": row['ticker'],
            "사/매도": row['side'],
            "수량": row['qty'],
            "가격": row['price'],
            "매수금액": buy_price*sell_qty if buy_row else None,
            "매도금액": sell_price*sell_qty,
            "이윤": profit,
            "이윤%": profit_percent
        })

st.subheader("최근 거래 (매수/매도 금액 + 이윤)")
st.dataframe(pd.DataFrame(enhanced_hist))
