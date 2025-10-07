import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Options Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(circle at 1px 1px, #334155 1px, transparent 0);
        background-size: 20px 20px;
    }
    .stMetric {
        background-color: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .stMetric .st-ax { /* Target metric label */
         color: #94a3b8;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc;
    }
    .card {
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA SIMULATION ---
@st.cache_data
def generate_price_history(start_price, vol, days):
    prices = [start_price]
    for _ in range(1, days):
        daily_return = (np.random.randn()) * vol / np.sqrt(252)
        prices.append(prices[-1] * (1 + daily_return))
    return prices

MOCK_STOCK_DATA = {
    "RELIANCE.NS": {"price": 2955.75, "history": generate_price_history(2955.75, 0.228, 252)},
    "INFY.NS": {"price": 1510.30, "history": generate_price_history(1510.30, 0.285, 252)},
    "TCS.NS": {"price": 3855.10, "history": generate_price_history(3855.10, 0.212, 252)},
    "HDFCBANK.NS": {"price": 1530.90, "history": generate_price_history(1530.90, 0.251, 252)},
}

def calculate_historical_volatility(prices):
    if len(prices) < 2: return 0
    log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    return np.std(log_returns) * np.sqrt(252)

# --- CORE CALCULATION LOGIC ---

# --- Binomial Model ---
def binomial_price(S0, K, T, r, sigma, steps, option_type):
    if T <= 0 or sigma <= 0 or steps <= 0:
        return {"option_price": 0, "dt": 0, "u": 0, "d": 0, "p": 0}
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not (0 <= p <= 1): # Arbitrage check
        return {"option_price": 0, "dt": dt, "u": u, "d": d, "p": p}

    prices = S0 * d**np.arange(steps, -1, -1) * u**np.arange(0, steps + 1, 1)
    
    if option_type == 'Call':
        option_values = np.maximum(0, prices - K)
    else:
        option_values = np.maximum(0, K - prices)

    for j in range(steps - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])

    return {"option_price": option_values[0], "dt": dt, "u": u, "d": d, "p": p}

def binomial_greeks(params):
    S0, K, T, r, sigma, steps, option_type = params.values()
    dS = S0 * 0.01
    dSigma = 0.001
    dT = 1 / 365.0
    dR = 0.0001
    
    base = binomial_price(S0, K, T, r, sigma, steps, option_type)
    price_plus_S = binomial_price(S0 + dS, K, T, r, sigma, steps, option_type)["option_price"]
    price_minus_S = binomial_price(S0 - dS, K, T, r, sigma, steps, option_type)["option_price"]
    price_plus_sigma = binomial_price(S0, K, T, r, sigma + dSigma, steps, option_type)["option_price"]
    price_minus_T = binomial_price(S0, K, T - dT if T > dT else 0, r, sigma, steps, option_type)["option_price"]
    price_plus_r = binomial_price(S0, K, T, r + dR, sigma, steps, option_type)["option_price"]

    delta = (price_plus_S - price_minus_S) / (2 * dS) if dS != 0 else 0
    gamma = (price_plus_S - 2 * base["option_price"] + price_minus_S) / (dS**2) if dS != 0 else 0
    vega = (price_plus_sigma - base["option_price"]) / (dSigma * 100) if dSigma != 0 else 0
    theta = (price_minus_T - base["option_price"])
    rho = (price_plus_r - base["option_price"]) / (dR * 100) if dR != 0 else 0
    
    details = {
        "dS": dS, "base_price": base['option_price'], "price_plus_S": price_plus_S,
        "price_minus_S": price_minus_S, "dSigma": dSigma, "price_plus_sigma": price_plus_sigma,
        "price_minus_T": price_minus_T, "dR": dR, "price_plus_r": price_plus_r
    }
    
    return {"greeks": {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}, "details": details, **base}

# --- Black-Scholes Model ---
def black_scholes_price(S0, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0: return {"option_price": 0, "d1": 0, "d2": 0}
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        price = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    return {"option_price": price, "d1": d1, "d2": d2}

def black_scholes_greeks(params):
    S0, K, T, r, sigma, _, option_type = params.values()
    if T <= 0 or sigma <= 0:
        return {"greeks": {k: 0 for k in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}, "d1": 0, "d2": 0, "option_price": 0}

    res = black_scholes_price(S0, K, T, r, sigma, option_type)
    d1, d2 = res['d1'], res['d2']
    pdf_d1 = norm.pdf(d1)
    
    delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * pdf_d1 * np.sqrt(T) / 100
    theta_p1 = -(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == 'Call':
        theta_p2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (theta_p1 - theta_p2) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        theta_p2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (theta_p1 + theta_p2) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
    return {"greeks": {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}, **res}

def calculate_implied_volatility(market_price, S0, K, T, r, option_type):
    low_vol, high_vol = 0.0001, 5.0
    for _ in range(100):
        mid_vol = (low_vol + high_vol) / 2
        if mid_vol < 1e-5: return 0
        price = black_scholes_price(S0, K, T, r, mid_vol, option_type)["option_price"]
        if price > market_price: high_vol = mid_vol
        else: low_vol = mid_vol
    return (low_vol + high_vol) / 2

# --- UI LAYOUT ---
st.title("Advanced Options Dashboard")
st.markdown("<p class='text-lg text-slate-400'>Multi-Model Option Pricing & Greeks Analysis</p>", unsafe_allow_html=True)

# Initialize session state for persistence
if 'ticker' not in st.session_state:
    st.session_state.ticker = "RELIANCE.NS"
    st.session_state.S0 = 2950.00
    st.session_state.sigma = 22.5

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1. Market Parameters")
        
        c1, c2 = st.columns([3,1])
        with c1:
            st.session_state.ticker = st.text_input("Stock Ticker", st.session_state.ticker, label_visibility="collapsed").upper()
        with c2:
            if st.button("Fetch"):
                if st.session_state.ticker in MOCK_STOCK_DATA:
                    data = MOCK_STOCK_DATA[st.session_state.ticker]
                    st.session_state.S0 = data['price']
                    st.session_state.sigma = calculate_historical_volatility(data['history']) * 100
                    st.toast(f"Data fetched for {st.session_state.ticker}!", icon="✅")
                else:
                    st.error("Ticker not found.")
        
        S0 = st.number_input("Stock Price (S₀)", value=st.session_state.S0, format="%.2f")
        sigma_pct = st.number_input("Volatility (σ %)", value=st.session_state.sigma, format="%.2f")
        r_pct = st.number_input("Risk-Free Rate (r %)", value=7.0, format="%.1f")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2. Option Parameters")
        K = st.number_input("Strike Price (K)", value=3000.00, format="%.2f")
        days_to_expiry = st.number_input("Days to Expiry", value=30, min_value=1)
        option_type = st.radio("Option Type", ('Call', 'Put'), horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("3. Model Parameters")
        model = st.radio("Calculation Model", ('Binomial Tree', 'Black-Scholes'), horizontal=True)
        if model == 'Binomial Tree':
            steps = st.slider("Binomial Steps (N)", min_value=1, max_value=200, value=50)
        else:
            steps = 0 # Not used for Black-Scholes
        st.markdown('</div>', unsafe_allow_html=True)
    
    if model == 'Black-Scholes':
         with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Implied Volatility Calculator")
            market_price = st.number_input("Enter Market Price (₹)", min_value=0.01, format="%.2f")
            if st.button("Calculate IV"):
                T = days_to_expiry / 365.0
                r = r_pct / 100.0
                iv = calculate_implied_volatility(market_price, S0, K, T, r, option_type)
                st.metric("Calculated Implied Volatility", f"{iv*100:.2f}%")
                st.info("Volatility input has been updated with this value.")
                st.session_state.sigma = iv * 100
                st.rerun() # Rerun to update the sigma input box
            st.markdown('</div>', unsafe_allow_html=True)


# --- CALCULATIONS ---
T = days_to_expiry / 365.0
r = r_pct / 100.0
sigma = sigma_pct / 100.0
params = {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "steps": steps, "option_type": option_type}

if model == 'Binomial Tree':
    results = binomial_greeks(params)
else:
    results = black_scholes_greeks(params)

# --- RESULTS DISPLAY ---
with right_col:
    st.metric(label=f"Calculated {option_type} Price ({model})", value=f"₹{results['option_price']:.4f}")

    if model == 'Binomial Tree':
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Binomial Model Internals")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Time Step (dt)", f"{results['dt']:.5f}")
            c2.metric("Up Factor (u)", f"{results['u']:.5f}")
            c3.metric("Down Factor (d)", f"{results['d']:.5f}")
            c4.metric("Probability (p)", f"{results['p']:.5f}")
            if not (0 <= results['p'] <= 1):
                st.warning("Arbitrage opportunity detected (p is not between 0 and 1). Results are unreliable.")
            st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Option Greeks")
        greeks = results["greeks"]
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        gc1.metric("Delta (Δ)", f"{greeks['Delta']:.4f}")
        gc2.metric("Gamma (Γ)", f"{greeks['Gamma']:.4f}")
        gc3.metric("Vega", f"{greeks['Vega']:.4f}")
        gc4.metric("Theta (Θ)", f"{greeks['Theta']:.4f}")
        gc5.metric("Rho (ρ)", f"{greeks['Rho']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Calculation Details")
        if model == 'Binomial Tree':
            d = results['details']
            st.markdown(f"**Delta (Δ):** `({d['price_plus_S']:.4f} - {d['price_minus_S']:.4f}) / (2 * {d['dS']:.2f})`")
            st.markdown(f"**Gamma (Γ):** `({d['price_plus_S']:.4f} - 2*{d['base_price']:.4f} + {d['price_minus_S']:.4f}) / {d['dS']:.2f}²`")
            st.markdown(f"**Vega:** `({d['price_plus_sigma']:.4f} - {d['base_price']:.4f}) / ({d['dSigma']} * 100)`")
            st.markdown(f"**Theta (Θ):** `{d['price_minus_T']:.4f} - {d['base_price']:.4f}`")
            st.markdown(f"**Rho (ρ):** `({d['price_plus_r']:.4f} - {d['base_price']:.4f}) / ({d['dR']} * 100)`")
        else: # Black-Scholes
            st.markdown(f"**d1:** `{results['d1']:.5f}`")
            st.markdown(f"**d2:** `{results['d2']:.5f}`")
            st.markdown(f"**N(d1):** `{norm.cdf(results['d1']):.5f}`")
            st.markdown(f"**N(d2):** `{norm.cdf(results['d2']):.5f}`")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Profit/Loss Payoff Diagram")
        start_price = K * 0.8
        end_price = K * 1.2
        stock_prices = np.linspace(start_price, end_price, 50)
        if option_type == 'Call':
            payoff = np.maximum(0, stock_prices - K) - results['option_price']
        else:
            payoff = np.maximum(0, K - stock_prices) - results['option_price']
        
        fig = go.Figure(data=go.Scatter(x=stock_prices, y=payoff, mode='lines', line=dict(color='#38bdf8')))
        fig.update_layout(
            title="Profit/Loss at Expiration",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit / Loss",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            xaxis=dict(gridcolor='#334155'),
            yaxis=dict(gridcolor='#334155'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
