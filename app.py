import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Options Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Core layout and background */
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(circle at 1px 1px, #334155 1px, transparent 0);
        background-size: 20px 20px;
    }
    /* Main title */
    h1 {
        color: #f8fafc;
        text-align: center;
    }
    /* Subtitle */
    .stMarkdown p {
        color: #94a3b8;
        text-align: center;
    }
    /* Metric styling */
    .stMetric {
        background-color: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
    }
    .stMetric .st-ax { /* Target metric label */
         color: #94a3b8;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        background-color: transparent;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
    }

</style>
""", unsafe_allow_html=True)


# --- LIVE DATA FETCHING & MAPPINGS ---
COMPANY_NAMES = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Wipro": "WIPRO.NS",
}

LOT_SIZES = {
    "RELIANCE.NS": 250, "TCS.NS": 150, "HDFCBANK.NS": 550, "INFY.NS": 400,
    "ICICIBANK.NS": 700, "HINDUNILVR.NS": 300, "SBIN.NS": 1500, "BHARTIARTL.NS": 950,
    "ITC.NS": 1600, "LT.NS": 300, "BAJFINANCE.NS": 125, "AXISBANK.NS": 650,
    "KOTAKBANK.NS": 400, "MARUTI.NS": 50, "ASIANPAINT.NS": 200, "WIPRO.NS": 1300,
}

@st.cache_data(ttl=900)  # Cache data for 15 minutes
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            st.error(f"No historical data found for {ticker}. Please check the ticker symbol.")
            return None, None
        current_price = hist['Close'].iloc[-1]
        return current_price, hist['Close'].tolist()
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return None, None

def calculate_historical_volatility(prices):
    if len(prices) < 2: return 0
    log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
    return np.std(log_returns) * np.sqrt(252)

# --- CALLBACKS ---
def update_stock_data():
    """Callback to update stock data when company changes."""
    ticker = COMPANY_NAMES[st.session_state.company_selector]
    current_price, history = get_stock_data(ticker)
    if current_price is not None and history is not None:
        # Update session state using the widget keys for proper state management
        st.session_state.s0_input = current_price
        st.session_state.sigma_input = calculate_historical_volatility(history) * 100
        st.session_state.ticker = ticker
        st.session_state.lot_size_input = LOT_SIZES.get(ticker, 1)

# --- CORE CALCULATION LOGIC (Identical to previous version) ---
def binomial_price(S0, K, T, r, sigma, steps, option_type):
    if T <= 0 or sigma <= 0 or steps <= 0: return {"option_price": 0, "dt": 0, "u": 0, "d": 0, "p": 0}
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not (0 <= p <= 1): return {"option_price": 0, "dt": dt, "u": u, "d": d, "p": p}
    prices = S0 * d**np.arange(steps, -1, -1) * u**np.arange(0, steps + 1, 1)
    option_values = np.maximum(0, prices - K) if option_type == 'Call' else np.maximum(0, K - prices)
    for j in range(steps - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
    return {"option_price": option_values[0], "dt": dt, "u": u, "d": d, "p": p}

def binomial_greeks(params):
    S0, K, T, r, sigma, steps, option_type = params.values()
    dS, dSigma, dT, dR = S0 * 0.01, 0.001, 1/365.0, 0.0001
    base = binomial_price(S0, K, T, r, sigma, steps, option_type)
    p_p_S = binomial_price(S0 + dS, K, T, r, sigma, steps, option_type)["option_price"]
    p_m_S = binomial_price(S0 - dS, K, T, r, sigma, steps, option_type)["option_price"]
    p_p_sig = binomial_price(S0, K, T, r, sigma + dSigma, steps, option_type)["option_price"]
    p_m_T = binomial_price(S0, K, T - dT if T > dT else 0, r, sigma, steps, option_type)["option_price"]
    p_p_r = binomial_price(S0, K, T, r + dR, sigma, steps, option_type)["option_price"]
    delta = (p_p_S - p_m_S) / (2 * dS) if dS else 0
    gamma = (p_p_S - 2 * base["option_price"] + p_m_S) / (dS**2) if dS else 0
    vega = (p_p_sig - base["option_price"]) / (dSigma * 100) if dSigma else 0
    theta = p_m_T - base["option_price"]
    rho = (p_p_r - base["option_price"]) / (dR * 100) if dR else 0
    details = {"dS": dS, "base_price": base['option_price'], "price_plus_S": p_p_S, "price_minus_S": p_m_S, "dSigma": dSigma, "price_plus_sigma": p_p_sig, "price_minus_T": p_m_T, "dR": dR, "price_plus_r": p_p_r}
    return {"greeks": {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}, "details": details, **base}

def black_scholes_price(S0, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0: return {"option_price": 0, "d1": 0, "d2": 0}
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'Call' else (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    return {"option_price": price, "d1": d1, "d2": d2}

def black_scholes_greeks(params):
    S0, K, T, r, sigma, _, option_type = params.values()
    if T <= 0 or sigma <= 0: return {"greeks": {k: 0 for k in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}, "d1": 0, "d2": 0, "option_price": 0}
    res = black_scholes_price(S0, K, T, r, sigma, option_type)
    d1, pdf_d1 = res['d1'], norm.pdf(res['d1'])
    delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S0 * sigma * np.sqrt(T)) if S0 > 0 and sigma > 0 else 0
    vega = S0 * pdf_d1 * np.sqrt(T) / 100
    theta_p1 = -(S0 * pdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == 'Call':
        theta_p2 = r * K * np.exp(-r * T) * norm.cdf(res['d2'])
        theta = (theta_p1 - theta_p2) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(res['d2']) / 100
    else:
        theta_p2 = r * K * np.exp(-r * T) * norm.cdf(-res['d2'])
        theta = (theta_p1 + theta_p2) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-res['d2']) / 100
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
st.markdown("<p>Live Multi-Model Option Pricing & Greeks Analysis</p>", unsafe_allow_html=True)

# Initialize session state
if 'ticker' not in st.session_state:
    st.session_state.company_selector = "Reliance Industries"
    update_stock_data() # Initial data fetch

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    tab1, tab2 = st.tabs(["Market & Option", "Model & Analysis"])
    with tab1:
        with st.container(border=True):
            st.subheader("📈 Market Parameters")
            st.selectbox("Company", options=list(COMPANY_NAMES.keys()), key='company_selector', on_change=update_stock_data)
            # These widgets now correctly get their values from session state via their keys
            S0 = st.number_input("Stock Price (S₀)", format="%.2f", key="s0_input")
            sigma_pct = st.number_input("Volatility (σ %)", format="%.2f", key="sigma_input")
            r_pct = st.number_input("Risk-Free Rate (r %)", value=7.0, format="%.1f")
        with st.container(border=True):
            st.subheader("⚙️ Option Parameters")
            K = st.number_input("Strike Price (K)", value=3000.00, format="%.2f")
            days_to_expiry = st.number_input("Days to Expiry", value=30, min_value=1)
            lot_size = st.number_input("Lot Size", min_value=1, key='lot_size_input')
            option_type = st.radio("Option Type", ('Call', 'Put'), horizontal=True)
    with tab2:
        with st.container(border=True):
            st.subheader("🧮 Model Parameters")
            model = st.radio("Calculation Model", ('Binomial Tree', 'Black-Scholes'), horizontal=True)
            steps = st.slider("Binomial Steps (N)", min_value=1, max_value=200, value=50) if model == 'Binomial Tree' else 0
        if model == 'Black-Scholes':
            with st.container(border=True):
                st.subheader("🔍 Implied Volatility")
                market_price = st.number_input("Enter Market Price (₹)", min_value=0.01, format="%.2f")
                if st.button("Calculate IV"):
                    T = days_to_expiry / 365.0; r = r_pct / 100.0
                    if market_price > 0:
                        iv = calculate_implied_volatility(market_price, S0, K, T, r, option_type)
                        st.metric("Calculated Implied Volatility", f"{iv*100:.2f}%")
                        st.info("Volatility input has been updated.")
                        st.session_state.sigma_input = iv * 100
                        st.rerun()

# --- CALCULATIONS ---
T, r, sigma = days_to_expiry / 365.0, r_pct / 100.0, sigma_pct / 100.0
params = {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "steps": steps, "option_type": option_type}
results = binomial_greeks(params) if model == 'Binomial Tree' else black_scholes_greeks(params)

# --- RESULTS DISPLAY ---
with right_col:
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.metric(label=f"Calculated {option_type} Price", value=f"₹{results['option_price']:.4f}")
    with res_c2:
        total_premium = results['option_price'] * lot_size
        st.metric(label="Total Premium (Price × Lot Size)", value=f"₹{total_premium:,.2f}")

    if model == 'Binomial Tree':
        with st.container(border=True):
            st.subheader("Binomial Model Internals")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Time Step (dt)", f"{results.get('dt', 0):.5f}")
            c2.metric("Up Factor (u)", f"{results.get('u', 0):.5f}")
            c3.metric("Down Factor (d)", f"{results.get('d', 0):.5f}")
            c4.metric("Probability (p)", f"{results.get('p', 0):.5f}")
            if not (0 <= results.get('p', 1) <= 1): st.warning("Arbitrage Alert: p is outside [0, 1].")

    with st.container(border=True):
        st.subheader("Option Greeks")
        greeks = results["greeks"]
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        gc1.metric("Delta (Δ)", f"{greeks.get('Delta', 0):.4f}")
        gc2.metric("Gamma (Γ)", f"{greeks.get('Gamma', 0):.4f}")
        gc3.metric("Vega", f"{greeks.get('Vega', 0):.4f}")
        gc4.metric("Theta (Θ)", f"{greeks.get('Theta', 0):.4f}")
        gc5.metric("Rho (ρ)", f"{greeks.get('Rho', 0):.4f}")

    with st.expander("Show Calculation Details"):
        with st.container(border=True):
            if model == 'Binomial Tree':
                d = results.get('details', {})
                st.markdown(f"**Delta (Δ):** `({d.get('price_plus_S', 0):.4f} - {d.get('price_minus_S', 0):.4f}) / (2 * {d.get('dS', 0):.2f})`")
                st.markdown(f"**Gamma (Γ):** `({d.get('price_plus_S', 0):.4f} - 2*{d.get('base_price', 0):.4f} + {d.get('price_minus_S', 0):.4f}) / {d.get('dS', 0):.2f}²`")
                st.markdown(f"**Vega:** `({d.get('price_plus_sigma', 0):.4f} - {d.get('base_price', 0):.4f}) / ({d.get('dSigma', 0)} * 100)`")
                st.markdown(f"**Theta (Θ):** `{d.get('price_minus_T', 0):.4f} - {d.get('base_price', 0):.4f}`")
                st.markdown(f"**Rho (ρ):** `({d.get('price_plus_r', 0):.4f} - {d.get('base_price', 0):.4f}) / ({d.get('dR', 0)} * 100)`")
            else:
                st.markdown(f"**d1:** `{results.get('d1', 0):.5f}`")
                st.markdown(f"**d2:** `{results.get('d2', 0):.5f}`")
                st.markdown(f"**N(d1):** `{norm.cdf(results.get('d1', 0)):.5f}`")
                st.markdown(f"**N(d2):** `{norm.cdf(results.get('d2', 0)):.5f}`")

    with st.container(border=True):
        st.subheader("Profit/Loss Payoff Diagram")
        start_price, end_price = K * 0.8, K * 1.2
        stock_prices = np.linspace(start_price, end_price, 50)
        payoff = ((np.maximum(0, stock_prices - K) if option_type == 'Call' else np.maximum(0, K - stock_prices)) - results['option_price']) * lot_size
        fig = go.Figure(data=go.Scatter(x=stock_prices, y=payoff, mode='lines', line=dict(color='#38bdf8')))
        fig.update_layout(xaxis_title="Stock Price at Expiration", yaxis_title="Profit / Loss for Lot", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'), height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

