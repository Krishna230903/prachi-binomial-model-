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
    "ACC": "ACC.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Balkrishna Industries": "BALKRISIND.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Berger Paints": "BERGEPAINT.NS",
    "Bharat Forge": "BHARATFORG.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Biocon": "BIOCON.NS",
    "Bosch": "BOSCHLTD.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Canara Bank": "CANBK.NS",
    "Cholamandalam Investment": "CHOLAFIN.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Coforge": "COFORGE.NS",
    "Colgate-Palmolive": "COLPAL.NS",
    "Container Corporation": "CONCOR.NS",
    "Coromandel International": "COROMANDEL.NS",
    "Cummins India": "CUMMINSIND.NS",
    "DLF": "DLF.NS",
    "Dabur India": "DABUR.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Escorts Kubota": "ESCORTS.NS",
    "Federal Bank": "FEDERALBNK.NS",
    "GAIL (India)": "GAIL.NS",
    "Glenmark Pharmaceuticals": "GLENMARK.NS",
    "Godrej Consumer Products": "GODREJCP.NS",
    "Godrej Properties": "GODREJPROP.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Havells India": "HAVELLS.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Hindustan Aeronautics": "HAL.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ICICI Lombard": "ICICIGI.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "Indian Oil Corporation": "IOC.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "InterGlobe Aviation (IndiGo)": "INDIGO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Jindal Steel & Power": "JINDALSTEL.NS",
    "Jubilant FoodWorks": "JUBLFOOD.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "L&T Technology Services": "LTTS.NS",
    "Larsen & Toubro": "LT.NS",
    "Lupin": "LUPIN.NS",
    "M&M Financial Services": "M&MFIN.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Marico": "MARICO.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mphasis": "MPHASIS.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Nestlé India": "NESTLEIND.NS",
    "Oracle Financial Services": "OFSS.NS",
    "PI Industries": "PIIND.NS",
    "Page Industries": "PAGEIND.NS",
    "Petronet LNG": "PETRONET.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Power Finance Corporation": "PFC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "RBL Bank": "RBLBANK.NS",
    "REC": "RECLTD.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI Cards": "SBICARD.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "SRF": "SRF.NS",
    "Samvardhana Motherson": "MOTHERSON.NS",
    "Siemens": "SIEMENS.NS",
    "State Bank of India": "SBIN.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Tata Chemicals": "TATACHEM.NS",
    "Tata Communications": "TATACOMM.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Power": "TATAPOWER.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan Company": "TITAN.NS",
    "Torrent Pharmaceuticals": "TORNTPHARM.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "United Spirits": "MCDOWELL-N.NS",
    "Vedanta": "VEDL.NS",
    "Wipro": "WIPRO.NS",
    "Zee Entertainment": "ZEEL.NS",
}

LOT_SIZES = {
    "ACC.NS": 250, "ADANIENT.NS": 250, "ADANIPORTS.NS": 800, "AMBUJACEM.NS": 1800,
    "APOLLOHOSP.NS": 125, "ASIANPAINT.NS": 200, "AXISBANK.NS": 650, "BAJAJ-AUTO.NS": 125,
    "BAJFINANCE.NS": 125, "BAJAJFINSV.NS": 500, "BALKRISIND.NS": 200, "BANDHANBNK.NS": 1800,
    "BANKBARODA.NS": 2700, "BERGEPAINT.NS": 1100, "BHARATFORG.NS": 600, "BHARTIARTL.NS": 950,
    "BIOCON.NS": 2300, "BOSCHLTD.NS": 25, "BRITANNIA.NS": 200, "CANBK.NS": 700,
    "CHOLAFIN.NS": 500, "CIPLA.NS": 500, "COALINDIA.NS": 1050, "COFORGE.NS": 100,
    "COLPAL.NS": 350, "CONCOR.NS": 1000, "COROMANDEL.NS": 550, "CUMMINSIND.NS": 400,
    "DLF.NS": 1350, "DABUR.NS": 1250, "DIVISLAB.NS": 200, "DRREDDY.NS": 125,
    "EICHERMOT.NS": 175, "ESCORTS.NS": 400, "FEDERALBNK.NS": 5000, "GAIL.NS": 4800,
    "GLENMARK.NS": 800, "GODREJCP.NS": 800, "GODREJPROP.NS": 475, "GRASIM.NS": 225,
    "HCLTECH.NS": 700, "HDFCBANK.NS": 550, "HDFCLIFE.NS": 1100, "HAVELLS.NS": 500,
    "HEROMOTOCO.NS": 150, "HINDALCO.NS": 1300, "HAL.NS": 300, "HINDUNILVR.NS": 300,
    "ICICIBANK.NS": 700, "ICICIGI.NS": 400, "ICICIPRULI.NS": 1300, "IDFCFIRSTB.NS": 9000,
    "IOC.NS": 3300, "INDUSINDBK.NS": 350, "INFY.NS": 400, "INDIGO.NS": 150,
    "JSWSTEEL.NS": 1350, "JINDALSTEL.NS": 1000, "JUBLFOOD.NS": 1250, "KOTAKBANK.NS": 400,
    "LTTS.NS": 150, "LT.NS": 300, "LUPIN.NS": 550, "M&MFIN.NS": 1600, "M&M.NS": 700,
    "MARICO.NS": 1000, "MARUTI.NS": 50, "MPHASIS.NS": 225, "MUTHOOTFIN.NS": 750,
    "NESTLEIND.NS": 40, "OFSS.NS": 75, "PIIND.NS": 225, "PAGEIND.NS": 20,
    "PETRONET.NS": 2500, "PIDILITIND.NS": 250, "PFC.NS": 2100, "POWERGRID.NS": 3600,
    "RBLBANK.NS": 2600, "RECLTD.NS": 1500, "RELIANCE.NS": 250, "SBICARD.NS": 1050,
    "SBILIFE.NS": 750, "SRF.NS": 300, "MOTHERSON.NS": 7000, "SIEMENS.NS": 150,
    "SBIN.NS": 1500, "SUNPHARMA.NS": 700, "TATACHEM.NS": 700, "TATACOMM.NS": 200,
    "TCS.NS": 150, "TATACONSUM.NS": 900, "TATAMOTORS.NS": 1425, "TATAPOWER.NS": 2700,
    "TATASTEEL.NS": 850, "TECHM.NS": 600, "TITAN.NS": 175, "TORNTPHARM.NS": 250,
    "ULTRACEMCO.NS": 100, "MCDOWELL-N.NS": 425, "VEDL.NS": 1300, "WIPRO.NS": 1300,
    "ZEEL.NS": 3000
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
        st.session_state.s0_input = float(current_price)
        st.session_state.sigma_input = float(calculate_historical_volatility(history) * 100)
        st.session_state.ticker = ticker
        st.session_state.lot_size_input = LOT_SIZES.get(ticker, 1)

# --- CORE CALCULATION LOGIC (Identical to previous version) ---
def binomial_price(S0, K, T, r, sigma, steps, option_type):
    if T <= 0 or sigma <= 0 or steps <= 0 or S0 <= 0: return {"option_price": 0, "dt": 0, "u": 0, "d": 0, "p": 0}
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
    if T <= 0 or sigma <= 0 or S0 <=0: return {"option_price": 0, "d1": 0, "d2": 0}
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'Call' else (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    return {"option_price": price, "d1": d1, "d2": d2}

def black_scholes_greeks(params):
    S0, K, T, r, sigma, _, option_type = params.values()
    if T <= 0 or sigma <= 0 or S0 <=0: return {"greeks": {k: 0 for k in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}, "d1": 0, "d2": 0, "option_price": 0}
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

# Define UI widgets in the left column
with left_col:
    tab1, tab2 = st.tabs(["Market & Option", "Model & Analysis"])
    with tab1:
        with st.container(border=True):
            st.subheader("📈 Market Parameters")
            st.selectbox("Company", options=list(COMPANY_NAMES.keys()), key='company_selector', on_change=update_stock_data)
            st.number_input("Stock Price (S₀)", format="%.2f", key="s0_input")
            st.number_input("Volatility (σ %)", format="%.2f", key="sigma_input")
            r_pct_val = st.number_input("Risk-Free Rate (r %)", value=7.0, format="%.1f")
        with st.container(border=True):
            st.subheader("⚙️ Option Parameters")
            K_val = st.number_input("Strike Price (K)", value=st.session_state.get('s0_input', 3000.0), format="%.2f")
            days_to_expiry_val = st.number_input("Days to Expiry", value=30, min_value=1)
            st.number_input("Lot Size", min_value=1, key='lot_size_input')
            option_type_val = st.radio("Option Type", ('Call', 'Put'), horizontal=True)
    with tab2:
        with st.container(border=True):
            st.subheader("🧮 Model Parameters")
            model_val = st.radio("Calculation Model", ('Binomial Tree', 'Black-Scholes'), horizontal=True)
            steps_val = st.slider("Binomial Steps (N)", min_value=1, max_value=200, value=50) if model_val == 'Binomial Tree' else 0
        if model_val == 'Black-Scholes':
            with st.container(border=True):
                st.subheader("🔍 Implied Volatility")
                market_price_val = st.number_input("Enter Market Price (₹)", min_value=0.01, format="%.2f")
                if st.button("Calculate IV"):
                    T_iv = days_to_expiry_val / 365.0
                    r_iv = r_pct_val / 100.0
                    S0_iv = st.session_state.s0_input
                    if market_price_val > 0:
                        iv = calculate_implied_volatility(market_price_val, S0_iv, K_val, T_iv, r_iv, option_type_val)
                        st.metric("Calculated Implied Volatility", f"{iv*100:.2f}%")
                        st.info("Volatility input has been updated.")
                        st.session_state.sigma_input = iv * 100
                        st.rerun()

# --- GATHER INPUTS FOR CALCULATION ---
# After widgets are drawn, read their values for the main calculation.
# For widgets controlled by callbacks, read directly from session state for robustness.
S0 = st.session_state.s0_input
sigma_pct = st.session_state.sigma_input
lot_size = st.session_state.lot_size_input

# For user-driven widgets, use the variables assigned during their creation
K = K_val
days_to_expiry = days_to_expiry_val
option_type = option_type_val
r_pct = r_pct_val
model = model_val
steps = steps_val

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
        gc4.metric("Theta (Θ)", f"{greeks.get('Theta', 0):.5f}")
        gc5.metric("Rho (ρ)", f"{greeks.get('Rho', 0):.5f}")

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

