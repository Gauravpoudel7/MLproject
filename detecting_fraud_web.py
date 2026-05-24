import streamlit as st
import pandas as pd
import joblib as jl

# 1. Page Configuration
st.set_page_config(
    page_title="FraudShield AI - Logistic Regression Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load ML Model Safely
@st.cache_resource
def load_model():
    return jl.load("fraud_detection_pipeline.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ 'fraud_detection_pipeline.pkl' not found. Please ensure your trained pipeline file is in the root folder.")
    st.stop()

# 3. Premium Dark Charcoal & Gold Terminal CSS Injection
st.markdown("""
    <style>
    /* Main Background (Deep Charcoal/Navy Slate) */
    .stApp {
        background-color: #0B0F19 !important;
    }
    
    /* General Text Color (Off-White and Pale Gold for premium contrast) */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #F1F5F9 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Section Headings highlighted in Amber Gold */
    h3 {
        color: #F59E0B !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #F59E0B;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        color: #F59E0B !important;
    }

    /* Style Form Inputs & Dropdowns to match the terminal look */
    div[data-baseweb="select"], div[data-baseweb="input"] {
        background-color: #1F2937 !important;
        border-radius: 8px;
    }
    
    /* Premium High-Risk Alert Card */
    .fraud-card {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.3);
        text-align: center;
        margin-top: 25px;
        transition: transform 0.3s ease;
        border: 1px solid #EF4444;
    }
    
    /* Premium Clean/Safe Result Card */
    .clean-card {
        background: linear-gradient(135deg, #059669 0%, #065F46 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.3);
        text-align: center;
        margin-top: 25px;
        transition: transform 0.3s ease;
        border: 1px solid #10B981;
    }
    
    .result-title {
        color: #FFFFFF !important;
        letter-spacing: 2px;
        font-size: 14px;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    .result-status {
        font-size: 42px !important;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 5px 0;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .result-note {
        color: #F3F4F6 !important;
        font-size: 15px;
        margin: 0;
    }
    
    /* Action Button Customization (Gold Theme + Interactive Scale) */
    div.stButton > button:first-child {
        background-color: #F59E0B !important;
        color: #0B0F19 !important;
        border: none;
        border-radius: 12px;
        padding: 15px 30px !important; 
        font-size: 22px !important;    
        font-weight: 800 !important;    
        letter-spacing: 1px;            
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    div.stButton > button:first-child:hover {
        background-color: #FBBF24 !important;
        box-shadow: 0 6px 25px rgba(245, 158, 11, 0.5);
        transform: scale(1.02);         
    }
    
    /* Divider lines */
    hr {
        border-color: #F59E0B !important;
    }
    </style>
""", unsafe_allow_html=True)

# 4. UI Layout - Sidebar
with st.sidebar:
    st.title("🛡️ Secure Node")
    st.markdown("""
    **Mathematical Engine:**
    * Classification: Logistic Regression
    * Boundary Analysis: Sigmoid Probability Mapping
    
    **System Status:** 
    * Network Core: Connected
    * Core Pipeline: Loaded
    * UI Theme: Amber Executive
    """)
    st.info("Update ledger variables on the right to trigger calculation.")

# 5. UI Layout - Main App Header
st.title("🛡️ FraudShield Core Predictive Engine")
st.markdown("### *High-volume transactional analysis environment.*")
st.caption("Utilizing log-odds coefficients and probability mapping to flag potential financial asset diversion.")

st.write("---")

# 6. Interactive Inputs Grid
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown("### 📊 Transaction Overview")
    
    transaction_type = st.selectbox(
        "Transaction Classification",
        ["PAYMENT", "TRANSFER", "CASHOUT", "DEPOSITS"]
    )
    
    amount = st.number_input(
        "Volumetric Amount ($)", 
        min_value=0.0, 
        value=1000.00,
        step=50.0,
        format="%.2f"
    )

with row2_col2:
    st.markdown("### 🔑 Structural Ledger Balances")
    
    # Sub-columns to split Origin and Destination accounts nicely
    orig_col, dest_col = st.columns(2)
    
    with orig_col:
        st.markdown("<small style='color:#F59E0B'>SENDER ACCOUNT (ORG)</small>", unsafe_allow_html=True)
        oldbalanceOrg = st.number_input(
            "Initial Balance", 
            min_value=0.0, 
            value=1000.00,
            key="org_old",
            format="%.2f"
        )
        newbalanceOrig = st.number_input(
            "Post-Balance", 
            min_value=0.0, 
            value=9000.00,
            key="org_new",
            format="%.2f"
        )
        
    with dest_col:
        st.markdown("<small style='color:#F59E0B'>RECEIVER ACCOUNT (DEST)</small>", unsafe_allow_html=True)
        oldbalanceDest = st.number_input(
            "Initial Balance", 
            min_value=0.0, 
            value=0.0,
            key="dest_old",
            format="%.2f"
        )
        newbalanceDest = st.number_input(
            "Post-Balance", 
            min_value=0.0, 
            value=0.0,
            key="dest_new",
            format="%.2f"
        )

st.write("---")

# 7. Execution Block
left_spacer, center_btn, right_spacer = st.columns([1, 1, 1])
with center_btn:
    assess_clicked = st.button("🛡️ RUN RISK ASSESSMENT", use_container_width=True)

if assess_clicked:
    # Build feature structure exactly matching model pipeline expectations
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "newbalanceDest": newbalanceDest,
        "oldbalanceDest": oldbalanceDest,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig
    }])
    
    # Extract structural components from the Logistic Regression model
    prediction = model.predict(input_data)[0]
    
    # predict_proba returns [[prob_of_0, prob_of_1]]. We grab prob_of_1 for fraud percentage.
    probabilities = model.predict_proba(input_data)[0]
    fraud_probability = probabilities[1] * 100 
    
    # Render customized executive banners depending on prediction output
    if prediction == 1:
        st.markdown(f"""
            <div class="fraud-card">
                <div class="result-title">Security Alert Flag</div>
                <div class="result-status">⚠️ FRAUD THREAT DETECTED</div>
                <div style="font-size: 24px; font-weight: 700; color: #FFF; margin-bottom: 10px;">
                    Confidence Metrics: {fraud_probability:.2f}% Match Probability
                </div>
                <p class="result-note">This vector closely satisfies known log-odds metrics of laundering or illegal cash divergence patterns.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="clean-card">
                <div class="result-title">Security Verification</div>
                <div class="result-status">✅ TRANSACT CLEAN</div>
                <div style="font-size: 24px; font-weight: 700; color: #FFF; margin-bottom: 10px;">
                    Anomalous Variance: {fraud_probability:.2f}% Base Risk
                </div>
                <p class="result-note">Structural auditing completed via sigmoid thresholding. No significant asset variance anomalies detected.</p>
            </div>
        """, unsafe_allow_html=True)