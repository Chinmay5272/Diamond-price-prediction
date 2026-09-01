import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* Header */
.hero {
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    font-size: 2rem;
}

.hero p {
    margin: 0.3rem 0 0;
    opacity: 0.8;
}

/* Cards */
.card {
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 0.8rem;
}

.price-card {
    padding: 1.4rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
}

.price {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.3rem 0;
}

.small {
    font-size: 0.85rem;
    opacity: 0.7;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 2.8rem;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("diamond_price_model.pkl")

model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>💎 Diamond Price Predictor</h1>
    <p>Estimate diamond prices using quality, size and physical characteristics.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Main Layout
# --------------------------------------------------

left, right = st.columns([1.4, 1], gap="large")

# --------------------------------------------------
# Inputs
# --------------------------------------------------

with left:

    st.subheader("Diamond Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.number_input(
            "Carat",
            min_value=0.10,
            max_value=5.00,
            value=1.00,
            step=0.01
        )

    with col2:
        cut = st.selectbox(
            "Cut",
            ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        )

    with col3:
        color = st.selectbox(
            "Color",
            ["D", "E", "F", "G", "H", "I", "J"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        clarity = st.selectbox(
            "Clarity",
            ["I1", "SI2", "SI1", "VS2",
             "VS1", "VVS2", "VVS1", "IF"]
        )

    with col2:
        depth = st.number_input(
            "Depth (%)",
            min_value=40.0,
            max_value=80.0,
            value=61.5,
            step=0.1
        )

    with col3:
        table = st.number_input(
            "Table (%)",
            min_value=40.0,
            max_value=80.0,
            value=57.0,
            step=0.1
        )

    st.markdown("#### 📐 Dimensions")

    col1, col2, col3 = st.columns(3)

    with col1:
        x = st.number_input(
            "Length (x)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.01
        )

    with col2:
        y = st.number_input(
            "Width (y)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.01
        )

    with col3:
        z = st.number_input(
            "Depth (z)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.01
        )

    st.markdown("")

    predict = st.button(
        "💎 Predict Diamond Price"
    )

# --------------------------------------------------
# Diamond Summary
# --------------------------------------------------

with right:

    st.subheader("Diamond Summary")

    st.markdown(
        f"""
        <div class="card">
        <b>Carat</b> &nbsp; {carat:.2f}<br>
        <b>Cut</b> &nbsp; {cut}<br>
        <b>Color</b> &nbsp; {color}<br>
        <b>Clarity</b> &nbsp; {clarity}<br>
        <b>Dimensions</b> &nbsp; {x:.2f} × {y:.2f} × {z:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Model: Tuned XGBoost Regressor")
    st.caption("Validation R²: 0.98")

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict:

    input_data = pd.DataFrame({
        "carat": [carat],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z],

        "cut_Good": [int(cut == "Good")],
        "cut_Ideal": [int(cut == "Ideal")],
        "cut_Premium": [int(cut == "Premium")],
        "cut_Very Good": [int(cut == "Very Good")],

        "color_E": [int(color == "E")],
        "color_F": [int(color == "F")],
        "color_G": [int(color == "G")],
        "color_H": [int(color == "H")],
        "color_I": [int(color == "I")],
        "color_J": [int(color == "J")],

        "clarity_IF": [int(clarity == "IF")],
        "clarity_SI1": [int(clarity == "SI1")],
        "clarity_SI2": [int(clarity == "SI2")],
        "clarity_VS1": [int(clarity == "VS1")],
        "clarity_VS2": [int(clarity == "VS2")],
        "clarity_VVS1": [int(clarity == "VVS1")],
        "clarity_VVS2": [int(clarity == "VVS2")]
    })

    # Match exact training feature order
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.markdown(
            f"""
            <div class="price-card">
                <div class="small">Estimated Price</div>
                <div class="price">${prediction:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col2:

        lower = prediction * 0.90
        upper = prediction * 1.10

        st.markdown(
            f"""
            <div class="price-card">
                <div class="small">Indicative Range</div>
                <b>${lower:,.0f} – ${upper:,.0f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col3:

        st.markdown(
            f"""
            <div class="price-card">
                <div class="small">Model R²</div>
                <div class="price">0.98</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.success(
        f"Prediction generated successfully for a {carat:.2f} carat "
        f"{cut} cut, {color} color, {clarity} clarity diamond."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Diamond Price Prediction • XGBoost Regression • "
    "Built with Python & Streamlit"
)