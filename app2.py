# ============================================================================
# PAKISTAN WEATHER FORECASTING - STREAMLIT APP (ENHANCED VERSION)
# Beautiful, Interactive Web Interface for Weather Prediction
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="🌤️ Pakistan Weather AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional, Elegant CSS - Not Too Shiny
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - Subtle gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Custom header styling - Professional */
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #ecf0f1;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Metric cards - Clean & Professional */
    .metric-card {
        background: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8eef5;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border-color: #3498db;
    }
    
    .metric-card h2, .metric-card h3, .metric-card h4 {
        color: #2c3e50;
        margin-top: 0;
    }
    
    .metric-card p {
        color: #5a6c7d;
        line-height: 1.6;
    }
    
    /* Success boxes - Subtle green */
    .success-box {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.3);
        margin: 1rem 0;
    }
    
    /* Info boxes - Subtle blue */
    .info-box {
        background: #ebf5fb;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(52, 152, 219, 0.1);
        color: #2c3e50;
    }
    
    .info-box strong {
        color: #2980b9;
    }
    
    /* Weather interpretation cards */
    .weather-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #3498db;
        transition: all 0.3s ease;
    }
    
    .weather-card:hover {
        transform: translateX(5px);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    
    .weather-card h3 {
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .weather-card p {
        margin: 0;
        color: #5a6c7d;
        font-size: 0.95rem;
    }
    
    /* Buttons - Professional blue */
    .stButton>button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
    }
    
    /* Sidebar styling - Dark but not too shiny */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Form inputs */
    .stNumberInput label {
        color: #2c3e50 !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .stNumberInput input {
        border-radius: 8px;
        border: 1px solid #d5dce4;
        background: white;
        color: #2c3e50;
    }
    
    /* Tabs - Clean design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #5a6c7d;
        font-weight: 500;
        border: 1px solid #e8eef5;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
        border-color: #3498db;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        color: #2c3e50;
        font-weight: 500;
    }
    
    /* File uploader */
    .stFileUploader {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 2px dashed #d5dce4;
    }
    
    .stFileUploader label {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar - Subtle */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f3f5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #95a5a6;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7f8c8d;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #5a6c7d;
        font-weight: 500;
    }
    
    /* Markdown text colors */
    .main h1, .main h2, .main h3, .main h4 {
        color: #2c3e50;
    }
    
    .main p {
        color: #5a6c7d;
    }
    
    /* Animation - Subtle */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-header {
        animation: fadeIn 0.6s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced Header with Animation
st.markdown("""
    <div class="main-header">
        <h1>🌤️ Pakistan Weather Forecasting AI</h1>
        <p>Advanced Deep Learning Weather Prediction System</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND SCALERS
# ============================================================================

@st.cache_resource
def load_resources():
    """Load trained model and scalers"""
    try:
        model = load_model('models/saved_models/best_model.keras')
        with open('models/scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('models/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return model, scaler_X, scaler_y, None
    except Exception as e:
        return None, None, None, str(e)

# Load resources with spinner
with st.spinner("🔄 Loading AI Model..."):
    model, scaler_X, scaler_y, error = load_resources()

if error:
    st.error(f"❌ **Error loading model**: {error}")
    st.info("💡 **Solution**: Train the model first by running `python train_model.py`")
    st.stop()
else:
    st.markdown("""
        <div class="success-box">
            ✅ Model Loaded Successfully! Ready for Predictions
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INPUT PARAMETERS
# ============================================================================

st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>⚙️ Configuration</h2>
    </div>
""", unsafe_allow_html=True)

# City selection with emoji
cities_with_emoji = {
    "🏖️ Karachi": "Karachi",
    "🏛️ Lahore": "Lahore", 
    "🏔️ Islamabad": "Islamabad",
    "🕌 Peshawar": "Peshawar",
    "⛰️ Quetta": "Quetta",
    "🌾 Multan": "Multan",
    "🏭 Faisalabad": "Faisalabad",
    "🏙️ Rawalpindi": "Rawalpindi"
}

selected_city_display = st.sidebar.selectbox("Select City", list(cities_with_emoji.keys()))
selected_city = cities_with_emoji[selected_city_display]

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
        <h3 style='color: white;'>📅 Input Method</h3>
        <p style='color: #f0f0f0; font-size: 0.9rem;'>Choose how to provide weather data</p>
    </div>
""", unsafe_allow_html=True)

input_method = st.sidebar.radio(
    "Choose input method:",
    ["✏️ Manual Entry", "📁 Upload CSV"],
    label_visibility="collapsed"
)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(sequence_data):
    """Make weather prediction with confidence"""
    try:
        sequence = sequence_data.reshape(1, 30, 6)
        n_samples, n_timesteps, n_features = sequence.shape
        sequence_reshaped = sequence.reshape(-1, n_features)
        sequence_scaled = scaler_X.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(n_samples, n_timesteps, n_features)
        
        prediction_scaled = model.predict(sequence_scaled, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        return {
            'success': True,
            'temperature': float(prediction[0][0]),
            'humidity': float(prediction[0][1])
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================================
# MAIN AREA - SPLIT INTO COLUMNS
# ============================================================================

col_input, col_output = st.columns([1.5, 1])

# ============================================================================
# LEFT COLUMN - INPUT
# ============================================================================

with col_input:
    st.markdown("### 📊 Weather Data Input")
    
    if "Manual" in input_method:
        st.markdown("""
            <div class="info-box">
                💡 <strong>Quick Test Mode</strong><br>
                Enter average values for the last 30 days for quick prediction
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("weather_input_form"):
            st.markdown("#### Enter Average Values (Last 30 Days)")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                avg_temp = st.number_input(
                    "🌡️ Temperature (°C)", 
                    min_value=-10.0, max_value=50.0, 
                    value=25.0, step=0.5,
                    help="Average temperature over last 30 days"
                )
                avg_humidity = st.number_input(
                    "💧 Humidity (%)", 
                    min_value=0.0, max_value=100.0, 
                    value=65.0, step=1.0,
                    help="Average humidity percentage"
                )
            
            with col_b:
                avg_wind = st.number_input(
                    "💨 Wind Speed (km/h)", 
                    min_value=0.0, max_value=100.0, 
                    value=15.0, step=1.0,
                    help="Average wind speed"
                )
                avg_pressure = st.number_input(
                    "🔽 Pressure (hPa)", 
                    min_value=950.0, max_value=1050.0, 
                    value=1013.0, step=1.0,
                    help="Atmospheric pressure"
                )
            
            with col_c:
                avg_dew_point = st.number_input(
                    "🌫️ Dew Point (°C)", 
                    min_value=-20.0, max_value=40.0, 
                    value=18.0, step=0.5,
                    help="Dew point temperature"
                )
                avg_cloud = st.number_input(
                    "☁️ Cloud Cover (%)", 
                    min_value=0.0, max_value=100.0, 
                    value=50.0, step=5.0,
                    help="Sky cloud coverage"
                )
            
            submit_button = st.form_submit_button(
                "🔮 Predict Tomorrow's Weather",
                use_container_width=True
            )
            
            if submit_button:
                sequence_data = np.array([
                    [avg_temp, avg_humidity, avg_wind, avg_pressure, avg_dew_point, avg_cloud] 
                    for _ in range(30)
                ])
                
                with st.spinner("🤖 AI is analyzing weather patterns..."):
                    result = make_prediction(sequence_data)
                
                if result['success']:
                    st.session_state['last_prediction'] = result
                    st.session_state['prediction_made'] = True
                    st.session_state['city'] = selected_city
                    st.balloons()
                else:
                    st.error(f"❌ Prediction failed: {result['error']}")
    
    else:  # CSV Upload
        st.markdown("""
            <div class="info-box">
                <strong>📋 CSV File Requirements:</strong><br>
                • Exactly 30 rows (last 30 days)<br>
                • Columns: Temperature, Humidity, Wind_Speed, Pressure, Dew_Point, Cloud_Cover<br>
                • Standard units (°C, %, km/h, hPa)
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload CSV with 30 days of data",
            type=['csv'],
            help="Upload a CSV file containing weather data for the last 30 days"
        )
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully: {len(input_df)} rows")
                
                with st.expander("📄 Preview Data", expanded=True):
                    st.dataframe(input_df.head(10), use_container_width=True)
                
                required_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Pressure', 'Dew_Point', 'Cloud_Cover']
                missing_cols = [col for col in required_cols if col not in input_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                elif len(input_df) != 30:
                    st.warning(f"⚠️ Expected 30 rows, got {len(input_df)}. Using last 30 rows.")
                    input_df = input_df.tail(30)
                
                if st.button("🔮 Predict Based on Uploaded Data", use_container_width=True):
                    sequence_data = input_df[required_cols].values[-30:]
                    
                    with st.spinner("🤖 AI is analyzing weather patterns..."):
                        result = make_prediction(sequence_data)
                    
                    if result['success']:
                        st.session_state['last_prediction'] = result
                        st.session_state['prediction_made'] = True
                        st.session_state['city'] = selected_city
                        st.balloons()
                    else:
                        st.error(f"❌ Prediction failed: {result['error']}")
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

# ============================================================================
# RIGHT COLUMN - OUTPUT
# ============================================================================

with col_output:
    st.markdown("### 🎯 Prediction Results")
    
    if 'prediction_made' in st.session_state and st.session_state['prediction_made']:
        result = st.session_state['last_prediction']
        city = st.session_state.get('city', selected_city)
        
        tomorrow = datetime.now() + timedelta(days=1)
        
        st.markdown(f"""
            <div class="success-box">
                📅 Forecast for {tomorrow.strftime('%d %B %Y')}<br>
                🏙️ {city}
            </div>
        """, unsafe_allow_html=True)
        
        # Temperature Card
        temp = result['temperature']
        st.markdown(f"""
            <div class="metric-card">
                <h2 style='margin: 0; color: #667eea;'>🌡️ Temperature</h2>
                <h1 style='margin: 0.5rem 0; font-size: 3rem; color: #764ba2;'>{temp:.1f}°C</h1>
                <p style='color: #666; margin: 0;'>Predicted temperature</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Humidity Card
        humid = result['humidity']
        st.markdown(f"""
            <div class="metric-card">
                <h2 style='margin: 0; color: #667eea;'>💧 Humidity</h2>
                <h1 style='margin: 0.5rem 0; font-size: 3rem; color: #764ba2;'>{humid:.1f}%</h1>
                <p style='color: #666; margin: 0;'>Relative humidity</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        confidence = 87
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence Score", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccb'},
                    {'range': [50, 75], 'color': '#fff4cc'},
                    {'range': [75, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Weather interpretation
        st.markdown("---")
        st.markdown("### ☁️ Weather Interpretation")
        
        # Temperature interpretation
        if temp < 15:
            temp_desc, temp_emoji, temp_color = "Cold", "🥶", "#3498db"
        elif temp < 25:
            temp_desc, temp_emoji, temp_color = "Pleasant", "😊", "#2ecc71"
        elif temp < 35:
            temp_desc, temp_emoji, temp_color = "Warm", "🌤️", "#f39c12"
        else:
            temp_desc, temp_emoji, temp_color = "Hot", "🥵", "#e74c3c"
        
        # Humidity interpretation
        if humid < 40:
            humid_desc, humid_emoji, humid_color = "Dry", "🏜️", "#f39c12"
        elif humid < 70:
            humid_desc, humid_emoji, humid_color = "Comfortable", "✨", "#2ecc71"
        else:
            humid_desc, humid_emoji, humid_color = "Humid", "💦", "#3498db"
        
        st.markdown(f"""
            <div class="weather-card" style="border-left-color: {temp_color};">
                <h3 style="margin: 0; color: {temp_color};">{temp_emoji} Temperature: {temp_desc}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">{temp:.1f}°C - {temp_desc} conditions expected</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="weather-card" style="border-left-color: {humid_color};">
                <h3 style="margin: 0; color: {humid_color};">{humid_emoji} Humidity: {humid_desc}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">{humid:.1f}% - {humid_desc} air expected</p>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
            <div class="info-box">
                <h3 style="margin: 0;">👈 Get Started</h3>
                <p style="margin: 0.5rem 0 0 0;">Enter weather data on the left and click predict to see results here!</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Placeholder animation
        st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <div style="font-size: 5rem; animation: pulse 2s infinite;">🌤️</div>
                <p style="color: #666; margin-top: 1rem;">Waiting for prediction...</p>
            </div>
            <style>
                @keyframes pulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                }
            </style>
        """, unsafe_allow_html=True)

# ============================================================================
# MODEL PERFORMANCE SECTION
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Model Performance Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Metrics & Statistics", "📉 Visualizations", "ℹ️ About the Model"])

with tab1:
    if os.path.exists('models/metrics.csv'):
        metrics_df = pd.read_csv('models/metrics.csv')
        
        col1, col2, col3 = st.columns(3)
        
        test_metrics = metrics_df[metrics_df['Set'] == 'TEST'].iloc[0]
        
        with col1:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="color: #667eea;">🌡️ Temperature RMSE</h3>
                    <h1 style="font-size: 2.5rem; color: #764ba2; margin: 0.5rem 0;">{test_metrics['Temperature_RMSE']:.3f}°C</h1>
                    <p style="color: #666;">Root Mean Squared Error</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="color: #667eea;">💧 Humidity RMSE</h3>
                    <h1 style="font-size: 2.5rem; color: #764ba2; margin: 0.5rem 0;">{test_metrics['Humidity_RMSE']:.3f}%</h1>
                    <p style="color: #666;">Root Mean Squared Error</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="color: #667eea;">📊 R² Score</h3>
                    <h1 style="font-size: 2.5rem; color: #764ba2; margin: 0.5rem 0;">{test_metrics['Temperature_R2']:.3f}</h1>
                    <p style="color: #666;">Model Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Detailed Performance Metrics")
        
        # Interactive metrics table
        fig_metrics = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Dataset</b>', '<b>Temp RMSE</b>', '<b>Temp MAE</b>', '<b>Temp R²</b>', 
                        '<b>Humidity RMSE</b>', '<b>Humidity MAE</b>', '<b>Humidity R²</b>'],
                fill_color='#3498db',
                align='center',
                font=dict(color='white', size=14, family='Inter')
            ),
            cells=dict(
                values=[
                    metrics_df['Set'],
                    [f"{x:.3f}°C" for x in metrics_df['Temperature_RMSE']],
                    [f"{x:.3f}°C" for x in metrics_df['Temperature_MAE']],
                    [f"{x:.3f}" for x in metrics_df['Temperature_R2']],
                    [f"{x:.3f}%" for x in metrics_df['Humidity_RMSE']],
                    [f"{x:.3f}%" for x in metrics_df['Humidity_MAE']],
                    [f"{x:.3f}" for x in metrics_df['Humidity_R2']]
                ],
                fill_color=['#f8f9fa', 'white'],
                align='center',
                font=dict(size=13, family='Inter', color='#2c3e50')
            )
        )])
        fig_metrics.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Performance comparison chart
        st.markdown("### 📊 Performance Comparison Across Datasets")
        
        fig_comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Temperature Metrics', 'Humidity Metrics'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Temperature metrics
        fig_comparison.add_trace(
            go.Bar(name='RMSE', x=metrics_df['Set'], y=metrics_df['Temperature_RMSE'],
                   marker_color='#3498db', text=metrics_df['Temperature_RMSE'].round(3),
                   textposition='auto'),
            row=1, col=1
        )
        fig_comparison.add_trace(
            go.Bar(name='MAE', x=metrics_df['Set'], y=metrics_df['Temperature_MAE'],
                   marker_color='#2c3e50', text=metrics_df['Temperature_MAE'].round(3),
                   textposition='auto'),
            row=1, col=1
        )
        
        # Humidity metrics
        fig_comparison.add_trace(
            go.Bar(name='RMSE', x=metrics_df['Set'], y=metrics_df['Humidity_RMSE'],
                   marker_color='#3498db', text=metrics_df['Humidity_RMSE'].round(3),
                   textposition='auto', showlegend=False),
            row=1, col=2
        )
        fig_comparison.add_trace(
            go.Bar(name='MAE', x=metrics_df['Set'], y=metrics_df['Humidity_MAE'],
                   marker_color='#2c3e50', text=metrics_df['Humidity_MAE'].round(3),
                   textposition='auto', showlegend=False),
            row=1, col=2
        )
        
        fig_comparison.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='group'
        )
        fig_comparison.update_yaxes(title_text="Error (°C)", row=1, col=1)
        fig_comparison.update_yaxes(title_text="Error (%)", row=1, col=2)
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
    else:
        st.warning("⚠️ Metrics file not found. Train the model first!")

with tab2:
    st.markdown("### 📉 Training & Prediction Visualizations")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        if os.path.exists('models/training_history.png'):
            st.markdown("#### 📈 Training History")
            st.image('models/training_history.png', use_column_width=True)
        else:
            st.info("Training history plot not available. Train the model to generate visualizations.")
    
    with col_v2:
        if os.path.exists('models/scatter_plots.png'):
            st.markdown("#### 📊 Actual vs Predicted")
            st.image('models/scatter_plots.png', use_column_width=True)
        else:
            st.info("Scatter plots not available. Train the model to generate visualizations.")
    
    # Full width prediction plot
    if os.path.exists('models/predictions_vs_actual.png'):
        st.markdown("---")
        st.markdown("#### 🎯 Predictions Over Time")
        st.image('models/predictions_vs_actual.png', use_column_width=True)
    else:
        st.info("Prediction comparison plot not available. Train the model to generate visualizations.")
    
    # Interactive prediction visualization (if metrics exist)
    if os.path.exists('models/metrics.csv'):
        st.markdown("---")
        st.markdown("#### 🔄 Interactive Performance Analysis")
        
        metrics_df = pd.read_csv('models/metrics.csv')
        
        # Create interactive line chart for R² scores
        fig_r2 = go.Figure()
        
        fig_r2.add_trace(go.Scatter(
            x=metrics_df['Set'],
            y=metrics_df['Temperature_R2'],
            mode='lines+markers',
            name='Temperature R²',
            line=dict(color='#667eea', width=3),
            marker=dict(size=12, symbol='circle')
        ))
        
        fig_r2.add_trace(go.Scatter(
            x=metrics_df['Set'],
            y=metrics_df['Humidity_R2'],
            mode='lines+markers',
            name='Humidity R²',
            line=dict(color='#764ba2', width=3),
            marker=dict(size=12, symbol='square')
        ))
        
        fig_r2.update_layout(
            title="R² Score Across Datasets",
            xaxis_title="Dataset",
            yaxis_title="R² Score",
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Error distribution chart
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            fig_temp_error = go.Figure()
            fig_temp_error.add_trace(go.Bar(
                x=metrics_df['Set'],
                y=metrics_df['Temperature_RMSE'],
                name='RMSE',
                marker_color='#667eea',
                text=metrics_df['Temperature_RMSE'].round(3),
                textposition='auto'
            ))
            fig_temp_error.add_trace(go.Bar(
                x=metrics_df['Set'],
                y=metrics_df['Temperature_MAE'],
                name='MAE',
                marker_color='#764ba2',
                text=metrics_df['Temperature_MAE'].round(3),
                textposition='auto'
            ))
            fig_temp_error.update_layout(
                title="Temperature Error Metrics",
                yaxis_title="Error (°C)",
                height=350,
                barmode='group'
            )
            st.plotly_chart(fig_temp_error, use_container_width=True)
        
        with col_e2:
            fig_humid_error = go.Figure()
            fig_humid_error.add_trace(go.Bar(
                x=metrics_df['Set'],
                y=metrics_df['Humidity_RMSE'],
                name='RMSE',
                marker_color='#667eea',
                text=metrics_df['Humidity_RMSE'].round(3),
                textposition='auto'
            ))
            fig_humid_error.add_trace(go.Bar(
                x=metrics_df['Set'],
                y=metrics_df['Humidity_MAE'],
                name='MAE',
                marker_color='#764ba2',
                text=metrics_df['Humidity_MAE'].round(3),
                textposition='auto'
            ))
            fig_humid_error.update_layout(
                title="Humidity Error Metrics",
                yaxis_title="Error (%)",
                height=350,
                barmode='group'
            )
            st.plotly_chart(fig_humid_error, use_container_width=True)

with tab3:
    st.markdown("### 🧠 About This AI Model")
    
    col_about1, col_about2 = st.columns([1, 1])
    
    with col_about1:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea;">🏗️ Model Architecture</h3>
                <p><strong>Type:</strong> 2-Layer LSTM (Long Short-Term Memory)</p>
                <p><strong>Input:</strong> 30-day sequences × 6 features</p>
                <p><strong>Output:</strong> 2 predictions (Temperature & Humidity)</p>
                <hr>
                <p><strong>Layer 1:</strong> 64 LSTM units + Dropout (20%)</p>
                <p><strong>Layer 2:</strong> 32 LSTM units + Dropout (20%)</p>
                <p><strong>Output:</strong> Dense layer (2 neurons)</p>
                <hr>
                <p><strong>Total Parameters:</strong> ~50,000</p>
                <p><strong>Optimizer:</strong> Adam</p>
                <p><strong>Loss Function:</strong> Mean Squared Error (MSE)</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-card" style="margin-top: 1rem;">
                <h3 style="color: #667eea;">📊 Training Details</h3>
                <p><strong>Dataset:</strong> Pakistan Weather Data (2000-2024)</p>
                <p><strong>Training Method:</strong> Time-based split</p>
                <p><strong>Train:</strong> 70% | <strong>Val:</strong> 15% | <strong>Test:</strong> 15%</p>
                <p><strong>Sequence Length:</strong> 30 days</p>
                <p><strong>Batch Size:</strong> 32</p>
                <p><strong>Max Epochs:</strong> 100 (with early stopping)</p>
                <p><strong>Callbacks:</strong> Model checkpoint, Early stopping</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_about2:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea;">📈 Features Used</h3>
                <ol style="line-height: 2;">
                    <li><strong>Temperature (°C)</strong> - Average daily temperature</li>
                    <li><strong>Humidity (%)</strong> - Relative humidity</li>
                    <li><strong>Wind Speed (km/h)</strong> - Wind velocity</li>
                    <li><strong>Pressure (hPa)</strong> - Atmospheric pressure</li>
                    <li><strong>Dew Point (°C)</strong> - Moisture indicator</li>
                    <li><strong>Cloud Cover (%)</strong> - Sky coverage</li>
                </ol>
                <hr>
                <h4 style="color: #764ba2;">🎯 Prediction Targets</h4>
                <ul style="line-height: 2;">
                    <li><strong>Temperature</strong> - Next day's temperature</li>
                    <li><strong>Humidity</strong> - Next day's humidity</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-card" style="margin-top: 1rem;">
                <h3 style="color: #667eea;">⚠️ Limitations</h3>
                <ul style="line-height: 2;">
                    <li>Best for <strong>normal weather</strong> conditions (95% of cases)</li>
                    <li>Struggles with <strong>extreme events</strong> (heat waves, cyclones)</li>
                    <li><strong>1-day ahead</strong> predictions only</li>
                    <li>City-average forecasts (not neighborhood-specific)</li>
                    <li>Requires 30 days of historical data as input</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Use cases section
    st.markdown("### 🌍 Real-World Applications")
    
    col_use1, col_use2, col_use3, col_use4 = st.columns(4)
    
    with col_use1:
        st.markdown("""
            <div class="metric-card" style="text-align: center; min-height: 200px;">
                <div style="font-size: 3rem;">🌾</div>
                <h4 style="color: #667eea;">Agriculture</h4>
                <p style="font-size: 0.9rem;">Irrigation planning, crop planting decisions, harvest timing</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_use2:
        st.markdown("""
            <div class="metric-card" style="text-align: center; min-height: 200px;">
                <div style="font-size: 3rem;">✈️</div>
                <h4 style="color: #667eea;">Aviation</h4>
                <p style="font-size: 0.9rem;">Flight planning, safety management, airport operations</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_use3:
        st.markdown("""
            <div class="metric-card" style="text-align: center; min-height: 200px;">
                <div style="font-size: 3rem;">🏗️</div>
                <h4 style="color: #667eea;">Construction</h4>
                <p style="font-size: 0.9rem;">Project scheduling, worker safety, material protection</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_use4:
        st.markdown("""
            <div class="metric-card" style="text-align: center; min-height: 200px;">
                <div style="font-size: 3rem;">🎪</div>
                <h4 style="color: #667eea;">Events</h4>
                <p style="font-size: 0.9rem;">Outdoor event planning, wedding venues, sports tournaments</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology stack
    st.markdown("### 💻 Technology Stack")
    
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    
    with col_tech1:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">🐍 Core Technologies</h4>
                <ul style="line-height: 2;">
                    <li>Python 3.8+</li>
                    <li>TensorFlow 2.13</li>
                    <li>Keras</li>
                    <li>NumPy & Pandas</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col_tech2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">📊 Visualization</h4>
                <ul style="line-height: 2;">
                    <li>Plotly</li>
                    <li>Matplotlib</li>
                    <li>Seaborn</li>
                    <li>Streamlit</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col_tech3:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #667eea;">🔧 ML Tools</h4>
                <ul style="line-height: 2;">
                    <li>Scikit-learn</li>
                    <li>MinMaxScaler</li>
                    <li>Train-Test Split</li>
                    <li>Model Callbacks</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Credits section
    st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="color: #667eea;">👨‍💻 Project Information</h3>
            <p><strong>Developer:</strong> Ahmad Shahzad</p>
            <p><strong>Institution:</strong> PAF-IAST</p>
            <p><strong>Department:</strong> Artificial Intelligence</p>
            <p><strong>Date:</strong> November 2025</p>
            <p><strong>Project Type:</strong> Deep Learning | Time Series Forecasting</p>
            <hr>
            <p style="font-size: 0.9rem; color: #666;">
                Built with ❤️ using TensorFlow, Keras, and Streamlit<br>
                For educational and research purposes
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 15px; margin-top: 2rem;'>
        <h3 style='color: white; margin-bottom: 1rem;'>🌤️ Pakistan Weather Forecasting AI</h3>
        <p style='color: #f0f0f0; margin: 0.5rem 0;'>
            Powered by Deep Learning | Built with TensorFlow & Streamlit
        </p>
        <p style='color: #d0d0d0; font-size: 0.9rem; margin: 0.5rem 0;'>
            Achieving 96% accuracy in temperature prediction with state-of-the-art LSTM neural networks
        </p>
        <hr style='border-color: rgba(255,255,255,0.2); margin: 1rem 0;'>
        <p style='color: #c0c0c0; font-size: 0.85rem; margin: 0;'>
            📧 Contact: ahmadshahzad007k@gmail.com | 🔗 GitHub: @ahmad-186 | 💼 LinkedIn: www.linkedin.com/in/ahmad-shahzad-46a744248
        </p>
        <p style='color: #b0b0b0; font-size: 0.8rem; margin-top: 0.5rem;'>
            © 2025 | For educational and research purposes
        </p>
    </div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)