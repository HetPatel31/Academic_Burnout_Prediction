import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Academic Burnout Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Models
@st.cache_resource
def load_models():
    with open('models/rf_clf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

# Premium CSS injected
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        .stApp {
            background: linear-gradient(-45deg, #090a0f, #1b1e2b, #11141e, #0d0f17);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            font-family: 'Outfit', sans-serif !important;
            color: #ffffff;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Style the main Streamlit columns directly as Glass Containers */
        div[data-testid="column"] {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            transition: transform 0.3s ease;
        }
        
        div[data-testid="column"]:hover {
            transform: translateY(-2px);
        }

        /* Prevent inner columns from becoming double-glass-containers */
        div[data-testid="column"] div[data-testid="column"] {
            background: transparent !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0px !important;
            transform: none !important;
        }

        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            background: -webkit-linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0px 4px 15px rgba(0, 242, 254, 0.3);
            margin-bottom: 20px;
        }

        div[data-baseweb="slider"] { margin-bottom: 10px; }

        .stButton>button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white !important;
            border: none;
            padding: 12px 24px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 18px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
            cursor: pointer;
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5);
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        }

        .result-high {
            background: rgba(255, 75, 75, 0.1);
            border-left: 5px solid #ff4b4b;
            padding: 20px;
            border-radius: 10px;
            color: #ff4b4b;
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        .result-low {
            background: rgba(33, 195, 84, 0.1);
            border-left: 5px solid #21c354;
            padding: 20px;
            border-radius: 10px;
            color: #21c354;
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        
        .insight-box {
            background: rgba(0, 242, 254, 0.05);
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            border: 1px solid rgba(0, 242, 254, 0.2);
            font-size: 0.95rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; margin-bottom: 1rem; font-size: 3rem;'>🎓 AI Academic Burnout Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #a1a1aa; margin-bottom: 3rem;'>Enter recent academic metrics to evaluate current burnout risk levels.</p>", unsafe_allow_html=True)

# Helper for Gauge Chart
def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "Burnout Risk Level (%)", 'font': {'color': 'white', 'size': 20, 'family': 'Outfit'}},
        number = {'font': {'color': 'white', 'family': 'Outfit'}, 'suffix': "%"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(255,255,255,0.4)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': "rgba(33, 195, 84, 0.7)"},      # Green
                {'range': [30, 70], 'color': "rgba(255, 193, 7, 0.7)"},     # Yellow
                {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.7)"}     # Red
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Helper for Radar Chart
def create_radar(input_dict):
    categories = ['Quiz Avg', 'Attendance', 'Score Stability', 'Study Freq', 'Deadline Load']
    quiz = input_dict['quiz_avg_3w'][0]
    att = input_dict['attendance_avg_3w'][0]
    stab = max(0, 100 - (input_dict['quiz_std_3w'][0] * 3.33)) 
    study = min(100, (input_dict['study_sessions'][0] / 14.0) * 100)
    load = (input_dict['deadline_load'][0] / 10.0) * 100

    values = [quiz, att, stab, study, load]
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 242, 254, 0.2)',
        line=dict(color='#00f2fe', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.2)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.2)")
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='white', family='Outfit'),
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# Helper for Explainability
def get_top_contributors(X_scaled_row, feature_names, importance, original_inputs):
    impact = np.abs(X_scaled_row) * importance
    sorted_idx = np.argsort(impact)[::-1]
    
    top_1 = feature_names[sorted_idx[0]]
    val_1 = original_inputs[top_1][0]
    top_2 = feature_names[sorted_idx[1]]
    val_2 = original_inputs[top_2][0]
    
    friendly_names = {
        'quiz_avg_3w': "Quiz Average",
        'delay_avg_3w': "Submission Delay",
        'attendance_avg_3w': "Attendance",
        'quiz_trend': "Quiz Trend",
        'attendance_trend': "Attendance Trend",
        'quiz_std_3w': "Score Volatility",
        'study_sessions': "Study Sessions",
        'deadline_load': "Deadline Load"
    }
    
    return f"Your **{friendly_names[top_1]}** ({val_1}) and **{friendly_names[top_2]}** ({val_2}) are currently the most significant factors driving this specific risk prediction based on your deviations from the average student."

# Layout Container
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<h3>📊 Academic Metrics</h3>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        quiz_avg_3w = st.slider("Quiz Average (Last 3 Weeks)", 0.0, 100.0, 75.0)
        delay_avg_3w = st.slider("Submission Delay Avg (Days)", 0.0, 10.0, 1.0)
        attendance_avg_3w = st.slider("Attendance Avg (%)", 0.0, 100.0, 85.0)
        study_sessions = st.slider("Study Sessions (Per Week)", 0, 14, 4)
    with c2:
        quiz_trend = st.slider("Quiz Score Trend", -30.0, 30.0, 0.0)
        attendance_trend = st.slider("Attendance Trend", -30.0, 30.0, 0.0)
        quiz_std_3w = st.slider("Quiz Score Volatility (StdD)", 0.0, 30.0, 5.0)
        deadline_load = st.slider("Deadline Load", 1, 10, 3)

with col2:
    st.markdown('<h3>🧠 Prediction Engine</h3>', unsafe_allow_html=True)
    
    input_dict = {
        'quiz_avg_3w': [quiz_avg_3w],
        'delay_avg_3w': [delay_avg_3w],
        'attendance_avg_3w': [attendance_avg_3w],
        'quiz_trend': [quiz_trend],
        'attendance_trend': [attendance_trend],
        'quiz_std_3w': [quiz_std_3w],
        'study_sessions': [study_sessions],
        'deadline_load': [deadline_load]
    }
    
    predict_clicked = st.button("Analyze Burnout Risk")
    
    if predict_clicked:
        df_input = pd.DataFrame(input_dict)
        X_scaled = scaler.transform(df_input)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Display Result Alert
        st.markdown('<br>', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f'<div class="result-high">⚠️ High Burnout Risk Detected!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-low">✅ Low Burnout Risk</div>', unsafe_allow_html=True)
            st.balloons()
            
        # Display Gauge
        gauge_fig = create_gauge(probability)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Explainability Insight
        try:
            importances = model.feature_importances_
            feature_names = df_input.columns.tolist()
            insight_text = get_top_contributors(X_scaled[0], feature_names, importances, input_dict)
            st.markdown(f'<div class="insight-box">💡 <b>AI Insight:</b><br>{insight_text}</div>', unsafe_allow_html=True)
        except Exception as e:
            pass
            
    else:
        st.write("Ready to analyze your metrics using the trained Random Forest Classifier.")
        st.markdown("<br>", unsafe_allow_html=True)
        # Display Radar Chart as a neutral state preview
        st.markdown("<b>Your Current Profile</b>", unsafe_allow_html=True)
        radar_fig = create_radar(input_dict)
        st.plotly_chart(radar_fig, use_container_width=True)
    
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: rgba(255,255,255,0.4); font-size: 0.8rem;">
        Model: Random Forest Classifier • Accuracy Validated • Scaler: StandardScaler
    </div>
    """,
    unsafe_allow_html=True
)
