import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set page title and configuration
st.set_page_config(
    page_title="Conversion Attribution Simulator",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Conversion Attribution Simulator")
st.markdown("""
This interactive simulator allows you to explore how different factors affect conversion probability 
and attribution in digital advertising campaigns using the Criteo Attribution Dataset.
""")

# Load models
@st.cache_resource
def load_models():
    models = {}
    
    # Try to load the new models first (if they exist)
    if os.path.exists('models/conversion_model_new.pkl'):
        with open('models/conversion_model_new.pkl', 'rb') as f:
            model = pickle.load(f)
            
            # For a pipeline, you can check the preprocessing step
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, columns in preprocessor.transformers_:
                        print(f"Transformer: {name}, Columns: {columns}")
            models['conversion'] = model
        st.success("✅ Loaded new conversion model")
    elif os.path.exists('models/conversion_model.pkl'):
        with open('models/conversion_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
            # For a pipeline, you can check the preprocessing step
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, columns in preprocessor.transformers_:
                        print(f"Transformer: {name}, Columns: {columns}")
            models['conversion'] = model
        st.success("✅ Loaded original conversion model")
    else:
        st.warning("⚠️ No conversion model found!")

    if os.path.exists('models/attribution_model_new.pkl'):
        with open('models/attribution_model_new.pkl', 'rb') as f:
            models['attribution'] = pickle.load(f)
        st.success("✅ Loaded new attribution model")
    elif os.path.exists('models/attribution_model.pkl'):
        with open('models/attribution_model.pkl', 'rb') as f:
            models['attribution'] = pickle.load(f)
        st.success("✅ Loaded original attribution model")
    else:
        st.warning("⚠️ No attribution model found!")
    
    return models

# Load sample data for reference values
@st.cache_data
def load_sample_data():
    try:
        if os.path.exists('processed_attribution_data.csv'):
            data = pd.read_csv('processed_attribution_data.csv')
            return data
        elif os.path.exists('criteo_attribution_sample.csv'):
            data = pd.read_csv('criteo_attribution_sample.csv')
            return data
        else:
            st.warning("⚠️ No sample data found. Using default values.")
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

# Load models and data
models = load_models()
sample_data = load_sample_data()

# Get campaign IDs from sample data
campaign_options = ["1"]  # Default value
if sample_data is not None and 'campaign' in sample_data.columns:
    campaign_options = sample_data['campaign'].unique().tolist()
    campaign_options = [str(c) for c in campaign_options]

# Sidebar for input parameters
st.sidebar.header("Campaign Parameters")

# Campaign selection
campaign = st.sidebar.selectbox(
    "Campaign ID",
    options=campaign_options,
    index=0
)

# Click parameters
st.sidebar.subheader("Click Parameters")
click = st.sidebar.checkbox("User clicked on ad", value=True)
click_pos = st.sidebar.slider("Click position (0 = first click)", 0, 5, 0, disabled=not click)
click_nb = st.sidebar.slider("Number of clicks", 1, 10, 1, disabled=not click)
time_since_last_click = st.sidebar.slider(
    "Time since last click (hours)", 
    0.0, 24.0, 1.0, 0.1, 
    disabled=not click
) * 3600  # Convert to seconds

# Cost parameters
st.sidebar.subheader("Cost Parameters")
cost = st.sidebar.slider("Cost of impression", 0.0, 10.0, 1.0, 0.1)

# User parameters
st.sidebar.subheader("User Parameters")
user_impression_count = st.sidebar.slider("Number of impressions for this user", 1, 50, 5)
user_click_rate = st.sidebar.slider("User's historical click rate", 0.0, 1.0, 0.2, 0.01)
user_conversion_rate = st.sidebar.slider("User's historical conversion rate", 0.0, 1.0, 0.1, 0.01)

# Campaign parameters
st.sidebar.subheader("Campaign Performance")
campaign_ctr = st.sidebar.slider("Campaign click-through rate", 0.0, 1.0, 0.15, 0.01)
campaign_conversion_rate = st.sidebar.slider("Campaign conversion rate", 0.0, 1.0, 0.05, 0.01)

# Derived features
cost_per_click = cost if click else 0
click_campaign_interaction = click * campaign_ctr
cost_campaign_conv_interaction = cost * campaign_conversion_rate

# Create input data frame
input_data = pd.DataFrame({
    'timestamp': [np.random.randint(0, 100000)],  # Random timestamp
    'campaign': [campaign],
    'click': [1 if click else 0],
    'click_pos': [click_pos],
    'click_nb': [click_nb],
    'cost': [cost],
    'time_since_last_click': [time_since_last_click],
    'impression_hour': [np.random.randint(0, 24)],  # Random hour
    'time_to_conversion_hours': [-1],  # Unknown at prediction time
    'user_impression_count': [user_impression_count],
    'user_click_rate': [user_click_rate],
    'user_conversion_rate': [user_conversion_rate],
    'campaign_ctr': [campaign_ctr],
    'campaign_conversion_rate': [campaign_conversion_rate],
    'cost_per_click': [cost_per_click],
    'click_campaign_interaction': [click_campaign_interaction],
    'cost_campaign_conv_interaction': [cost_campaign_conv_interaction]
})

# Main content area
st.header("Prediction Results")

# Make predictions if models are available
if 'conversion' in models:
    # Get conversion probability
    conversion_prob = models['conversion'].predict_proba(input_data)[0, 1]
    
    # Display conversion probability
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversion Probability")
        st.markdown(f"<h1 style='text-align: center; color: {'green' if conversion_prob > 0.5 else 'red'};'>{conversion_prob:.2%}</h1>", unsafe_allow_html=True)
        
        # Gauge chart for conversion probability
        fig, ax = plt.subplots(figsize=(4, 0.3))
        ax.barh(0, conversion_prob, color='green', height=0.2)
        ax.barh(0, 1-conversion_prob, left=conversion_prob, color='lightgray', height=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        st.pyplot(fig)
        
        # Interpretation
        if conversion_prob > 0.7:
            st.success("High conversion probability! This impression is very likely to lead to a conversion.")
        elif conversion_prob > 0.3:
            st.info("Moderate conversion probability. This impression has a reasonable chance of leading to a conversion.")
        else:
            st.error("Low conversion probability. This impression is unlikely to lead to a conversion.")
    
    # Get attribution probability if conversion is likely and attribution model exists
    if 'attribution' in models and conversion_prob > 0.3:
        attribution_prob = models['attribution'].predict_proba(input_data)[0, 1]
        
        with col2:
            st.subheader("Attribution Probability")
            st.markdown(f"<h1 style='text-align: center; color: {'green' if attribution_prob > 0.5 else 'orange'};'>{attribution_prob:.2%}</h1>", unsafe_allow_html=True)
            
            # Gauge chart for attribution probability
            fig, ax = plt.subplots(figsize=(4, 0.3))
            ax.barh(0, attribution_prob, color='green', height=0.2)
            ax.barh(0, 1-attribution_prob, left=attribution_prob, color='lightgray', height=0.2)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            st.pyplot(fig)
            
            # Interpretation
            if attribution_prob > 0.7:
                st.success("High attribution probability! If conversion occurs, it's likely to be attributed to this impression.")
            elif attribution_prob > 0.3:
                st.info("Moderate attribution probability. This impression may receive credit for a conversion.")
            else:
                st.warning("Low attribution probability. Even if conversion occurs, this impression is unlikely to receive credit.")
    else:
        with col2:
            st.subheader("Attribution Probability")
            st.markdown("<h1 style='text-align: center; color: gray;'>N/A</h1>", unsafe_allow_html=True)
            st.info("Attribution probability is only calculated for impressions with moderate to high conversion probability.")
    
    # Bidding recommendation
    st.header("Bidding Recommendation")
    
    if conversion_prob > 0.7:
        if 'attribution' in models and attribution_prob > 0.5:
            st.success("📈 **Bid Higher**: High conversion probability with good attribution likelihood suggests increasing your bid for this type of impression.")
        else:
            st.info("⚖️ **Maintain Bid**: High conversion probability but uncertain attribution suggests maintaining your current bid.")
    elif conversion_prob > 0.3:
        st.info("⚖️ **Maintain Bid**: Moderate conversion probability suggests maintaining your current bid for this type of impression.")
    else:
        st.error("📉 **Bid Lower**: Low conversion probability suggests reducing your bid for this type of impression.")

    # Feature importance analysis
    st.header("What-If Analysis")
    
    # Create a function to simulate changes
    def simulate_change(feature, change_pct):
        new_data = input_data.copy()
        current_value = new_data[feature].values[0]
        
        # Don't modify categorical features or binary features
        if feature == 'campaign' or feature == 'click':
            return new_data, current_value, current_value
        
        # Calculate new value
        new_value = current_value * (1 + change_pct/100)
        new_data[feature] = new_value
        
        # Update dependent features
        if feature == 'cost':
            new_data['cost_per_click'] = new_value if click else 0
            new_data['cost_campaign_conv_interaction'] = new_value * campaign_conversion_rate
        
        return new_data, current_value, new_value
    
    # Select feature to modify
    feature_to_modify = st.selectbox(
        "Select feature to modify",
        options=[
            'cost', 
            'user_impression_count', 
            'user_click_rate', 
            'user_conversion_rate',
            'campaign_ctr', 
            'campaign_conversion_rate'
        ]
    )
    
    # Slider for percent change
    change_pct = st.slider(
        "Percent change",
        min_value=-50,
        max_value=100,
        value=20,
        step=5
    )
    
    # Simulate the change
    new_data, old_value, new_value = simulate_change(feature_to_modify, change_pct)
    
    # Get new predictions
    new_conversion_prob = models['conversion'].predict_proba(new_data)[0, 1]
    
    if 'attribution' in models and new_conversion_prob > 0.3:
        new_attribution_prob = models['attribution'].predict_proba(new_data)[0, 1]
    else:
        new_attribution_prob = None
    
    # Display the impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Impact on Conversion Probability")
        
        # Calculate change
        conv_change = new_conversion_prob - conversion_prob
        conv_change_pct = (conv_change / conversion_prob) * 100 if conversion_prob > 0 else 0
        
        # Display change
        st.metric(
            label=f"Conversion Probability ({feature_to_modify}: {old_value:.2f} → {new_value:.2f})",
            value=f"{new_conversion_prob:.2%}",
            delta=f"{conv_change_pct:.1f}%"
        )
    
    with col2:
        st.subheader("Impact on Attribution Probability")
        
        if new_attribution_prob is not None and 'attribution' in models:
            # Calculate change
            attr_change = new_attribution_prob - attribution_prob
            attr_change_pct = (attr_change / attribution_prob) * 100 if attribution_prob > 0 else 0
            
            # Display change
            st.metric(
                label=f"Attribution Probability ({feature_to_modify}: {old_value:.2f} → {new_value:.2f})",
                value=f"{new_attribution_prob:.2%}",
                delta=f"{attr_change_pct:.1f}%"
            )
        else:
            st.info("Attribution probability not available for this scenario.")
    
    # Sensitivity analysis
    st.subheader("Sensitivity Analysis")
    
    # Create range of values for the selected feature
    if feature_to_modify in ['cost', 'user_impression_count']:
        range_values = np.linspace(old_value * 0.5, old_value * 2, 10)
    else:  # For rate features, keep within 0-1
        min_val = max(0, old_value * 0.5)
        max_val = min(1, old_value * 2)
        range_values = np.linspace(min_val, max_val, 10)
    
    # Calculate conversion probabilities for each value
    conv_probs = []
    attr_probs = []
    
    for val in range_values:
        test_data = input_data.copy()
        test_data[feature_to_modify] = val
        
        # Update dependent features
        if feature_to_modify == 'cost':
            test_data['cost_per_click'] = val if click else 0
            test_data['cost_campaign_conv_interaction'] = val * campaign_conversion_rate
        
        # Get predictions
        test_conv_prob = models['conversion'].predict_proba(test_data)[0, 1]
        conv_probs.append(test_conv_prob)
        
        if 'attribution' in models and test_conv_prob > 0.3:
            test_attr_prob = models['attribution'].predict_proba(test_data)[0, 1]
        else:
            test_attr_prob = None
        
        attr_probs.append(test_attr_prob)
    
    # Plot conversion probability
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range_values, conv_probs, label='Conversion Probability')
    ax.set_title('Conversion Probability Sensitivity')
    ax.set_xlabel(feature_to_modify)
    ax.set_ylabel('Probability')
    ax.legend()
    st.pyplot(fig)
    
    # Plot attribution probability
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range_values, attr_probs, label='Attribution Probability')
    ax.set_title('Attribution Probability Sensitivity')
    ax.set_xlabel(feature_to_modify)
    ax.set_ylabel('Probability')
    ax.legend()
    st.pyplot(fig)
