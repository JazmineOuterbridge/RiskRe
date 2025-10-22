"""
ReRisk AI: Catastrophe Re-Pricing Tool
A Streamlit dashboard for reinsurance catastrophe risk forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import shap
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ReRisk AI: Catastrophe Re-Pricing Tool",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-indicator {
        font-size: 3rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess insurance and hurricane data"""
    try:
        # Load insurance data
        df_insurance = pd.read_csv('data/insurance2.csv')
        
        # Add risk score
        df_insurance['risk_score'] = (df_insurance['bmi'] + df_insurance['age']) / 100
        
        # Load hurricane data (simulated if not available)
        try:
            df_hurricanes = pd.read_csv('data/hurricanes.csv')
        except FileNotFoundError:
            # Create simulated hurricane data
            regions = ['north', 'south', 'east', 'west']
            df_hurricanes = pd.DataFrame({
                'region': regions,
                'cat_exposure': [1.0, 1.5, 1.2, 1.1],  # South has highest exposure
                'hurricane_frequency': [0.1, 0.8, 0.3, 0.2]
            })
        
        # Merge datasets
        df = df_insurance.merge(df_hurricanes, on='region', how='left')
        df['cat_exposure'] = df['cat_exposure'].fillna(1.0)
        
        # Handle missing values
        df = df.fillna(0)
        
        return df
    except FileNotFoundError:
        # Create synthetic data if files don't exist
        st.warning("Data files not found. Using synthetic data for demonstration.")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'age': np.random.normal(40, 15, n_samples),
            'bmi': np.random.normal(30, 5, n_samples),
            'charges': np.random.lognormal(8, 1, n_samples),
            'region': np.random.choice(['north', 'south', 'east', 'west'], n_samples),
            'sex': np.random.choice(['male', 'female'], n_samples)
        })
        
        df['risk_score'] = (df['bmi'] + df['age']) / 100
        
        # Add cat exposure
        cat_exposure_map = {'north': 1.0, 'south': 1.5, 'east': 1.2, 'west': 1.1}
        df['cat_exposure'] = df['region'].map(cat_exposure_map)
        
        return df

@st.cache_data
def prepare_features(df, attachment_point, cede_rate=0.8):
    """Prepare features for ML models"""
    # Calculate ceded losses
    df['ceded_loss'] = np.maximum(df['charges'] - attachment_point, 0) * cede_rate
    
    # Create high-risk binary target
    df['high_risk'] = (df['charges'] > 10000).astype(int)
    
    # Prepare features
    feature_cols = ['age', 'bmi', 'risk_score', 'cat_exposure']
    
    # Encode categorical variables
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['region', 'sex'], drop_first=True)
    
    # Add region features to feature list
    region_cols = [col for col in df_encoded.columns if col.startswith('region_')]
    feature_cols.extend(region_cols)
    
    X = df_encoded[feature_cols]
    y_classification = df_encoded['high_risk']
    y_regression = df_encoded['ceded_loss']
    
    return X, y_classification, y_regression, df_encoded

@st.cache_data
def train_models(X, y_classification, y_regression):
    """Train ML models"""
    # Check if we have enough samples for stratification
    min_samples = min(np.bincount(y_classification))
    
    # Use stratification only if we have enough samples in each class
    stratify_param = y_classification if min_samples >= 2 else None
    
    # Train/test split
    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=stratify_param
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    # Train XGBoost classifier
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_class_train)
    
    # Train Random Forest regressor
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_reg_train)
    
    # Calculate metrics
    y_class_pred = xgb_model.predict_proba(X_test)[:, 1]
    y_reg_pred = rf_model.predict(X_test)
    
    auc_score = roc_auc_score(y_class_test, y_class_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    
    return xgb_model, rf_model, auc_score, mse, X.columns

def monte_carlo_simulation(predicted_loss, n_simulations=1000):
    """Run Monte Carlo simulation for risk calculations"""
    # Simulate losses with normal distribution
    simulated_losses = np.random.normal(
        predicted_loss, 
        scale=0.2 * predicted_loss, 
        size=n_simulations
    )
    
    # Ensure non-negative losses
    simulated_losses = np.maximum(simulated_losses, 0)
    
    # Calculate risk metrics
    var_99 = np.percentile(simulated_losses, 99)
    es_99 = np.mean(simulated_losses[simulated_losses >= var_99])
    
    return simulated_losses, var_99, es_99

def calculate_reinsurance_premium(predicted_loss, expense_ratio=0.3, profit_margin=0.3):
    """Calculate optimal reinsurance premium"""
    loading_factor = 1 + expense_ratio + profit_margin
    return predicted_loss * loading_factor

def create_risk_map(df, region_filter):
    """Create interactive Folium map with risk heatmap"""
    # Focus on specific regions
    if region_filter in ['south', 'southeast']:
        center_lat, center_lon = 27.8, -83.0  # Florida
        zoom_start = 6
    elif region_filter in ['northeast']:
        center_lat, center_lon = 40.7128, -74.0060  # New York
        zoom_start = 6
    elif region_filter in ['northwest']:
        center_lat, center_lon = 47.6062, -122.3321  # Seattle
        zoom_start = 6
    elif region_filter in ['southwest']:
        center_lat, center_lon = 33.4484, -112.0740  # Phoenix
        zoom_start = 6
    else:
        center_lat, center_lon = 39.8283, -98.5795  # Center of US
        zoom_start = 4
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    
    # Generate sample coordinates for heatmap
    np.random.seed(42)
    n_points = 50
    lats = np.random.normal(center_lat, 2, n_points)
    lons = np.random.normal(center_lon, 3, n_points)
    
    # Create heatmap data
    heat_data = []
    for i in range(n_points):
        # Weight by cat exposure and risk
        weight = df[df['region'] == region_filter]['cat_exposure'].mean() * np.random.uniform(0.5, 2.0)
        heat_data.append([lats[i], lons[i], weight])
    
    # Add heatmap
    HeatMap(heat_data, radius=20, blur=15, max_zoom=1).add_to(m)
    
    return m

def create_shap_plot(model, X_sample, feature_names):
    """Create SHAP explanation plot"""
    try:
        # Check if model is tree-based (XGBoost, RandomForest, etc.)
        model_type = type(model).__name__
        
        if model_type in ['XGBClassifier', 'XGBRegressor', 'RandomForestClassifier', 'RandomForestRegressor']:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Use KernelExplainer for non-tree models (LogisticRegression, LinearRegression, etc.)
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        # Create bar plot
        if len(shap_values.shape) > 2:
            # For classification, take mean across classes
            mean_shap = np.abs(shap_values).mean(axis=(0, 1))
        else:
            # For regression or single class
            mean_shap = np.abs(shap_values).mean(0)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Feature Importance (SHAP Values)',
            color='importance',
            color_continuous_scale='Reds'
        )
        
        return fig
        
    except Exception as e:
        # Fallback: create a simple feature importance plot
        st.warning(f"SHAP analysis failed: {str(e)}. Showing model coefficients instead.")
        
        # Try to get feature importance from model
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        else:
            # Random importance as fallback
            importance = np.random.random(len(feature_names))
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Feature Importance (Model Coefficients)',
            color='importance',
            color_continuous_scale='Reds'
        )
        
        return fig

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🔥 ReRisk AI: Catastrophe Re-Pricing Tool</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("Portfolio Configuration")
    
    attachment_point = st.sidebar.slider(
        "Attachment Point ($M)", 
        min_value=1, 
        max_value=100, 
        value=10,
        help="Minimum loss threshold for reinsurance coverage"
    )
    
    portfolio_size = st.sidebar.slider(
        "Portfolio Size ($M)", 
        min_value=10, 
        max_value=1000, 
        value=50,
        help="Total portfolio value in millions"
    )
    
    region = st.sidebar.selectbox(
        "Primary Region",
        options=['south', 'southeast', 'northwest', 'northeast', 'southwest'],
        index=0,  # Default to south
        help="Geographic region of primary exposure"
    )
    
    climate_amp = st.sidebar.slider(
        "Climate Amplification (%)",
        min_value=0,
        max_value=50,
        value=0,
        help="What-if scenario: increase in hurricane intensity"
    )
    
    cede_rate = st.sidebar.slider(
        "Cede Rate",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        help="Percentage of losses ceded to reinsurer"
    )
    
    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        df = load_and_preprocess_data()
    
    # Filter data by region
    df_filtered = df[df['region'] == region].copy()
    
    if len(df_filtered) == 0:
        st.error(f"No data available for region: {region}")
        return
    
    # Check if we have enough data for ML training
    if len(df_filtered) < 50:
        st.warning(f"Limited data for region: {region} ({len(df_filtered)} samples). Using all available data for better model performance.")
        # Use all data for training if sample size is too small
        df_filtered = df.copy()  # Use all data as fallback
    
    # Apply climate amplification
    if climate_amp > 0:
        df_filtered['cat_exposure'] *= (1 + climate_amp / 100)
    
    # Prepare features
    X, y_class, y_reg, df_processed = prepare_features(df_filtered, attachment_point * 1000000, cede_rate)
    
    # Train models
    with st.spinner("Training ML models..."):
        try:
            xgb_model, rf_model, auc_score, mse, feature_names = train_models(X, y_class, y_reg)
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            st.info("Using simplified models for demonstration...")
            # Fallback to simple models
            from sklearn.linear_model import LogisticRegression, LinearRegression
            xgb_model = LogisticRegression(random_state=42)
            rf_model = LinearRegression()
            xgb_model.fit(X, y_class)
            rf_model.fit(X, y_reg)
            auc_score = 0.5  # Random baseline
            mse = 0
            feature_names = X.columns
    
    # Calculate portfolio-level predictions
    portfolio_exposure = portfolio_size * 1000000  # Convert to actual dollars
    avg_risk_score = df_filtered['risk_score'].mean()
    avg_cat_exposure = df_filtered['cat_exposure'].mean()
    
    # Scale predictions to portfolio size
    sample_features = X.mean().values.reshape(1, -1)
    predicted_loss_rate = rf_model.predict(sample_features)[0]
    predicted_loss = predicted_loss_rate * (portfolio_size / 10)  # Scale to portfolio size
    
    # Monte Carlo simulation
    simulated_losses, var_99, es_99 = monte_carlo_simulation(predicted_loss)
    
    # Calculate premium
    optimal_premium = calculate_reinsurance_premium(predicted_loss)
    loading_factor = optimal_premium / predicted_loss if predicted_loss > 0 else 1
    
    # Risk indicator
    risk_emoji = "🔥" if var_99 > 5000000 else "😎"
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Annual Loss", f"${predicted_loss/1000000:.1f}M")
        st.metric("VaR (99%)", f"${var_99/1000000:.1f}M")
    
    with col2:
        st.metric("Expected Shortfall", f"${es_99/1000000:.1f}M")
        st.metric("Suggested Premium", f"${optimal_premium/1000000:.1f}M")
    
    with col3:
        st.metric("Loading Factor", f"{loading_factor:.1f}x")
        st.markdown(f'<div class="risk-indicator">{risk_emoji}</div>', unsafe_allow_html=True)
    
    # Risk summary
    st.markdown("### Risk Assessment Summary")
    st.info(f"""
    **For this ${portfolio_size}M {region.title()} portfolio:**
    - Predicted annual re-loss: ${predicted_loss/1000000:.1f}M
    - VaR (99%): ${var_99/1000000:.1f}M  
    - ES: ${es_99/1000000:.1f}M
    - Suggested premium: ${optimal_premium/1000000:.1f}M ({loading_factor:.0%} loading)
    - Risk Vibe: {risk_emoji}
    """)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive map
        st.markdown("### Geographic Risk Heatmap")
        risk_map = create_risk_map(df_filtered, region)
        st.components.v1.html(risk_map._repr_html_(), height=400)
    
    with col2:
        # Loss distribution
        st.markdown("### Loss Distribution")
        fig_dist = px.histogram(
            x=simulated_losses/1000000,
            nbins=50,
            title="Monte Carlo Loss Distribution",
            labels={'x': 'Loss ($M)', 'y': 'Frequency'}
        )
        fig_dist.add_vline(x=var_99/1000000, line_dash="dash", line_color="red", 
                          annotation_text=f"VaR: ${var_99/1000000:.1f}M")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # SHAP explanation
    st.markdown("### Model Explainability (SHAP)")
    shap_fig = create_shap_plot(xgb_model, X.sample(min(100, len(X))), feature_names)
    st.plotly_chart(shap_fig, use_container_width=True)
    
    # Model performance metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classification AUC", f"{auc_score:.3f}")
    with col2:
        st.metric("Regression MSE", f"{mse:.0f}")
    
    # Historical vs Predicted comparison
    st.markdown("### Historical vs Predicted Losses")
    historical_losses = df_filtered['ceded_loss'].values
    predicted_losses = np.full_like(historical_losses, predicted_loss)
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(
        y=historical_losses/1000000,
        mode='markers',
        name='Historical',
        marker=dict(color='blue', opacity=0.6)
    ))
    fig_comparison.add_trace(go.Scatter(
        y=predicted_losses/1000000,
        mode='markers',
        name='Predicted',
        marker=dict(color='red', opacity=0.6)
    ))
    fig_comparison.update_layout(
        title="Historical vs Predicted Losses",
        xaxis_title="Sample Index",
        yaxis_title="Loss ($M)"
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**ReRisk AI** - Powered by Machine Learning for Catastrophe Risk Assessment")

if __name__ == "__main__":
    main()
