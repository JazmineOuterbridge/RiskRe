# 🔥 ReRisk AI: Catastrophe Re-Pricing Tool

A comprehensive Python-based Streamlit dashboard for reinsurance catastrophe risk forecasting. This application simulates treaty pricing for natural catastrophes like hurricanes and earthquakes using machine learning models on historical claims data.

## Features

- **Interactive Risk Assessment**: Real-time portfolio analysis with sliders for exposure, attachment points, and regional settings
- **Machine Learning Models**: XGBoost classification and Random Forest regression for risk prediction
- **Monte Carlo Simulations**: 1000-run simulations for VaR and Expected Shortfall calculations
- **Geographic Visualization**: Interactive Folium maps with risk heatmaps
- **SHAP Explainability**: Feature importance analysis for model transparency
- **Risk Indicators**: Emoji-based risk assessment (🔥 for high risk, 😎 for low risk)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   # Create project directory
   mkdir rerisk-ai
   cd rerisk-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data files are in place**
   ```
   data/
   ├── insurance2.csv    # Insurance claims data
   └── hurricanes.csv    # Hurricane exposure data
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - The dashboard will load with sample data if CSV files are missing

## Usage

### Dashboard Controls

**Sidebar Configuration:**
- **Attachment Point**: Minimum loss threshold for reinsurance coverage ($1M - $100M)
- **Portfolio Size**: Total portfolio value ($10M - $1000M)
- **Primary Region**: Geographic region (North, South, East, West)
- **Climate Amplification**: What-if scenario for increased hurricane intensity (0-50%)
- **Cede Rate**: Percentage of losses ceded to reinsurer (10-100%)

### Key Metrics

The dashboard displays:
- **Predicted Annual Loss**: Expected reinsurance losses
- **VaR (99%)**: Value at Risk at 99% confidence level
- **Expected Shortfall**: Average loss above VaR threshold
- **Suggested Premium**: Optimal reinsurance premium with loading factors
- **Risk Indicator**: Visual risk assessment (🔥/😎)

### Visualizations

1. **Geographic Risk Heatmap**: Interactive map showing regional risk exposure
2. **Loss Distribution**: Monte Carlo simulation results
3. **SHAP Feature Importance**: Model explainability charts
4. **Historical vs Predicted**: Comparison of actual vs model predictions

## Technical Architecture

### Data Pipeline
1. **Data Loading**: Insurance claims and hurricane exposure data
2. **Feature Engineering**: Risk scores, ceded losses, cat exposure multipliers
3. **Data Preprocessing**: Missing value handling, categorical encoding
4. **Train/Test Split**: 80/20 split for model validation

### Machine Learning Models
- **XGBoost Classifier**: Binary classification for high-risk prediction
- **Random Forest Regressor**: Loss amount prediction
- **SHAP Integration**: Model explainability and feature importance

### Risk Calculations
- **Monte Carlo Simulation**: 1000 iterations with normal distribution
- **VaR Calculation**: 99th percentile of simulated losses
- **Expected Shortfall**: Mean of losses above VaR threshold
- **Premium Calculation**: Loss prediction × (1 + expense ratio + profit margin)

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment (Vercel)

1. **Create vercel.json**:
   ```json
   {
     "builds": [
       {
         "src": "app.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "app.py"
       }
     ]
   }
   ```

2. **Deploy**:
   ```bash
   vercel --prod
   ```

### Environment Variables
- Set any required API keys or database connections
- Configure data source paths if needed

## Data Requirements

### Input Data Format

**insurance2.csv** (Insurance claims data):
```csv
age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,south,16884.924
...
```

**hurricanes.csv** (Hurricane exposure data):
```csv
region,cat_exposure,hurricane_frequency,avg_intensity
north,1.0,0.1,2.5
south,1.5,0.8,3.8
...
```

## Performance Optimizations

- **Caching**: `@st.cache_data` decorators for data loading and model training
- **Efficient Processing**: Vectorized operations with NumPy and Pandas
- **Memory Management**: Optimized data structures and minimal data copying

## Edge Cases Handled

- **Zero Attachment Points**: Minimum threshold validation
- **Extreme Values**: Capped predictions and outlier handling
- **Missing Data**: Graceful fallback to synthetic data
- **Empty Datasets**: Error handling with informative messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the documentation above
2. Review the code comments in `app.py`
3. Open an issue with detailed error messages

## Future Enhancements

- Real-time data integration
- Additional catastrophe types (earthquakes, floods)
- Advanced ML models (neural networks, ensemble methods)
- API endpoints for external integrations
- Multi-currency support
- Advanced scenario analysis

---

**ReRisk AI** - Powered by Machine Learning for Catastrophe Risk Assessment 🔥
