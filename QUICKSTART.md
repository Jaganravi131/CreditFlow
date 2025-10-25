# CreditFlow - Quick Start Guide

## Live Demo Instructions

This guide will help you run the CreditFlow Conversion Attribution Simulator locally.

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/Jaganravi131/CreditFlow.git
cd CreditFlow
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Live Demo

#### Option 1: Using the start script (Linux/Mac)
```bash
./start_demo.sh
```

#### Option 2: Direct command (All platforms)
```bash
streamlit run conversion_simulator.py
```

The application will automatically open in your browser at `http://localhost:8501`

### What You Can Do in the Demo

The **Conversion Attribution Simulator** provides an interactive interface where you can:

1. **Adjust Campaign Parameters**
   - Select different campaign IDs
   - Modify click behavior (position, count, time since last click)
   - Set cost metrics

2. **Explore User Characteristics**
   - User impression count
   - Historical click rate
   - Historical conversion rate

3. **View Real-Time Predictions**
   - **Conversion Probability**: Likelihood that an ad impression leads to a conversion
   - **Attribution Probability**: Likelihood that a converting impression receives credit

4. **Get Bidding Recommendations**
   - The system provides actionable insights on whether to bid higher, maintain, or lower your bid
   - Based on both conversion and attribution probabilities

5. **Perform What-If Analysis**
   - Select any feature to modify
   - See how changes impact conversion and attribution probabilities
   - View sensitivity analysis charts

### Demo Features Explained

#### Conversion Probability
- **High (>70%)**: Strong likelihood of conversion - consider increasing bids
- **Moderate (30-70%)**: Reasonable chance of conversion - maintain current bids
- **Low (<30%)**: Unlikely to convert - consider reducing bids

#### Attribution Probability
- Only calculated for impressions with moderate to high conversion probability
- Indicates whether the impression will receive credit if conversion occurs
- Important for understanding true campaign effectiveness

#### Sensitivity Analysis
- Interactive charts showing how feature changes affect predictions
- Helps understand which factors are most important for conversions and attribution

### Troubleshooting

**Issue**: `ModuleNotFoundError` when running the simulator
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Port 8501 already in use
**Solution**: Either stop the existing Streamlit process or specify a different port:
```bash
streamlit run conversion_simulator.py --server.port=8502
```

**Issue**: Models not found
**Solution**: The pre-trained models are included in the `models/` directory. Ensure you're running the command from the project root.

### Deployment Options

#### Deploy to Streamlit Community Cloud
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy! Streamlit will use the `requirements.txt` automatically

#### Deploy to Heroku
1. Create a Heroku account
2. Install the Heroku CLI
3. Run:
```bash
heroku create your-app-name
git push heroku main
```

The `Procfile` is already configured for Heroku deployment.

### Project Structure

```
CreditFlow/
├── conversion_simulator.py      # Main Streamlit app (LIVE DEMO)
├── attribution_models.py        # Model training script
├── requirements.txt             # Python dependencies
├── start_demo.sh               # Quick start script
├── Procfile                    # Heroku deployment config
├── models/                     # Pre-trained ML models
│   ├── conversion_model.pkl
│   ├── attribution_model.pkl
│   └── feature_names.pkl
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
└── processed_attribution_data.csv  # Sample data for demo
```

### Support

For issues or questions:
- Open an issue on GitHub
- Contact: Jagan babu.R or Mrunal Waghmare

### Next Steps

- Explore the interactive simulator
- Try different parameter combinations
- Review the visualizations and recommendations
- Consider integrating the API with your bidding system

---

**Note**: This demo uses pre-trained models on a sample of the Criteo Attribution Dataset. The models can predict conversion and attribution probabilities for new ad impressions in real-time.
