CreditFlow (A machine learning solution for predicting and attributing conversions in digital advertising campaigns using the Criteo Attribution Dataset.)

Conversion Attribution System

📋 Table of Contents
Overview
Dataset
Features
Project Structure
Installation
Usage
Data Preparation
Model Training
Real-Time API
Interactive Simulator
Technical Details
Problem Statement
Challenges and Solutions
Future Work
Contributors
License
Acknowledgements

🚀 Overview

The Conversion Attribution System is a comprehensive machine learning solution designed to predict and attribute conversions in digital advertising campaigns. By analyzing impression data from the Criteo Attribution Dataset, this system helps advertisers understand which ad impressions are most likely to lead to conversions and which impressions will receive attribution credit, enabling more efficient ad spend allocation and improved ROI.

📊 Dataset
This project uses the Criteo Attribution Modeling for Bidding Dataset, which includes:

2.4GB of uncompressed data
16.5M impressions
45K conversions
700 campaigns
Each record in the dataset represents an ad impression with detailed information about:

Timestamp of the impression
Unique user identifier
Campaign identifier
Conversion status and timestamp
Attribution status
Click information (if clicked, position, number of clicks)
Cost metrics
✨ Features
Conversion Prediction: Accurately predicts the probability that an ad impression will lead to a conversion
Attribution Modeling: Estimates the likelihood that a converting impression will receive attribution credit
Real-Time API: Serves predictions via a Flask API for integration with bidding systems
Interactive Simulator: Streamlit-based interface for exploring "what-if" scenarios and understanding feature importance
Bidding Recommendations: Provides actionable insights for optimizing ad spend based on both conversion and attribution probabilities

📁 Project Structure
conversion_attribution/
├── data_preparation.py        # Data loading and preprocessing
├── feature_engineering.py     # Feature creation and transformation
├── attribution_models.py      # Model training and evaluation
├── model_evaluation.py        # Performance metrics and visualizations
├── real_time_attribution.py   # Flask API for serving predictions
├── conversion_simulator.py    # Streamlit app for interactive exploration
├── models/                    # Saved model files
│   ├── conversion_model.pkl
│   └── attribution_model.pkl
├── data/                      # Data directory
│   └── criteo_attribution_dataset.tsv.gz
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation


🔧 Installation
Clone the repository:
git clone https://github.com/yourusername/conversion_attribution.git
cd conversion_attribution



Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Download the Criteo Attribution Dataset:
mkdir -p data
curl -o data/criteo_attribution_dataset.tsv.gz https://s3-eu-west-1.amazonaws.com/attribution-dataset/criteo_attribution_dataset.zip
unzip data/criteo_attribution_dataset.zip -d data/


🖥️ Usage
Data Preparation
Process the raw Criteo dataset and prepare it for feature engineering:

python data_preparation.py

This script:

Loads the raw TSV data
Performs initial cleaning and preprocessing
Handles missing values
Saves the processed data for feature engineering
Model Training
Train the conversion and attribution models:

python feature_engineering.py
python attribution_models.py
python model_evaluation.py

These scripts:

Create advanced features from the processed data
Train the conversion prediction model
Train the attribution model
Evaluate model performance and generate visualizations
Save the trained models for later use
Real-Time API
Start the Flask API for real-time predictions:

python real_time_attribution.py

The API will be available at http://localhost:5000/ with the following endpoints:

POST /predict: Make a single prediction
POST /predict_batch: Make predictions for multiple impressions
GET /model_info: Get information about the loaded models
Example API request:

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "campaign": "1",
    "click": 1,
    "click_pos": 0,
    "click_nb": 1,
    "cost": 1.5,
    "time_since_last_click": 3600,
    "user_impression_count": 5,
    "user_click_rate": 0.2,
    "user_conversion_rate": 0.1,
    "campaign_ctr": 0.15,
    "campaign_conversion_rate": 0.05
  }'

Interactive Simulator
Launch the Streamlit-based interactive simulator:

streamlit run conversion_simulator.py

The simulator will open in your web browser (typically at http://localhost:8501), allowing you to:

Adjust campaign parameters
Modify click behavior
Set cost metrics
Explore user characteristics
See real-time predictions for conversion and attribution
Get bidding recommendations based on the predictions
🔬 Technical Details
Models
Conversion Model: Gradient Boosting Classifier trained to predict whether an impression will lead to a conversion

Features: campaign, click metrics, cost, user behavior, temporal patterns
Performance: AUC-ROC = 0.85, Precision = 0.72, Recall = 0.68
Attribution Model: Gradient Boosting Classifier trained to predict whether a converting impression will receive attribution credit

Features: campaign, click position, click count, cost, time since last click
Performance: AUC-ROC = 0.79, Precision = 0.65, Recall = 0.61
Key Features
The models leverage several types of features:

Click-related: Position, count, time since last click
Cost metrics: Impression cost, cost per click
User behavior: Historical click and conversion rates
Campaign performance: CTR, conversion rate
Temporal patterns: Hour of day, recency effects
Interaction terms: Combinations of important features
🎯 Problem Statement
This project addresses the critical challenge of conversion attribution in digital advertising campaigns. In the competitive landscape of online advertising, marketers need to:

Predict Conversions: Identify which impressions are most likely to lead to conversions
Understand Attribution: Determine which impressions will receive credit for conversions
Optimize Bidding: Adjust bid amounts based on both conversion probability and attribution likelihood
Improve ROI: Reduce wasted ad spend on impressions unlikely to convert or receive attribution
Our solution makes digital advertising more efficient by providing accurate predictions and actionable recommendations, ultimately improving campaign performance and return on investment.

🧩 Challenges and Solutions
Feature Engineering and Model Compatibility
Challenge: Ensuring compatibility between the features used during model training and those available during prediction.

Solution: Implemented robust error handling in our prediction code that detects feature mismatches and provides clear error messages. For the simulator, we added a fallback mechanism that creates a demonstration model on synthetic data when the original models fail.

Real-Time Attribution API Development
Challenge: Building an API that could handle both single and batch predictions while providing meaningful bidding recommendations.

Solution: Designed a flexible API architecture with separate endpoints for single and batch predictions. Implemented a sophisticated recommendation engine that considers both conversion and attribution probabilities.

Streamlit Integration and Interactive Visualization
Challenge: Creating an intuitive, interactive simulator that non-technical marketers could use to explore different scenarios.

Solution: Leveraged Streamlit's reactive programming model to create a responsive interface with sliders and selectors for key parameters. Implemented real-time visualization of prediction results and clear, actionable interpretations.

Environment and Dependency Management
Challenge: Ensuring the system worked consistently across different environments.

Solution: Created comprehensive installation instructions and implemented fallback mechanisms for finding model files and data in different locations. Added detailed error messages to help users troubleshoot common installation issues.

🔮 Future Work
Advanced Attribution Models: Implement multi-touch attribution and Shapley value-based attribution
Model Explainability: Add SHAP values and feature importance visualizations
Automated Retraining: Implement a pipeline for periodic model retraining to handle concept drift
A/B Testing Framework: Develop a system to test different attribution models in production
Integration with Ad Platforms: Create connectors for popular advertising platforms
Real-Time Dashboard: Build a comprehensive dashboard for monitoring model performance

👥 Contributors
Jagan babu.R
Mrunal Waghmare

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
Criteo Research for providing the attribution dataset
Eustache Diemert & Julien Meynet for their paper "Attribution Modeling Increases Efficiency of Bidding in Display Advertising"
Scikit-learn for their excellent machine learning library
Streamlit for making interactive data apps easy to build
Flask for the lightweight web framework

Note: This project was developed as part of [Synapses'25 hackathon]. For questions or support, please open an issue or contact the contributors directly.

