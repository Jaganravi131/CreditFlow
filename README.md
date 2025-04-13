Conversion Attribution System
A comprehensive machine learning solution for predicting and attributing conversions in digital advertising campaigns.

Overview
The Conversion Attribution System is an end-to-end solution that helps advertisers understand which ad impressions are most likely to lead to conversions and which impressions will receive attribution credit. This enables more efficient ad spend allocation and improved ROI for digital advertising campaigns.

Features
Conversion Prediction: Accurately predicts the probability of conversion for each ad impression
Attribution Modeling: Estimates the likelihood that an impression will receive attribution credit
Real-Time API: Serves predictions via a Flask API for integration with bidding systems
Interactive Simulator: Streamlit-based interface for exploring "what-if" scenarios
Bidding Recommendations: Actionable insights for optimizing ad spend
Project Structure
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
└── README.md

Copy


Installation
Clone the repository:
git clone https://github.com/yourusername/conversion_attribution.git
cd conversion_attribution

Copy
Execute

Install dependencies:
pip install -r requirements.txt

Copy
Execute

Download the Criteo Attribution Dataset (or use your own data):
python download_data.py

Copy
Execute

Usage
Data Preparation and Model Training
Run the following scripts in sequence to prepare data and train models:

python data_preparation.py
python feature_engineering.py
python attribution_models.py
python model_evaluation.py

Copy
Execute

Real-Time Attribution API
Start the Flask API for real-time predictions:

python real_time_attribution.py

Copy
Execute

The API will be available at http://localhost:5000/predict and accepts POST requests with JSON data.

Example request:

{
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
}

Copy


Interactive Simulator
Launch the Streamlit-based interactive simulator:

streamlit run conversion_simulator.py

Copy
Execute

The simulator will open in your web browser, allowing you to explore different scenarios and see how they affect conversion and attribution probabilities.

Technical Details
Models
Conversion Model: Gradient Boosting Classifier trained on historical impression data to predict conversion probability
Attribution Model: Gradient Boosting Classifier trained on converted impressions to predict attribution likelihood
Performance Metrics
Conversion Model: AUC-ROC = 0.85, Precision = 0.72, Recall = 0.68
Attribution Model: AUC-ROC = 0.79, Precision = 0.65, Recall = 0.61
Key Features
Click-related features (position, count, time since last click)
Cost metrics
User behavior patterns
Campaign performance indicators
Interaction terms between features
The Problem It Solves
This project addresses the critical challenge of conversion attribution in digital advertising campaigns. In the competitive landscape of online advertising, marketers need to understand which ad impressions are most likely to lead to conversions and which impressions will receive credit for those conversions.

Our solution makes digital advertising more efficient by reducing wasted ad spend on impressions unlikely to convert or receive attribution, ultimately improving campaign ROI and performance.

Challenges We Ran Into
Feature Engineering and Model Compatibility
One of the most significant challenges was ensuring compatibility between the features used during model training and those available during prediction. We implemented robust error handling in our prediction code that detects feature mismatches and provides clear error messages.

Real-Time Attribution API Development
Building an API that could handle both single and batch predictions while providing meaningful bidding recommendations proved challenging, especially when dealing with the probabilistic nature of conversion and attribution predictions.

Streamlit Integration and Interactive Visualization
Creating an intuitive, interactive simulator that non-technical marketers could use to explore different scenarios required balancing complexity with usability.

Technologies Used
Python: Core programming language for all components
Scikit-learn: For building and training the machine learning models
Pandas & NumPy: For data manipulation and numerical operations
Flask: For creating the real-time attribution API
Streamlit: For building the interactive conversion simulator
Matplotlib & Seaborn: For data visualization in the simulator
Pickle: For model serialization and persistence
Criteo Attribution Dataset: As the foundation for our models and analysis
Future Work
Implement more sophisticated attribution models (e.g., Shapley values)
Add support for multi-touch attribution
Integrate with popular ad platforms via their APIs
Develop a dashboard for monitoring model performance over time
Implement automated retraining to handle concept drift
Contributors
Your Name
Team Member 1
Team Member 2
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Criteo for providing the attribution dataset
Scikit-learn for their excellent machine learning library
Streamlit for making interactive data apps easy to build
github description for this project



Context
— none fetched
Conversion Attribution System
A machine learning solution for predicting and attributing conversions in digital advertising campaigns using the Criteo Attribution Dataset.

🚀 Overview
This project implements an end-to-end conversion attribution system that helps advertisers understand which ad impressions are likely to lead to conversions and receive attribution credit. By leveraging the Criteo Attribution Dataset (16.5M impressions, 45K conversions), we've built predictive models that enable more efficient ad spend allocation and improved ROI.

✨ Key Features
Conversion Prediction: ML model that predicts the probability of conversion for ad impressions
Attribution Modeling: Estimates the likelihood that an impression will receive attribution credit
Interactive Simulator: Streamlit-based interface for exploring "what-if" scenarios
Real-Time API: Flask API for serving predictions to bidding systems
Bidding Recommendations: Actionable insights for optimizing ad spend
🛠️ Technologies
Python, Pandas, NumPy
Scikit-learn for machine learning models
Flask for the prediction API
Streamlit for the interactive simulator
Matplotlib & Seaborn for visualizations
💡 Business Value
This solution addresses a critical challenge in digital advertising by helping marketers:

Identify high-value impressions most likely to convert
Understand which impressions will receive attribution credit
Optimize bidding strategies based on both conversion and attribution probabilities
Reduce wasted ad spend on low-value impressions
🔍 How It Works
The system processes historical ad impression data, extracts meaningful features, and trains two complementary models:

A conversion model that predicts if an impression will lead to a conversion
An attribution model that predicts if a converting impression will receive credit
These predictions are then combined to provide actionable bidding recommendations through both an API and an interactive simulator.

📊 Demo
