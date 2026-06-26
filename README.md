# Solar Power Prediction System

A web application for predicting solar power generation across various locations in Tamil Nadu, India, using machine learning models.

## Overview

This application uses trained machine learning models to predict solar power generation based on various environmental and weather parameters. It includes:

- A user-friendly frontend for inputting prediction parameters
- A Flask backend for handling predictions and serving model results
- Visualization of prediction results and model performance metrics

## Features

- Solar power generation prediction for 10 different locations in Tamil Nadu
- Input form for environmental parameters (temperature, humidity, wind speed, etc.)
- Visual display of prediction results
- Model performance metrics visualization
- Feature importance visualization

## Project Structure

```
.
├── app.py                 # Flask application 
├── static/                # Static files
│   ├── css/               # CSS stylesheets
│   ├── js/                # JavaScript files
│   └── images/            # Image assets
├── templates/             # HTML templates
│   ├── index.html         # Home page template
│   └── results.html       # Results page template
├── content/               # Data and model files
│   ├── data/              # CSV data files
│   ├── trained_models/    # Trained ML models
│   └── model_results/     # Model performance metrics
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd solar-power-prediction
   ```

2. Create a virtual environment and activate it:
   ```
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Select a location and enter the environmental parameters to get a prediction.

## Model Information

The application uses XGBoost regression models trained on historical weather and solar power generation data for various locations in Tamil Nadu. The models have been trained with optimized hyperparameters for each location.

## Technologies Used

- Backend: Python, Flask
- Frontend: HTML, CSS, JavaScript
- Data Processing: Pandas, NumPy
- Machine Learning: XGBoost, Scikit-learn, Joblib
- Visualization: Chart.js 