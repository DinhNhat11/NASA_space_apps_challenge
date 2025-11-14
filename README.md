# The NASA Space Apps Challenge 2025

 This is a 2-day global hackathon where participants from all backgrounds—coders, designers, scientists come together to tackle real-world challenges posed by NASA. The Winnipeg edition will be held at the University of Manitoba, providing teams with an opportunity to collaborate, innovate, and present solutions that could impact space exploration, Earth science, and technology.

One of the featured challenges, “A World Away: Hunting for Exoplanets with AI”, invites participants to create AI/ML models that analyze NASA’s open-source exoplanet datasets (Kepler, K2, TESS) to automatically identify exoplanets, moving beyond the manual identification process. Teams are encouraged to build models, preprocess data intelligently, and even provide user interfaces to interact with the predictions.

Full challenge details available at the [Winnipeg event page](https://www.spaceappschallenge.org/2025/local-events/winnipeg/).

# About the project

A web-based AI/ML application to automatically identify exoplanets using open-source NASA datasets (Kepler, K2, TESS). This FastAPI backend allows researchers and enthusiasts to upload datasets, explore statistics, train custom machine learning models, and make predictions—both single and batch—through a RESTful API or web interface.

## Features

* Upload datasets in CSV, Excel, or JSON format.
* Automatic data exploration with numeric/categorical summaries, missing value counts, histograms, and correlation matrices.
* Train custom machine learning models with configurable parameters.
* Supported ML algorithms: Random Forest, XGBoost, SVM, Neural Network (MLP).
* Monitor training jobs with progress updates.
* Make predictions on single data points or entire datasets.
* Includes pretrained models for Kepler, K2, and TESS missions for simulated predictions.
* Delete datasets and models to manage storage.
* Simple web interface served from the root path (`/`) with API documentation.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/exoplanet-detection-api.git
cd exoplanet-detection-api
```

2. Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000/`.

## Usage

### Run the Server

```bash
python main.py
```

or using uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Supported Machine Learning Algorithms

* **Random Forest:** Ensemble learning method for classification
* **XGBoost:** Gradient boosting framework
* **SVM:** Support Vector Machine for classification
* **Neural Network (MLP):** Multi-layer Perceptron for deep learning
