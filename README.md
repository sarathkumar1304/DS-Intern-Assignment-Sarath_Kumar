# Smart Factory Energy Prediction Challenge

## Problem Overview

You've been hired as a data scientist for SmartManufacture Inc., a leading industrial automation company. The company has deployed an extensive sensor network throughout one of their client's manufacturing facilities to monitor environmental conditions and energy usage.

The client is concerned about the increasing energy costs associated with their manufacturing equipment. They want to implement a predictive system that can forecast equipment energy consumption based on various environmental factors and sensor readings from different zones of the factory.


---


# 🏠 Energy Consumption Prediction with ZenML

This project builds a machine learning pipeline to predict energy consumption using environmental and operational data from multiple zones. The pipeline is developed using **ZenML**, and incorporates preprocessing, model training, hyperparameter tuning with **Optuna**, model deployment using **MLflow**, and real-time inference.

---

## 📁 Project Structure

energy-consumption-predictor/
├── data/
│   ├── raw/                         # Raw input data files
│   └── processed/                   # Processed/cleaned datasets
├── pipelines/
│   ├── training_pipeline.py        # ZenML training pipeline definition
│   └── inference_pipeline.py 
├── src/                             # conatin core logic for this project 
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├──data_splitting.py
│   ├──outerlier_detection.py
│   ├──model_building.py
│   ├──model_evalaution.py
├── steps/                                  # ZenML inference pipeline definition
│   ├── data_ingestion_step.py              # Step to load the dataset
│   ├── data_preprocessor_step.py        # Step to clean and scale the data
│   ├── model_building_step.py            # Step to train the ML model
│   ├── model_evaluator_evaluation_step.py          # Step to evaluate model performance
│   ├── prediction_service_loader.py           # Step to deploy model with MLflow
│   ├── predictor.py                # Step for making predictions
│   └── dynamic_importer.py         # Step to import test input data dynamically
├── models/                         # (optional) Saved trained models
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis and testing
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py
├── run_deployment.py           # Entry point to run the ZenML pipeline



---

## 🔧 Features

- **Regression Model Selector**: Choose from Linear Regression, Random Forest, or Decision Tree.
- **StandardScaler Integration**: Data is scaled using Scikit-learn pipelines.
- **Hyperparameter Tuning with Optuna**: Tune models when enabled.
- **MLflow Integration**: Track experiments, visualize metrics, and deploy models.
- **ZenML Pipelines**: Modular and reproducible pipelines for model training and inference.

---

## 🧪 Input Features

| Feature                   | Description                       |
|---------------------------|-----------------------------------|
| equipment_energy_consumption | Energy used by equipment         |
| lighting_energy            | Energy used for lighting          |
| zone1_temperature - zone9_temperature | Temperatures of different zones |
| zone1_humidity - zone9_humidity       | Humidity levels in each zone   |
| outdoor_temperature        | Temperature outside               |
| atmospheric_pressure       | Atmospheric pressure              |
| outdoor_humidity           | Outdoor humidity                  |
| wind_speed                 | Wind speed                        |
| visibility_index           | Visibility index                  |
| dew_point                  | Dew point                         |
| random_variable1,2         | Experimental variables            |
| hour, dayofweek, month     | Time-based features               |
| is_weekend                 | Indicates weekend or not          |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/energy-consumption-predictor.git
cd energy-consumption-predictor
````

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Initialize ZenML

```bash
zenml init
zenml integration install mlflow -y
zenml experiment-tracker  register energy_compsumption_prediction_tracker --flavor=mlflow
zenml model-deployer register energy_model_deployer
zenml stack register -a d -o d -d energy_model_deployer -e energy_compsumption_prediction_tracker --set 
```

### 4. Run the Pipeline

```bash
python3 run_pipeline.py  # or your main execution file
```


### Model Deployment

```bash
python3 run_deployment.py
```

for deployment the model using mlflow and zenml and that return mlflow prediction service.

---

## 🔮 Predict with Deployed Model

You can send JSON input to the deployed MLflow endpoint using ZenML service:

```json
{
  "data": [
    [0.5, 0.2, 22.1, 40.5, 21.7, 38.2, 20.5, 37.9, 22.5, 42.1, 21.3, 41.2, 22.0, 43.0, 21.9, 44.1, 22.2, 39.4, 21.6, 40.6, 18.0, 1015.2, 55.0, 3.2, 10.0, 12.5, 0.3, 0.6, 14, 3, 6, 0]
  ]
}
```

---

## 📦 Requirements

* Python 3.12
* ZenML
* Scikit-learn
* Optuna
* MLflow
* Pandas, NumPy,


---

## 🧠 Author

**R. Sarath Kumar**
*Machine Learning Engineer | MLOps Enthusiast*

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.



