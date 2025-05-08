from steps.data_ingestion_step import data_ingestion_step
from steps.data_preprocessing_step import data_preprocessing_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitting_step import data_splitter_step
from steps.model_building_step import model_builder_step
from zenml import pipeline
from steps.model_evaluation_step import model_evaluation_step



@pipeline()
def ml_training_pipeline():
    raw_data = data_ingestion_step(path="data/data.csv")
    processed_data = data_preprocessing_step(df=raw_data)
    processed_data = outlier_detection_step(data=processed_data)
    X_train, X_test, y_train, y_test = data_splitter_step(df=processed_data, target_column="equipment_energy_consumption")
    model = model_builder_step(model_name = "xgboost",X_train=X_train, y_train=y_train,tune=True)
    metrics = model_evaluation_step(model=model, X_test=X_test, y_test=y_test)
    return model




