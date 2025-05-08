import pandas as pd
from zenml import step

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Dynamically imports dummy data for testing the model."""
    data = {
        'equipment_energy_consumption': [150, 200, 180],
        'lighting_energy': [30, 40, 35],
        'zone1_temperature': [22.5, 23.0, 21.0],
        'zone1_humidity': [60, 65, 55],
        'zone2_temperature': [21.0, 22.5, 23.0],
        'zone2_humidity': [62, 58, 65],
        'zone3_temperature': [24.5, 25.0, 23.5],
        'zone3_humidity': [55, 60, 50],
        'zone4_temperature': [22.0, 23.5, 21.5],
        'zone4_humidity': [64, 63, 66],
        'zone5_temperature': [22.5, 23.0, 24.0],
        'zone5_humidity': [60, 59, 61],
        'zone6_temperature': [21.5, 22.0, 22.5],
        'zone6_humidity': [63, 61, 64],
        'zone7_temperature': [20.0, 19.5, 21.0],
        'zone7_humidity': [67, 66, 65],
        'zone8_temperature': [23.0, 23.5, 22.5],
        'zone8_humidity': [59, 62, 58],
        'zone9_temperature': [25.0, 24.0, 26.0],
        'zone9_humidity': [50, 55, 52],
        'outdoor_temperature': [18.0, 20.0, 19.5],
        'atmospheric_pressure': [1012, 1015, 1013],
        'outdoor_humidity': [45, 50, 47],
        'wind_speed': [5.0, 7.0, 6.0],
        'visibility_index': [10, 8, 9],
        'dew_point': [12.0, 14.5, 13.0],
        'random_variable1': [0.5, 0.7, 0.6],
        'random_variable2': [0.9, 1.2, 1.0],
        'hour': [14, 16, 18],
        'dayofweek': [1, 3, 5],  # 0 = Monday, 6 = Sunday
        'month': [6, 7, 8],
        'is_weekend': [0, 1, 0]  # 0 = Weekday, 1 = Weekend
    }
    
    df = pd.DataFrame(data)
    json_data = df.to_json(orient="split")
    return json_data
