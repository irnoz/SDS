from sensor_data_processor import SensorDataProcessor

def main():
    processor = SensorDataProcessor(app_name="Sensor Data Cleaning", partitions=400)
    df = processor.read_data("data/in/unreliable_sensor.csv")
    cleaned_df = processor.preprocess_data(df)
    processor.save_results(cleaned_df, "data/out/cleaned_sensor_data")
    processor.close()

if __name__ == "__main__":
    main()
