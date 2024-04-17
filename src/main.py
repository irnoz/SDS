from sensor_data_processor import SensorDataProcessor

def main():
    input_path = "data/in/unreliable_sensor.csv"
    output_path = "data/out/cleaned_sensor_data"

    # Create an instance of SensorDataProcessor
    processor = SensorDataProcessor()

    # Read data from file
    df = processor.read_data(input_path)

    # Process the data
    processed_df = processor.process_data(df)

    # Save results to file
    processor.save_results(processed_df, output_path)

    # Close the Spark session
    processor.close()

if __name__ == "__main__":
    main()
