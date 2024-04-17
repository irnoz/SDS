import unittest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import timedelta
from src.FPcode import fill_missing_timestamps, remove_outliers, fill_outliers

def read_data(spark, data, schema=["ts", "value"]):
    return spark.createDataFrame(data, schema).withColumn("ts", col("ts").cast("timestamp"))

def save_results(df, output_path):
    # Not saving to file system during unit testing, this is a placeholder
    pass

def process_data(df, spark, fill_timestamps, remove_outliers, fill_outliers):
    df = fill_timestamps(df, spark)
    df = remove_outliers(df)
    df = fill_outliers(df)
    return df

class TestSensorDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize a Spark session for the testing environment
        cls.spark = SparkSession.builder \
            .appName("Test Sensor Data Cleaning") \
            .master("local[2]") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark session after all tests have run
        cls.spark.stop()

    def test_zeros_in_data(self):
        data = [
            ("2021-01-01 12:00:00", 0.0),
            ("2021-01-01 12:00:15", 1.5),
            ("2021-01-01 12:00:30", 2.5),
            ("2021-01-01 12:00:45", 4.5),
            ("2021-01-01 12:01:00", 2.5),
            ("2021-01-01 12:01:15", 3.5),
            ("2021-01-01 12:01:30", 2.5),
            ("2021-01-01 12:01:45", 0.0),
            ("2021-01-01 12:02:00", -1.0)
        ]
        df = read_data(self.spark, data)

        # Process the data
        processed_df = process_data(df, self.spark, fill_missing_timestamps, remove_outliers, fill_outliers)

        # Test that no values are zero after processing
        self.assertTrue(processed_df.filter(col("value") == 0).count() == 0)

    def test_missing_timestamps(self):
        data = [
            ("2021-01-01 12:00:00", 2.0),
            ("2021-01-01 12:00:45", 3.0)
        ]
        df = read_data(self.spark, data)

        processed_df = fill_missing_timestamps(df, self.spark)

        # Check that the count of timestamps now includes filled timestamps
        self.assertEqual(processed_df.count(), 4)

    def test_outliers_in_data(self):
        data = [
            ("2021-01-01 12:00:00", 1.23),
            ("2021-01-01 12:00:15", 1.5),
            ("2021-01-01 12:00:30", 2.0),
            ("2021-01-01 12:00:45", 100.0),
            ("2021-01-01 12:01:00", 2.5),
            ("2021-01-01 12:01:15", 3.5),
            ("2021-01-01 12:01:30", 3.7),
            ("2021-01-01 12:01:45", 222.5),
            ("2021-01-01 12:02:00", 4.0)
        ]
        df = read_data(self.spark, data)

        processed_df = process_data(df, self.spark, fill_missing_timestamps, remove_outliers, fill_outliers)
        
        # Ensure no values are outside the specified range after processing
        self.assertTrue(all(row['value'] >= -10 and row['value'] <= 10 for row in processed_df.collect()))

if __name__ == "__main__":
    unittest.main()
