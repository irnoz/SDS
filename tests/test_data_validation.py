import unittest
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.functions import col
from src.sensor_data_processor import SensorDataProcessor

class SensorDataCleaningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start Spark session for testing."""
        cls.processor = SensorDataProcessor(app_name="Test Sensor Data Cleaning", partitions=2)

    @classmethod
    def tearDownClass(cls):
        """Stop Spark session after tests."""
        cls.processor.close()

    def test_missing_values_filled(self):
        """Test if missing values are filled correctly using last known values."""
        data = [
            ("2021-01-01 00:00:00", 5.0),
            ("2021-01-01 00:00:15", None),
            ("2021-01-01 00:00:30", None),
            ("2021-01-01 00:00:45", 5.5)
        ]
        schema = ["ts", "value"]
        df = self.processor.spark.createDataFrame(data, schema)
        df = df.withColumn("ts", col("ts").cast("timestamp"))
        
        processed_df = self.processor.preprocess_data(df)
        results = processed_df.filter(processed_df["ts"].cast("string") == "2021-01-01 00:00:30").collect()

        self.assertIsNotNone(results[0]["value"])
        self.assertEqual(results[0]["value"], 5.0)

    def test_outlier_removal(self):
        """Test if values outside the range -10 to 10 are properly filtered out."""
        data = [
            ("2021-01-01 00:00:00", -20.0),
            ("2021-01-01 00:00:15", 0.0),
            ("2021-01-01 00:00:30", 15.0),
            ("2021-01-01 00:00:45", 8.0)
        ]
        schema = ["ts", "value"]
        df = self.processor.spark.createDataFrame(data, schema)
        df = df.withColumn("ts", col("ts").cast("timestamp"))  # Ensure proper casting
        
        processed_df = self.processor.preprocess_data(df)
        results = processed_df.filter((col("value") < -10) | (col("value") > 10)).count()

        self.assertEqual(results, 0)  # Expecting no results since they should be filtered out

if __name__ == "__main__":
    unittest.main()
