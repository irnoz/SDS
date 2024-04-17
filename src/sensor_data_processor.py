from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, min, max, last, year, month, dayofmonth
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType

class SensorDataProcessor:
    def __init__(self, app_name="Sensor Data Cleaning", partitions=400):
        self.spark = self.create_spark_session(app_name, partitions)

    @staticmethod
    def create_spark_session(app_name, partitions):
        """Create and return a Spark session."""
        return SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.shuffle.partitions", str(partitions)) \
            .getOrCreate()

    def read_data(self, file_path):
        """Read data from a CSV file into a DataFrame."""
        schema = StructType([
            StructField("ts", StringType(), True),
            StructField("value", FloatType(), True)
        ])
        return self.spark.read.csv(file_path, header=True, schema=schema, sep=';')

    def preprocess_data(self, df):
        """Preprocess data to handle missing values, zeros, and outliers."""
        df = df.withColumn("ts", to_timestamp("ts"))
        min_ts = df.select(min("ts")).collect()[0][0]
        max_ts = df.select(max("ts")).collect()[0][0]

        time_df = self.spark.range(
            start=min_ts.timestamp(),
            end=max_ts.timestamp() + 15,
            step=15
        ).select(col("id").cast("timestamp").alias("ts")).repartition("ts")

        df = time_df.join(df, "ts", "left_outer")

        window_spec = Window.partitionBy(year("ts"), month("ts"), dayofmonth("ts")).orderBy("ts") \
                            .rowsBetween(Window.unboundedPreceding, 0)
        df = df.withColumn("value", last("value", ignorenulls=True).over(window_spec)).filter(col("value").isNotNull())
        df = df.filter((col("value") >= -10) & (col("value") <= 10))

        return df

    def save_results(self, df, output_path):
        """Save the DataFrame results to the given path."""
        df.write.csv(output_path, mode="overwrite", header=True)

    def close(self):
        """Stop the Spark session."""
        self.spark.stop()
