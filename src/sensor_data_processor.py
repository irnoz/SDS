from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when
from pyspark.sql.window import Window
from datetime import timedelta
import pyspark.sql.functions as F

class SensorDataProcessor:
    def __init__(self, spark=None):
        self.spark = spark if spark else SparkSession.builder \
            .appName("Sensor Data Cleaning") \
            .getOrCreate()

    def read_data(self, file_path):
        df = self.spark.read.option("header", "true").option("delimiter", ";").csv(file_path)
        df = df.withColumn("ts", F.to_timestamp("ts", "yyyy-MM-dd HH:mm:ss"))
        df = df.withColumn("value", col("value").cast("float"))
        return df

    def save_results(self, df, file_path):
        df.write.mode("overwrite").option("header", True).csv(file_path)

    def process_data(self, df):
        df = self.fill_missing_timestamps(df)
        df = self.remove_outliers(df)
        df = self.fill_zeros(df)
        return df

    def fill_missing_timestamps(self, df):
        min_time = df.select(F.min("ts")).collect()[0][0]
        max_time = df.select(F.max("ts")).collect()[0][0]
        step = timedelta(seconds=15)
        time_range = self.spark.range(int(min_time.timestamp()), int(max_time.timestamp()) + 1, step.seconds)\
            .select(F.col("id").cast("timestamp").alias("full_ts"))
        df = time_range.join(df, time_range.full_ts == df.ts, "left_outer")\
            .select(time_range.full_ts.alias("ts"), "value")
        return df

    def remove_outliers(self, df):
        df = df.withColumn("value", when((col("value") < -10) | (col("value") > 10), 0.0).otherwise(col("value")))
        return df

    def fill_zeros(self, df):
        window = Window.orderBy("ts").rowsBetween(-3, 3)
        df = df.withColumn("value", when(col("value") == 0.0, avg(col("value")).over(window)).otherwise(col("value")))
        return df

    def close(self):
        self.spark.stop()
