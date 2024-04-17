from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, expr
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from datetime import timedelta

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
        # self.save_results(df, "data/out/middle")
        df = self.remove_outliers(df)
        df = self.fill_outliers(df)
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
        window = Window.partitionBy(F.date_format('ts', 'yyyyMMdd')).orderBy("ts").rowsBetween(-3, 3)

        q1 = F.expr("percentile_approx(value, 0.25)").over(window)
        q3 = F.expr("percentile_approx(value, 0.75)").over(window)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df = df.withColumn("value",
                        when((col("value") < lower_bound) | (col("value") > upper_bound), None)
                        .otherwise(col("value")))

        return df
    
    # def remove_outliers(self, df):
    #     window_spec = Window.partitionBy(F.date_format('ts', 'yyyyMMdd')).orderBy("ts").rowsBetween(-5, 5)

    #     q1 = expr("percentile_approx(value, 0.25)").over(window_spec)
    #     q3 = expr("percentile_approx(value, 0.75)").over(window_spec)
    #     iqr = q3 - q1

    #     lower_bound = q1 - 1.5 * iqr
    #     upper_bound = q3 + 1.5 * iqr

    #     shifted_window = Window.partitionBy(F.date_format('ts', 'yyyyMMdd')).orderBy("ts").rowsBetween(-5, -1)
    #     next_shifted_window = Window.partitionBy(F.date_format('ts', 'yyyyMMdd')).orderBy("ts").rowsBetween(1, 5)
        
    #     prev_avg = avg(col("value")).over(shifted_window)
    #     next_avg = avg(col("value")).over(next_shifted_window)
    #     combined_avg = (prev_avg + next_avg) / 2

    #     df = df.withColumn("value",
    #                     when((col("value") < lower_bound) | (col("value") > upper_bound),
    #                             combined_avg)
    #                     .otherwise(col("value")))

    #     return df

    def fill_outliers(self, df):
        window = Window.partitionBy(F.date_format('ts', 'yyyyMMdd')).orderBy('ts').rowsBetween(-3, 3)

        valid_avg = avg(when((col("value") != 0.0) & (col("value").isNotNull()), col("value"))).over(window)

        df = df.withColumn("value",
                        when((col("value") == 0.0) | (col("value").isNull()), valid_avg)
                        .otherwise(col("value")))

        return df

    def close(self):
        self.spark.stop()


#2021-01-01 05:49:45;-787.8021241913912
