from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, expr, lit, min as min_, max as max_
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from datetime import timedelta, datetime

def create_spark_session():
    return SparkSession.builder \
        .appName("Sensor Data Cleaning") \
        .getOrCreate()

def read_data(spark, file_path):
    return spark.read.option("header", "true").option("delimiter", ";").csv(file_path) \
        .withColumn("ts", F.to_timestamp("ts", "yyyy-MM-dd HH:mm:ss")) \
        .withColumn("value", col("value").cast("float"))

def save_results(df, file_path):
    df.write.mode("overwrite").option("header", True).csv(file_path)

def calculate_time_range(min_time, max_time, step_seconds=15):
    timestamp_range = range(int(min_time.timestamp()), int(max_time.timestamp()) + 1, step_seconds)
    return [datetime.fromtimestamp(ts) for ts in timestamp_range]

def fill_missing_timestamps(df, spark):
    time_bounds = df.select(min_("ts"), max_("ts")).first()
    time_range = calculate_time_range(time_bounds[0], time_bounds[1], 15)
    full_time_df = spark.createDataFrame(time_range, "timestamp").toDF("full_ts")
    return full_time_df.join(df, full_time_df.full_ts == df.ts, "left_outer") \
        .select(full_time_df.full_ts.alias("ts"), "value")

def remove_outliers(df):
    window = Window.partitionBy(F.date_format('ts', 'yyyyMMdd')).orderBy("ts").rowsBetween(-3, 3)
    q1 = expr("percentile_approx(value, 0.25)").over(window)
    q3 = expr("percentile_approx(value, 0.75)").over(window)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df.withColumn("value", when((col("value") < lower_bound) | (col("value") > upper_bound), None).otherwise(col("value")))

def fill_outliers(df):
    window = Window.partitionBy(F.date_format('ts', 'yyyyMMdd')).orderBy('ts').rowsBetween(-3, 3)
    valid_avg = avg(when((col("value") != 0.0) & (col("value").isNotNull()), col("value"))).over(window)
    return df.withColumn("value", when((col("value") == 0.0) | (col("value").isNull()), valid_avg).otherwise(col("value")))

def process_data(spark, file_path, output_path):
    df = read_data(spark, file_path)
    df = fill_missing_timestamps(df, spark)
    df = remove_outliers(df)
    df = fill_outliers(df)
    save_results(df, output_path)

def close_spark(spark):
    spark.stop()

if __name__ == "__main__":
    spark_session = create_spark_session()
    process_data(spark_session, "data/in/unreliable_sensor.csv", "data/out/cleaned_sensor_dataFP")
    close_spark(spark_session)