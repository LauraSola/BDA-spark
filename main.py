import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, expr, substring, col, date_format, sum

JDBC_JAR =  "./resources/postgresql-42.2.8.jar"
HADOOP_HOME = "./resources/hadoop_home"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

if(__name__== "__main__"):
  os.environ["HADOOP_HOME"] = HADOOP_HOME
  sys.path.append(HADOOP_HOME + "\\bin")
  os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
  os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON
    
  conf = SparkConf() \
    .set("spark.master","local") \
    .set("spark.app.name","DBALab") \
    .set("spark.jars",JDBC_JAR)

  # Initialize a Spark session
  spark = SparkSession.builder.config(conf=conf).getOrCreate()

  #Create and point to your pipelines here
  # OI = (spark.read.format("jdbc") \
  #            .option("driver","org.postgresql.Driver") \
  #            .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require") \
  #            .option("dbtable", "oldinstance.operationinterruption") \
  #            .option("user", "laura.sola.garcia") \
  #            .option("password", "DB191103") \
  #            .load())
    
  DW = (spark.read.format("jdbc") \
             .option("driver","org.postgresql.Driver") \
             .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
             .option("dbtable", "public.aircraftutilization") \
             .option("user", "laura.sola.garcia") \
             .option("password", "DB191103") \
             .load())
    
  DW.show(10)

  #OI.show(10)

  csv_files = "./resources/trainingData/*.csv"
  df = spark.read.csv(csv_files, header=True, inferSchema=True, sep=";")

  df = df.withColumn("flightid", expr(f"substring(input_file_name(),  length(input_file_name()) - {30} + 1)"))
  df = df.withColumn("aircraftid", substring("flightid", 21, 6))
  df = df.withColumn("timeid", date_format(col("date"), "yyyy-MM-dd"))
  
  #df.show(10)

  avg_sensors = df.groupBy("aircraftid", "timeid").avg("value")
  #avg_sensors.show(10)

  final = avg_sensors.join(DW, ['aircraftid','timeid'])
  
  final = final.withColumn("Label", lit(0))
  final = final.withColumn("Label", when((DW["timeid"] == final["timeid"]) & (DW["unscheduledoutofservice"] == 1), lit(1)).otherwise(col("Label")))

  final = final.withColumn("timeid", date_format(col("timeid")))
  start, end = final.select(min("timeid"), max("timeid")).first() # First and last dates of the dataset

  time_df = ("timeid", expr("sequence(start, end, interval 1 day)")) # Dataset with only dates
  final2 = final.join(time_df, "timeid", "inner") # Join of aircraft table and dates table

  final2.show(10)

  final2.orderBy(("aircraftid", "dateid").desc())

  id_partitions = Window.partitionBy("aircraftid").orderBy("timeid").rangeBetween(-7, 0)
  final2 = final2.withColumn("label_7days", lag("Label").over(id_partitions))
  final2 = final2.withColumn("Label", when((col("Label") == 1) | (col("label_7days") == 1), 1).otherwise(col("Label")))
  final2 = final2.drop("label_7days")

  final2.show()
