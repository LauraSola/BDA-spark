import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name,expr, substring

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
#     OI = (spark.read.format("jdbc") \
#             .option("driver","org.postgresql.Driver") \
#             .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require") \
#             .option("dbtable", "oldinstance.operationinterruption") \
#             .option("user", "laura.sola.garcia") \
#             .option("password", "DB191103") \
#             .load())
    
#     DW = (spark.read.format("jdbc") \
#             .option("driver","org.postgresql.Driver") \
#             .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
#             .option("dbtable", "public.aircraftutilization") \
#             .option("user", "laura.sola.garcia") \
#             .option("password", "DB191103") \
#             .load())
    
    #DW.show(10)

    #OI.show(10)

    csv_files = "./resources/trainingData/*.csv"
    df = spark.read.csv(csv_files, header=True, inferSchema=True, sep=";")

    df = df.withColumn("flight_id", expr(f"substring(input_file_name(),  length(input_file_name()) - {30} + 1)"))

    #(df.select("flight_id")).show(10)

    df = df.withColumn("aircraft_id", substring("flight_id", 21, 6))

    df.show(10)

    #print(df.count())


