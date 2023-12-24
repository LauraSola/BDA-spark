import os
import sys
import glob # windows
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import input_file_name, expr, substring, col, date_format, lit
from pyspark.sql.window import Window
from datetime import timedelta

import Manag_pipeline
import RT_pipeline

# Building a spark session
HADOOP_HOME = "C:/Users/Maria/OneDrive/Escritorio/BDA/Projecte2/CodeSkeleton/resources/hadoop_home"
PYSPARK_PYTHON = "python3.11"
PYSPARK_DRIVER_PYTHON = "python3.11"

os.environ["HADOOP_HOME"] = HADOOP_HOME
sys.path.append(HADOOP_HOME + "\\bin")
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

POSTGRESQL_DRIVER_PATH = "C:/Users/Maria/OneDrive/Escritorio/BDA/Projecte2/CodeSkeleton/resources/postgresql-42.2.8.jar"
conf = SparkConf() \
      .set("spark.master","local") \
      .set("spark.app.name","DBALab") \
      .set("spark.jars",POSTGRESQL_DRIVER_PATH)

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Loading the operationinterruption table from the AMOS database
OI = (spark.read.format("jdbc") \
            .option("driver","org.postgresql.Driver") \
            .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require") \
            .option("dbtable", "oldinstance.operationinterruption") \
            .option("user", "maria.tubella") \
            .option("password", "DB311003") \
            .load())

# Loading the DW database
DW = (spark.read.format("jdbc") \
             .option("driver","org.postgresql.Driver") \
             .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
             .option("dbtable", "public.aircraftutilization") \
             .option("user", "maria.tubella") \
             .option("password", "DB311003") \
             .load())

# Loading the information about the sensors from all csv
csv_files = "C:\\Users\\Maria\\OneDrive\\Escritorio\\BDA\\Projecte2\\CodeSkeleton\\resources\\trainingData\\trainingData\\trainingData\\*.csv"
df = spark.read.csv(csv_files, header=True, inferSchema=True, sep=";")
df.show(5)

if(__name__== "__main__"):
      if(sys.argv[1] == "Management Pipeline"):
            data_matrix = spark.read.csv(Manag_pipeline.management_pipeline(OI, DW, df), header=True, inferSchema=True, sep=";") # CSV with data matrix
      elif(sys.argv[1] = "Data Analysis Pipeline"):
            print("Models and metrics now accessible in MLflow")
      elif(sys.argv[1] = "Runtime Pipeline"):
            print(RT_pipeline.prediction(DW, df))  


