import os
import sys
import glob # windows
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf


import Manag_pipeline
import data_analysis_pipe
import rt

def read_config(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip().replace('"', '')
    return config


if __name__ == "__main__":
  config = read_config('./config.txt')
  
  JDBC_JAR = config['JDBC_JAR']
  HADOOP_HOME = config['HADOOP_HOME']
  PYSPARK_PYTHON = config['PYSPARK_PYTHON']
  PYSPARK_DRIVER_PYTHON = config['PYSPARK_DRIVER_PYTHON']

  os.environ["HADOOP_HOME"] = HADOOP_HOME
  sys.path.append(os.path.join(HADOOP_HOME, "bin"))
  os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
  os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

  conf = SparkConf() \
      .set("spark.master", "local") \
      .set("spark.app.name", "DBALab") \
      .set("spark.jars", JDBC_JAR)

  spark = SparkSession.builder.config(conf=conf).getOrCreate()

  OI = (spark.read.format("jdbc")
        .option("driver", "org.postgresql.Driver")
        .option("url", config['POSTGRES_URL_AMOS'])
        .option("dbtable", config['OPERATION_INTERRUPTION'])
        .option("user", config['USERNAME'])
        .option("password", config['PASSWORD'])
        .load())

  DW = (spark.read.format("jdbc")
        .option("driver", "org.postgresql.Driver")
        .option("url", config['POSTGRES_URL_DW'])
        .option("dbtable", config['AIRCRAFT_UTILIZATION'])
        .option("user", config['USERNAME'])
        .option("password", config['PASSWORD'])
        .load())
  
  csv_files = "./resources/trainingData/*.csv"
  df = spark.read.csv(csv_files, header=True, inferSchema=True, sep=";")
  
  #data_matrix = Manag_pipeline.management_pipeline(OI, DW, df)

  #data_analysis_pipe.data_analysis_pipeline(data_matrix)

  rt.load_best_model()

