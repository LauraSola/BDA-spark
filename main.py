import os
import sys
import glob # windows
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf


import Manag_pipeline
import data_analysis_pipe
import RT_pipeline

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
  
    if(sys.argv[1] == "Management"):
        Manag_pipeline.management_pipeline(OI, DW, df)
    elif(sys.argv[1] == "Analysis"):
        if os.path.isdir('./data_matrix'):
            print("File Exists")
            data_matrix = spark.read.csv("./data_matrix/*.csv", header=True, inferSchema=True, sep=",")
            data_analysis_pipe.data_analysis_pipeline(data_matrix)
            print("Models and metrics now accessible in MLflow")
        else:            
            print("File doesn't exist, need to execute Management Pipeline first")
    elif(sys.argv[1] == "Classifier"):
        RT_pipeline.prediction(DW, df)
