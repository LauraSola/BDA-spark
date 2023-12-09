import os
import sys
import glob # windows
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf

import Manag_pipeline

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import mlflow
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

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
  OI = (spark.read.format("jdbc") \
              .option("driver","org.postgresql.Driver") \
              .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require") \
              .option("dbtable", "oldinstance.operationinterruption") \
              .option("user", "laura.sola.garcia") \
              .option("password", "DB191103") \
              .load())
    
  DW = (spark.read.format("jdbc") \
             .option("driver","org.postgresql.Driver") \
             .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
             .option("dbtable", "public.aircraftutilization") \
             .option("user", "laura.sola.garcia") \
             .option("password", "DB191103") \
             .load())
  
  csv_files = "./resources/trainingData/*.csv"
  df = spark.read.csv(csv_files, header=True, inferSchema=True, sep=";")
  
  data_matrix = Manag_pipeline.management_pipeline(OI, DW, df)
  
  ########### models ##################

  feature_columns = ["flighthours", "flightcycles", "delayedminutes", "avg(value)"] 
  vecAssembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

  train_split, test_split = data_matrix.randomSplit(weights = [0.80, 0.20], seed = 13)

  experiment_name = "/spark-models-tests/"
  mlflow.set_experiment(experiment_name)
  
  def train_model(params, classifier, train_split, test_split, model_name):
    global iters
    name = f"{model_name}_iteration_{iters}"
    iters +=1
    with mlflow.start_run(nested = True, run_name = name  ):
      
      # Convert hyperparameter values to appropriate types
      for key in params:
          if key in ['maxBins', 'maxDepth', 'numTrees', 'maxIter']:
              params[key] = int(params[key])

      
      # Create classifier instance with hyperparameters
      clf = classifier(**params, labelCol="label_prop", featuresCol="features" )

      # Build the ML Pipeline
      pipeline = Pipeline(stages=[vecAssembler, clf])
      model = pipeline.fit(train_split)

      # Evaluate the model
      evaluator = BinaryClassificationEvaluator(labelCol="label_prop", metricName="areaUnderROC")
      predictions = model.transform(test_split)
      validation_metric = evaluator.evaluate(predictions)

      mlflow.log_metrics({"AUC": validation_metric})
      mlflow.log_params(params)
    
      #mlflow.spark.save_model(model,name)


      #print(validation_metric)

    return {'loss': -validation_metric, 'status': STATUS_OK}

def train_with_hyperopt(train_function, space, max_evals, classifier, train_split, test_split, model_name):
    
    with mlflow.start_run(run_name = model_name):
        best_params = fmin(
            fn=lambda params: train_function(params, classifier, train_split, test_split, model_name),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals
        )
    mlflow.end_run()

    return best_params

iters = 1
# Decision Tree Hyperopt
dt_space = {
    'maxDepth': hp.uniform('maxDepth', 3, 11),
    'maxBins': hp.uniform('maxBins', 8, 64),
}

best_dt_params = train_with_hyperopt(train_model, dt_space, 2, DecisionTreeClassifier, train_split, test_split, "decision_tree")

# iters = 1
# # Random Forest Hyperopt
# rf_space = {
#     'numTrees': hp.choice('numTrees', range(5, 50)),
#     'maxDepth': hp.uniform('maxDepth', 3, 11),
#     'maxBins': hp.uniform('maxBins', 8, 64),
# }

# best_rf_params = train_with_hyperopt(train_model, rf_space, 2, RandomForestClassifier, train_split, test_split, "random forest")

iters = 1
# Logistic Regression Hyperopt
lr_space = {
    'regParam': hp.uniform('regParam', 0, 1),
    'elasticNetParam': hp.uniform('elasticNetParam', 0, 1),
    'maxIter' : hp.uniform('maxIter', 5, 25)
}

best_lr_params = train_with_hyperopt(train_model, lr_space, 2, LogisticRegression, train_split, test_split, "logistic_regression")



# ### FALTA:
#     - VAL SPLIT
#     - GUARDAR-SE ELS MILLORS MODELS
#     - ELS MILLOR MODEL REENTRENARLO AMB TOTES LES DADES DE TRAIN PER DECIDIR EL MILLOR MODEL DE TOTS