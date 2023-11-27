import os
import sys
import pyspark
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, expr, substring, col, date_format, sum, lit, when, filter, min, max, lead
import pyspark.sql.functions as F
from pyspark.sql.window import Window

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler
#import mlflow

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
    
  #DW.show(10)

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
  #final.show(10)

  final = final.withColumn("Label", lit(0))
  final = final.withColumn("Label", when((DW["timeid"] == final["timeid"]) & (DW["unscheduledoutofservice"] == 1.0), lit(1)).otherwise(col("Label")))

  #final.show(10)

  final = final.withColumn("timeid", date_format(col("timeid"), "yyyy-MM-dd" ))
  start, end = final.select(min("timeid"), max("timeid")).first() # First and last dates of the dataset

  from datetime import datetime

  start_date= datetime.strptime(start, '%Y-%m-%d').date()
  end_date= datetime.strptime(end, '%Y-%m-%d').date()

  date_range = pd.date_range(start=start_date, end=end_date, freq='D')

  # Convert the pandas DataFrame to a PySpark DataFrame
  time_df = spark.createDataFrame(date_range.to_frame(name="timeid"))
  time_df = time_df.withColumn("timeid", date_format(col("timeid"), "yyyy-MM-dd" ))

  #time_df.show(5)

  aircraft_df = final.select(col('aircraftid')).distinct()

  #aircraft_df.show(10)

  time_aircraft = time_df.crossJoin(aircraft_df)

  #time_aircraft.show(10)

  final2 = time_aircraft.join(final, on = ['timeid', 'aircraftid'], how = "left") # Join of aircraft table and dates table


  #final2_sorted = final2.orderBy(col("timeid").desc())


  #final2_sorted.show(50)

  window_spec = Window().partitionBy('aircraftid').orderBy('timeid')

  # Crear una nueva columna 'label7days' utilizando la lógica especificada
  final2 = (
      final2
      .withColumn('label7days', F.when(F.col('label') == 1, 1).otherwise(0))
      .withColumn('label7days',
                F.max('label7days').over(window_spec.rowsBetween(0,6)))
  ) 

  final2_filtered = final2.filter((final2['aircraftid']=='XY-OQQ') & (final2['timeid']>'2013-06-20'))

  final2_filtered.show(50)

  final_sin_nul = final2.na.drop()

  final_sin_nul.show(50)

  

  #seven_days = Window.partitionBy("aircraftid").orderBy("timeid").rowsBetween(-7, 0)
  #final = final.withColumn("7days", lag("Label").over(seven_days))
  #final = final.withColumn("Label", when((col("Label") == 1) | (col("7days") == 1), 1).otherwise(col("Label")))
  #final = final.drop("7days")

  #final.show(10)


  ############## ANALYSIS #####################  provarem decision tree, random forest, svm

  # create experiment id for tracking using mlflow
  #exp_id = mlflow.create_experiment("solving-mnist1")
  
  # train_split, test_split = final.randomSplit(weights = [0.80, 0.20], seed = 13)

  # feature_columns = ["flighthours", "flightcycles", "delayedminutes", "avg(value)"] 
  # vecAssembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

  # # decision_tree = DecisionTreeClassifier(labelCol="Label", featuresCol="features", impurity = "gini" )

  # # pipeline = Pipeline(stages=[vecAssembler, decision_tree])

  # # paramGrid = (ParamGridBuilder()
  # #            .addGrid(dt.maxDepth, [3, 5, 7, 10])
  # #            .addGrid(dt.maxBins, [16, 32, 64])  
  # #            .build())

  # # # Define the evaluator
  # # evaluator = BinaryClassificationEvaluator(labelCol="Label", metricName="areaUnderROC")  #metrica bona per unbalanced datasets

  # # #Create a cross-validator
  # # crossval = CrossValidator(estimator=pipeline,
  # #                           estimatorParamMaps=param_grid,
  # #                           evaluator=evaluator,
  # #                           numFolds=5)
  

  # rf = RandomForestClassifier(labelCol="Label", featuresCol="features")

  # pipeline = Pipeline(stages=[vecAssembler, rf])

  # # Evaluate model
  # rfevaluator = BinaryClassificationEvaluator(labelCol="Label", metricName="areaUnderROC")

  # # Create ParamGrid for Cross Validation
  # rfparamGrid = (ParamGridBuilder()
  #            .addGrid(rf.maxDepth, [3, 5, 10])
  #            .addGrid(rf.maxBins, [16, 32, 64])
  #            .addGrid(rf.numTrees, [10, 50, 100])
  #            .build())

  # # Create 5-fold CrossValidator
  # rfcv = CrossValidator(estimator = pipeline,
  #                       estimatorParamMaps = rfparamGrid,
  #                       evaluator = rfevaluator,
  #                       numFolds = 5)

  # # Fit the model
  # cv_model = rfcv.fit(train_split)

  # # Make predictions on the test set
  # predictions = cv_model.transform(test_split)

  # # Evaluate the model on the test set
  # area_under_roc = rfevaluator.evaluate(predictions)

  # print(area_under_roc)

  #posem tota la merdeta de MlFlow

  # def evaluate_performance(y_test, y_pred):
  #     accuracy = accuracy_score(y_test, y_pred)
  #     f1_macro = f1_score(y_test, y_pred, average='macro')
  #     f1_micro = f1_score(y_test, y_pred, average='micro')
  #     precision = precision_score(y_test, y_pred, average='micro')
  #     recall = recall_score(y_test, y_pred, average='micro')
          
  #     return accuracy, precision, recall, f1_micro, f1_macro

  # def track_performance_metrics(accuracy, precision, recall, f1_micro, f1_macro):
  #     # performance test set
  #     mlflow.log_metric('accuracy', accuracy)
  #     mlflow.log_metric('f1_macro', f1_macro)
  #     mlflow.log_metric('f1_micro', f1_micro)
  #     mlflow.log_metric('precision', precision)
  #     mlflow.log_metric('recall', recall)

  #accuracy, precision, recall, f1_micro, f1_macro = evaluate_performance(y_test, y_pred)
  #track_performance_metrics(accuracy, precision, recall, f1_micro, f1_macro)

  # logreg = LogisticRegression(random_state = 33) # use default hyperparameters
  # with mlflow.start_run(experiment_id = exp_id):
  #     mlflow.log_param('model', 'logistic_regression')
  #     perform_experiment(logreg,'sklearn', X_train_flat, X_test_flat, y_train, y_test)

  #log_model() --> mlflow.spark.log_model(model, "spark-model")

  





  



