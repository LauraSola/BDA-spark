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

  
  data_matrix = Manag_pipeline.management_pipeline(OI, DW, df)
  
  ########### models ##################

  feature_columns = ["flighthours", "flightcycles", "delayedminutes", "avg(value)"] 
  vecAssembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

  train_split, test_split = data_matrix.randomSplit(weights = [0.80, 0.20], seed = 13)

  evaluator = BinaryClassificationEvaluator(labelCol="label_prop")
  metric_names = ["areaUnderPR", "areaUnderROC", "accuracy"]


  experiment_name = "/spark-models-tests/"
  mlflow.set_experiment(experiment_name)
  
  def train_model(params, classifier, train_split, model_name):

    train2_split, val_split = train_split.randomSplit(weights = [0.80, 0.20], seed = 13)

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
      model = pipeline.fit(train2_split)

      # Evaluate the model
      
      AUC_value = evaluate_models(model, val_split, metric_names)

      mlflow.log_params(params)
    
      #mlflow.spark.save_model(model,name)

      #mlflow.log_model()


      #print(validation_metric)

    return {'loss': -AUC_value, 'status': STATUS_OK}

def train_with_hyperopt(train_function, space, max_evals, classifier, train_split, model_name):
    
    with mlflow.start_run(run_name = model_name):
        best_params = fmin(
            fn=lambda params: train_function(params, classifier, train_split, model_name),
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

best_dt_params = train_with_hyperopt(train_model, dt_space, 2, DecisionTreeClassifier, train_split, "decision_tree")

iters = 1
# Random Forest Hyperopt
rf_space = {
    'numTrees': hp.uniform('numTrees', 5, 50),
    'maxDepth': hp.uniform('maxDepth', 3, 11),
    'maxBins': hp.uniform('maxBins', 8, 64),
}

best_rf_params = train_with_hyperopt(train_model, rf_space, 2, RandomForestClassifier, train_split, "random forest")

iters = 1
# Logistic Regression Hyperopt
lr_space = {
    'regParam': hp.uniform('regParam', 0, 1),
    'elasticNetParam': hp.uniform('elasticNetParam', 0, 1),
    'maxIter' : hp.uniform('maxIter', 5, 25)
}

best_lr_params = train_with_hyperopt(train_model, lr_space, 2, LogisticRegression, train_split, "logistic_regression")

def evaluate_models(model, split, metric_names):
    predictions = model.transform(test_split)
    test_AUC = evaluator.evaluate(predictions, {evaluator.metricName{metric_names[0]}})
    test_PR = evaluator.evaluate(predictions, {evaluator.metricName{metric_names[1]}})
    test_accuracy = evaluator.evaluate(predictions, {evaluator.metricName{metric_names[2]}})

    mlflow.log_metrics({"AUC": test_AUC})
    mlflow.log_metrics({"Precision-Recall": test_PR})
    mlflow.log_metrics({"Accuracy": test_accuracy})

    return test_AUC


def train_and_test_best_models(best_model_params, train_split, test_split):
    with mlflow.start_run(run_name="best models"):
        for classifier, (model_name, params) in model_params.items():
            with mlflow.start_run(run_name = model_name):
                # Create classifier instance with hyperparameters
                clf = classifier(**params, labelCol="label_prop", featuresCol="features")

                # Build the ML Pipeline
                pipeline = Pipeline(stages=[vecAssembler, clf])
                model = pipeline.fit(train_split)

                # Evaluate the model on test data
                AUC_value = evaluate_models(model, test_split, metric_names)

                mlflow.log_params(params)

model_params = {DecisionTreeClassifier : ["decision_tree",best_dt_params], RandomForestClassifier : ["random_forest",best_rf_params], LogisticRegression : ["logistic regression", best_lr_params]}
train_and_test_best_models(model_params, train_split, test_split)
