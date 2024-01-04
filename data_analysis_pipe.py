"""
This code trains and evaluates three classification models (decision tree, random forest and logisitc regression) following this steps:

    1. Split data into train and test
    2. Define each classification model and the search space for their hyperparameters
    3. By training the models on a subset of the training data, and using the rest as a validation set, use bayesian optimization 
        to find the best hyperparameters (split training data into )
    4. With the best parameters for each model, retrain them on the whole training set and evaluate their prediction capacity on the test set

Along all these steps, MlFlow is used to store the models, their corresponding hyperparameters and evaluation metrics 

"""
# Import necessary libraries from PySpark and MLFlow
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK

# Initialize iteration counter
iters = 1

def evaluate_models(model, split):
    """
    Evaluate the models using the MulticlassClassificationEvaluator. Computes AUC and Precision-Recall metrics and logs them using MLFlow.
    
    """

    # Generate predictions using the model
    predictions = model.transform(split)
    
    # Define the evaluator and metrics
    evaluator = MulticlassClassificationEvaluator(labelCol="label")
    metric_names = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]

    # Compute evaluation metrics
    test_accuracy = evaluator.evaluate(predictions, {evaluator.metricName: metric_names[0]})
    test_precision = evaluator.evaluate(predictions, {evaluator.metricName: metric_names[1]})
    test_recall = evaluator.evaluate(predictions, {evaluator.metricName: metric_names[2]})
    test_f1 = evaluator.evaluate(predictions, {evaluator.metricName: metric_names[3]})

    # Log evaluation metrics using MLFlow
    mlflow.log_metrics({"Accuracy": test_accuracy})
    mlflow.log_metrics({"Precision": test_precision})
    mlflow.log_metrics({"Recall": test_recall})
    mlflow.log_metrics({"F1": test_f1})

    return test_accuracy


def train_model(params, classifier, train_split, model_name, vecAssembler):
    """
    Train the model with specified hyperparameters. Splits the training data into training and validation sets.
    Uses Bayesian optimization for hyperparameter tuning and logs the results using MLFlow.
    
    """

    # Split the training data into train2_split (80%) and val_split (20%)
    train2_split, val_split = train_split.randomSplit(weights=[0.80, 0.20])
    
    # Generate a unique name for the MLFlow run
    global iters
    name = f"{model_name}_iteration_{iters}"
    iters += 1
    with mlflow.start_run(nested=True, run_name=name):
        # Convert hyperparameter values to appropriate types
        for key in params:
            if key in ['maxBins', 'maxDepth', 'numTrees', 'maxIter']:
                params[key] = int(params[key])
        
        # Create classifier instance with specified hyperparameters
        clf = classifier(**params, labelCol="label", featuresCol="features")

        # Build the ML Pipeline
        pipeline = Pipeline(stages=[vecAssembler, clf])
        model = pipeline.fit(train2_split)

        # Evaluate the model
        accuracy_value = evaluate_models(model, val_split)

        # Log hyperparameters using MLFlow
        mlflow.log_params(params)

    return {'loss': -accuracy_value, 'status': STATUS_OK}


def train_with_hyperopt(train_function, space, max_evals, classifier, train_split, model_name, vecAssembler):
    """
    Perform hyperparameter tuning using Bayesian optimization. Searches for the best hyperparameters within the
    specified search space. Logs the best hyperparameters using MLFlow.
    
    """

    # Start MLFlow run for the specific model
    with mlflow.start_run(run_name=model_name):
        # Use Hyperopt's fmin function to find the best hyperparameters
        best_params = fmin(
            fn=lambda params: train_function(params, classifier, train_split, model_name, vecAssembler),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals
        )
    mlflow.end_run()

    # Convert specific hyperparameters to integer types
    for key in best_params:
        if key in ['maxBins', 'maxDepth', 'numTrees', 'maxIter']:
            best_params[key] = int(best_params[key])

    return best_params


def train_and_test_best_models(model_params, train_split, test_split, vecAssembler):
    """
    Train and evaluate the best models with optimal hyperparameters. Builds the ML Pipeline, trains the model, 
    evaluates it on the test set, and logs the results using MLFlow.

    """

    # Iterate through each classifier and its corresponding hyperparameters
    for classifier, (model_name, params) in model_params.items():
        with mlflow.start_run(run_name=model_name):
            # Create classifier instance with optimal hyperparameters
            clf = classifier(**params, labelCol="label", featuresCol="features")
            
            # Build the ML Pipeline
            pipeline = Pipeline(stages=[vecAssembler, clf])
            model = pipeline.fit(train_split)

            # Evaluate the model on the test data
            accuracy = evaluate_models(model, test_split)

            # Log hyperparameters and the model using MLFlow
            mlflow.log_params(params)
            mlflow.spark.log_model(model, model_name, registered_model_name = model_name)

        mlflow.end_run()

    # Set tag to identify the best model
    experiment = mlflow.get_experiment_by_name("/spark-best-models/")
    best_run = mlflow.MlflowClient().search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="",
        max_results=1,
        order_by=["metrics.Accuracy DESC"],
    )[0]
    mlflow.start_run(run_id=best_run.info.run_id)
    mlflow.set_tag("best_model", "this is the best model")



############# Functions to perform hyperparameter tuning for each classification model using Bayesian optimization ############
    
def train_decision_tree(train_split, vecAssembler):
    global iters
    iters = 1
    dt_space = {
        'maxDepth': hp.uniform('maxDepth', 3, 11),
        'maxBins': hp.uniform('maxBins', 8, 64),
    }
    return train_with_hyperopt(train_model, dt_space, 8, DecisionTreeClassifier, train_split, "decision_tree", vecAssembler)

def train_random_forest(train_split, vecAssembler):
    global iters
    iters = 1
    rf_space = {
        'numTrees': hp.uniform('numTrees', 5, 50),
        'maxDepth': hp.uniform('maxDepth', 3, 11),
        'maxBins': hp.uniform('maxBins', 8, 64),
    }
    return train_with_hyperopt(train_model, rf_space, 8, RandomForestClassifier, train_split, "random forest", vecAssembler)

def train_logistic_regression(train_split, vecAssembler):
    global iters
    iters = 1
    lr_space = {
        'regParam': hp.uniform('regParam', 0, 1),
        'elasticNetParam': hp.uniform('elasticNetParam', 0, 1),
        'maxIter': hp.uniform('maxIter', 5, 25)
    }
    return train_with_hyperopt(train_model, lr_space, 8, LogisticRegression, train_split, "logistic_regression", vecAssembler)



def data_analysis_pipeline(data_matrix):

    """
    Main pipeline function to execute the entire workflow. Sets up the feature columns, splits the data, performs hyperparameter tuning, 
    and evaluates the best models using MLFlow.

    """

    # Format data accordin to what is expected by the classifier algorithms.
    feature_columns = ["flighthours", "flightcycles", "delayedminutes", "avg(value)"]
    vecAssembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Split the data into training and test sets
    train_split, test_split = data_matrix.randomSplit(weights=[0.80, 0.20])

    # Set MLFlow experiment name for hyperparameter tuning
    experiment_name = "/spark-bayesian-opt/"
    mlflow.set_experiment(experiment_name)

    # Perform hyperparameter tuning for each classifier to find the optimal hyperparameters
    best_dt_params = train_decision_tree(train_split, vecAssembler)
    best_rf_params = train_random_forest(train_split, vecAssembler)
    best_lr_params = train_logistic_regression(train_split, vecAssembler)

    # Set MLFlow experiment name for evaluating and testing the best models
    experiment_name = "/spark-best-models/"
    mlflow.set_experiment(experiment_name)

    # Define the best hyperparameters for each classifier
    model_params = {
        DecisionTreeClassifier: ["decision_tree", best_dt_params],
        RandomForestClassifier: ["random_forest", best_rf_params],
        LogisticRegression: ["logistic_regression", best_lr_params]
    }

    # Train and test the best models using the optimal hyperparameters
    train_and_test_best_models(model_params, train_split, test_split, vecAssembler)
