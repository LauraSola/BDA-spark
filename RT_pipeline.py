import mlflow
from pyspark.sql.functions import input_file_name, expr, substring, col, date_format

"""
Given an aircraft and a day, the Run-Time classifier pipeline:
1- Replicates the datamanagement pipeline (extract KPIs from DW and computes the avg of the sensor of that day).
2- Prepares the tuple to be inputed into the model.
3- Classifies the record and outputs maintenance / no maintenance
"""



def create_data(aircraft, day, dataframe):
    """ Given an aircraft and a day it returns a new dataframe with only 1 row. That row
    contains the information that aircraft and day had in the dataframe."""
    new_dataframe = dataframe.filter((col("aircraftid") == aircraft) & (col("timeid") == day))

    return new_dataframe


def dataframe_construction(DW, df, aircraft, day):
    # Selection of the last letters of the file name (to extract the flightid) and writing the date in the correct format
    df = df.withColumn("flightid", expr(f"substring(input_file_name(),  length(input_file_name()) - {30} + 1)"))
    df = df.withColumn("aircraftid", substring("flightid", 21, 6))
    df = df.withColumn("timeid", date_format(col("date"), "yyyy-MM-dd"))

    # Computing the average of the sensor per aircraft per day
    avg_sensors = df.groupBy("aircraftid", "timeid").avg("value")

    # Filtering both datasets so we have only a row with the data we are interested in
    avg_sensors = avg_sensors.filter((col("aircraftid") == aircraft) & (col("timeid") == day))
    DW = DW.filter((col("aircraftid") == aircraft) & (col("timeid") == day))

    # Join between data from the csv (sensors) and DW (KPIs)
    df2 = DW.join(avg_sensors, ['aircraftid','timeid'], how = "inner")

    return df2

def model_prediction(pred_data):

    experiment = mlflow.get_experiment_by_name("/spark-best-models/")

    if not experiment:
        raise Exception("Model not found, need to execute data analysis pipeline first")


    runs = mlflow.MlflowClient().search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="tags.best_model = 'this is the best model'",
        max_results=1)[0]
    
    loaded_model = mlflow.spark.load_model(f"runs:/{runs.info.run_id}/{runs.info.run_name}")

    pred = loaded_model.transform(pred_data)

    # Extract both 'prediction' and 'probability' columns from the DataFrame
    predictions_with_probabilities = pred.select("prediction", "probability").first()

    
    return predictions_with_probabilities['prediction'], predictions_with_probabilities['probability']

    

def prediction(DW, df):
    aircraft = input("Aircraft model:")
    day = input("Day in format yyyy-mm-dd:")
    dataframe = dataframe_construction(DW, df, aircraft, day)
    if dataframe.count() > 0:
        pred_data = create_data(aircraft, day, dataframe)
        predicted_value, pred_prob = model_prediction(pred_data)
        if  predicted_value == 0.0:
            print(f"That aircraft and day will not have an Operation Interruption with probability {pred_prob[0]:.3f}")
        else:
            print(f"That aircraft and day will have an Operation Interruption with probability {pred_prob[1]:.3f}")
    else:
        print("Invalid input data")
