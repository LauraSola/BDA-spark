"""
Given an aircraft and a day, the Run-Time classifier pipeline:
1- Replicates the datamanagement pipeline (extract KPIs from DW and computes the avg of the sensor of that day).
2- Prepares the tuple to be inputed into the model.
3- Classifies the record and outputs maintenance / no maintenance
"""

from pyspark.sql.functions import input_file_name, expr, substring, col, date_format, lit, date_add


def valid_data(aircraft, day, dataframe):
    """ Given an aircraft and a day returns True if those have a sensor"""
    row = dataframe.filter((dataframe["aircraft"] == aircraft) & (dataframe["day"] == day))
    if row.count() > 0: 
        return True
    
    return False


def create_data(aircraft, day, dataframe):
    """ Given an aircraft and a day it returns a new dataframe with only 1 row. That row
    contains the information that aircraft and day had in the dataframe."""
    new_dataframe = dataframe.filter((col("aircraftid") == aircraft) & (col("timeid") == day))

    return new_dataframe


def dataframe_construction(DW, df):
    # Selection of the last letters of the file name (to extract the flightid) and writing the date in the correct format
    df = df.withColumn("flightid", expr(f"substring(input_file_name(),  length(input_file_name()) - {30} + 1)"))
    df = df.withColumn("aircraftid", substring("flightid", 21, 6))
    df = df.withColumn("timeid", date_format(col("date"), "yyyy-MM-dd"))

    # Computing the average of the sensor per aircraft per day
    avg_sensors = df.groupBy("aircraftid", "timeid").avg("value")

    # Join between data from the csv (sensors) and DW (KPIs)
    df2 = DW.join(avg_sensors, ['aircraftid','timeid'], how = "inner")

    return df2

def model_prediction(pred_data, 'MODEL'):
    # aqui caldria fer les prediccions amb el millor model
    # i un cop fet veure si la label surt 0 o 1

def prediction(DW, df):
    aircraft = input("Aircraft model:")
    day = input("Day in format yyy-mm-dd:")
    dataframe = dataframe_construction(DW, df)
    if valid_data(aircraft, day, dataframe):
        pred_data = create_data(aircraft, day, dataframe)
        if model_prediction(pred_data, 'MODEL') == 0:
            print("That aircraft and day will not have an Operation Interruption")
        else:
            print("That aircraft and day will have an Operation Interruption")
    else:
        print("Invalid input data")

