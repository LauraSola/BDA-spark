""""
This code performs the construction of the management pipeline. It consists of a single function with input parameters OI, DW and df: 
- OI is referred to the operationinterruption table from AMOS database.
- DW is the database provided to do the project.
- df is a dataframe which consists of all csv with sensors information joined.
With all that we will return a matrix with the 7 days labeled whenever there is an operation interruption.
"""

from pyspark.sql.functions import input_file_name, expr, substring, col, date_format, lit, date_add
from datetime import timedelta

def label_propagation(df2, OI):
    # Create column with the timeid plus 7 days
    df2 = df2.select(date_add(df2.timeid, 7).alias('7day_timeid'))

    range_condition = [df2.timeid >= OI.timeid, df2.7day_timeid <= OI.timeid]
    labels_df = df2.join(OI, range_condition, 'left')
    labels_df = labels_df.fillna(0, subset=["label"])

    return labels_df


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


def management_pipeline(OI, DW, df):
    # Selection of the attributes we'll need from DW
    DW = DW.select(['aircraftid', 'timeid', 'flightcycles', 'flighthours', 'delayedminutes'])

    # Selection and transformation attributes from the operation
    OI = OI.filter("subsystem = '3453'").withColumn("starttime", date_format(col("starttime"), "yyyy-MM-dd"))\
        .withColumnRenamed('aircraftregistration', 'aircraftid').withColumnRenamed('starttime', 'timeid') \
        .select('aircraftid', 'timeid').distinct().withColumn("label", lit(1))

    df2 = dataframe_construction(DW, df)
    matrix = label_propagation(df2, OI)
    
    matrix.write.options(header='True', delimiter=',').csv("./data_matrix.csv")
