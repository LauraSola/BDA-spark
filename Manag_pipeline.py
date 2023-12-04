""""
This code performs the construction of the management pipeline. It consists of a single function with input parameters OI, DW and df: 
- OI is referred to the operationinterruption table from AMOS database.
- DW is the database provided to do the project.
- df is a dataframe which consists of all csv with sensors information joined.
With all that we will return a matrix with the 7 days labeled whenever there is an operation interruption.
"""

from pyspark.sql.functions import input_file_name, expr, substring, col, date_format, lit
from datetime import timedelta


def management_pipeline(OI, DW, df):
    # Selection of the attributes we'll need from DW
    DW = DW.select(['aircraftid', 'timeid', 'flightcycles', 'flighthours', 'delayedminutes'])

    # Selection and transformation attributes from the operation
    OI = OI.filter("subsystem = '3453'").withColumn("starttime", date_format(col("starttime"), "yyyy-MM-dd"))\
        .withColumnRenamed('aircraftregistration', 'aircraftid').withColumnRenamed('starttime', 'timeid') \
        .select('aircraftid', 'timeid').distinct().withColumn("label", lit(1))

    # Selection of the last letters of the file name (to extract the flightid) and writing the date in the correct format
    df = df.withColumn("flightid", expr(f"substring(input_file_name(),  length(input_file_name()) - {30} + 1)"))
    df = df.withColumn("aircraftid", substring("flightid", 21, 6))
    df = df.withColumn("timeid", date_format(col("date"), "yyyy-MM-dd"))

    # Computing the average of the sensor per aircraft per day
    avg_sensors = df.groupBy("aircraftid", "timeid").avg("value")

    # Join between data from the csv (sensors) and DW (KPIs)
    df2 = DW.join(avg_sensors, ['aircraftid','timeid'], how = "inner")

    # Join to add the column with the label to the df with the KPIs and sensors
    labels_df = df2.join(OI, ['aircraftid','timeid'], how = "left")

    # Map to propagate the rows that have an operation interruption
    prop_labels = labels_df.filter((labels_df['label']== 1)).rdd.flatMap(lambda t: [((t['timeid'] - timedelta(days=days),t['aircraftid']),t['label']) for days in range(8)])\
                            .distinct().map(lambda t : (t[0][0], t[0][1], t[1])).toDF(['timeid', 'aircraftid', 'label_prop'])

    matrix = labels_df.join(prop_labels, ['aircraftid','timeid'], how = "left")   
    matrix = matrix.select(['aircraftid','timeid','flightcycles', 'flighthours', 'delayedminutes', 'avg(value)', 'label_prop']).fillna(0, subset = ['label_prop'])  

    return matrix