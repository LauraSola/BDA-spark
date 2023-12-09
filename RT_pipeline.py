"""
Given an aircraft and a day, the Run-Time classifier pipeline:
1- Replicates the datamanagement pipeline (extract KPIs from DW and computes the avg of the sensor of that day).
2- Prepares the tuple to be inputed into the model.
3- Classifies the record and outputs maintenance / no maintenance
"""

from pyspark.sql import Row, union


""" Given an aircraft and a day returns True if those have a sensor"""
def valid_data(aircraft, day, dataframe):
    row = dataframe.filter((dataframe["aircraft"] == aircraft) & (dataframe["day"] == day))
    if row.count() > 0: 
        return True
    
    return False

def add_row(aircraft, day, dataframe):
    # Get the row that coincides with the input aircraft and day
    existent_row = dataframe.filter((dataframe["aircraft"] == aircraft) & (dataframe["day"] == day))
    new_row = Row(**existent_row) # Copy the values from that row
    dataframe = dataframe.union(new_row) # Add the row to the dataframe

    return dataframe