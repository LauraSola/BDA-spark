import os
import sys
import glob # windows
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import input_file_name, expr, substring, col, date_format, lit
from pyspark.sql.window import Window
from datetime import timedelta


def runtime_classifier(df):
    