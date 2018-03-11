# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import argparse
from pyspark.sql import SparkSession
#from pyspark.sql import Row
#from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import functions as F

class FileConverter(object):
    def __init__(self, reviewFilePath, userFilePath, businessFilePath, fullRateFilePath,
                 destPath):
        self.reviewFilePath = reviewFilePath
        self.userFilePath = userFilePath
        self.businessFilePath = businessFilePath
        self.fullRateFilePath = fullRateFilePath
        self.destPath = destPath

        self.spark = SparkSession.builder.appName("YelpPreprocessor").getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.allRateDF = None

    def process(self):
        if os.path.exists(self.fullRateFilePath):
            rateSchema = StructType([StructField("userIndex", IntegerType(), True),
                                       StructField("businessIndex", IntegerType(), True),
                                       StructField("city", StringType(), True),
                                       StructField("rate", FloatType(), True)])
            self.allRateDF = self.spark.read.csv(path=self.fullRateFilePath, schema=rateSchema).cache()
        else:
            self.__genAllDataCSVFile()

        print('allRateDF count:{}'.format(self.allRateDF.count()))

        self.__getCityRateFile()


    def __genAllDataCSVFile(self):
        print('__genAllDataCSVFile')

        reviewSchema = StructType([StructField("userID", StringType(), True),
                                   StructField("businessID", StringType(), True),
                                   StructField("rate", FloatType(), True)])
        reviewDF = self.spark.read.csv(path=self.reviewFilePath, schema=reviewSchema, header=True)

        userSchema = StructType([StructField("userID", StringType(), True),
                                 StructField("userIndex", IntegerType(), True)])
        userDF = self.spark.read.csv(path=self.userFilePath, schema=userSchema, header=True)

        businessSchema = StructType([StructField("businessID", StringType(), True),
                                     StructField("businessIndex", IntegerType(), True),
                                     StructField("city", StringType(), True)])
        businessDF = self.spark.read.csv(path=self.businessFilePath, schema=businessSchema, header=True)

        self.allRateDF = reviewDF.join(userDF, reviewDF.userID == userDF.userID).join(businessDF,
                                reviewDF.businessID == businessDF.businessID).select('userIndex',
                                'businessIndex', "rate", "city").groupby(
                                'userIndex', 'businessIndex', 'city').agg(F.avg(
                                reviewDF.rate).alias('rate')).cache()

        # another way to avg
        # 'userIndex', 'businessIndex', 'city').agg({'rate': 'mean'}).cache()

        self.allRateDF.printSchema()
        #print(self.allRateDF.collect())

        #fullPath = os.path.join(os.getcwd(), 'processCSV', 'allRate')
        fullPath = os.path.join(self.destPath, 'allRate')
        print(fullPath)
        self.allRateDF.write.csv(fullPath, 'overwrite')

    def __getCityRateFile(self):
        cities = ['Vegas', 'Phoenix', 'Toronto', 'Charlotte', 'Scottsdale', 'Pittsbur',
                  'Montr', 'Mesa', 'Henderson', 'Tempe']
        for city in cities:
            self.__getOneCityRateFile(city)

    def __getOneCityRateFile(self, city):
        cityDF = self.allRateDF.filter(F.lower(self.allRateDF.city).contains(city.lower()))
        cityPath = os.path.join(self.destPath, city)
        print(cityPath)
        cityDF.write.csv(cityPath, 'overwrite')





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reviewFilePath')
    parser.add_argument('userFilePath')
    parser.add_argument('businessFilePath')
    parser.add_argument('fullRateFilePath')
    parser.add_argument('destPath')

    args = parser.parse_args()
    csvFileConverter = FileConverter(args.reviewFilePath, args.userFilePath, args.businessFilePath,
                                     args.fullRateFilePath, args.destPath)
    csvFileConverter.process()




if __name__ == '__main__':
    main()
