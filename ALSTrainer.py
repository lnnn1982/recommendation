# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import time

class ALSTrainer(object):
    def __init__(self, rateFilePath):
        self.rateFilePath = rateFilePath

        self.spark = SparkSession.builder.appName("ALSModelTraining").getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    def train(self):
        rateSchema = StructType([StructField("userIndex", IntegerType(), True),
                                 StructField("businessIndex", IntegerType(), True),
                                 StructField("city", StringType(), True),
                                 StructField("rate", FloatType(), True)])
        rateDF = self.spark.read.csv(path=self.rateFilePath, schema=rateSchema).select(
            'userIndex', 'businessIndex', 'rate')
        rateDF.cache()

        (training, test) = rateDF.randomSplit([0.8, 0.2])
        print('train for file:' + self.rateFilePath)
        print('training count:{}'.format(training.count()))
        print('test count:{}'.format(test.count()))

        beforeTime = time.time()
        alsModel = self.trainALSModelWithException(training)
        trainTime = time.time() - beforeTime

        rmseList = []
        #trainTimes = []
        testTimes = []
        for i in range(0,20):
            #rmse, trainTime, testTime = self.oneTimeTrain(rateDF)
            rmse, testTime = self.oneTimeTest(alsModel, test)
            print("one time Root-mean-square for test error = " + str(rmse))
            #print("one time train time = " + str(trainTime))
            print("one time test time = " + str(testTime))

            rmseList.append(rmse)
            testTimes.append(testTime)

        meanRmse = self.getMeanValue(rmseList)
        meanTestTime = self.getMeanValue(testTimes)

        print('mean rmse:' + str(meanRmse) + ', mean test time:' + str(meanTestTime)
              + ', mean train time:' + trainTime)

        self.spark.stop()

    def getMeanValue(self, valueList):
        return float(sum(valueList)) / len(valueList)

    def oneTimeTest(self, alsModel, test):
        beforeTime = time.time()
        predictions = alsModel.transform(test)
        testTime = time.time() - beforeTime

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rate",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        return rmse, testTime

    def oneTimeTrain(self, rateDF):
        (training, test) = rateDF.randomSplit([0.8, 0.2])
        print('training count:{}'.format(training.count()))
        print('test count:{}'.format(test.count()))

        beforeTime = time.time()
        alsModel = self.trainALSModelWithException(training)
        trainTime = time.time() - beforeTime

        beforeTime = time.time()
        predictions = alsModel.transform(test)
        testTime = time.time() - beforeTime

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rate",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        return rmse, trainTime, testTime

    def trainALSModelWithException(self, training):
        try:
            return self.trainALSMode('drop', training)
        except:
            print('fit exception happen. use non drop')
            return self.trainALSMode('nan', training)

    def trainALSMode(self, strategy, training):
        als = ALS(maxIter=20, userCol="userIndex", itemCol="businessIndex", ratingCol="rate",
                  coldStartStrategy=strategy, rank=200, regParam=0.1)
        return als.fit(training)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvFilePath')

    args = parser.parse_args()
    trainer = ALSTrainer(args.csvFilePath)
    trainer.train()




if __name__ == '__main__':
    main()






