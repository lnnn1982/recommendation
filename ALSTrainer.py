# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class ALSTrainer(object):
    def __init__(self, rateFilePath, modelFilePath):
        self.rateFilePath = rateFilePath
        self.modelFilePath = modelFilePath

        self.spark = SparkSession.builder.appName("YelpPreprocessor").getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    def train(self):
        rateSchema = StructType([StructField("userIndex", IntegerType(), True),
                                 StructField("businessIndex", IntegerType(), True),
                                 StructField("city", StringType(), True),
                                 StructField("rate", FloatType(), True)])
        rateDF = self.spark.read.csv(path=self.rateFilePath, schema=rateSchema).cache()
        (training, test) = rateDF.randomSplit([0.8, 0.2])

        als = ALS(maxIter=10, userCol="userIndex", itemCol="businessIndex", ratingCol="rate",
                  coldStartStrategy="drop")
        paramGrid = ParamGridBuilder() \
            .addGrid(als.rank, [20, 50, 100]) \
            .addGrid(als.regParam, [0.1, 0.02]) \
            .build()
        crossval = CrossValidator(estimator=als,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=RegressionEvaluator(metricName="rmse", labelCol="rate",
                                                                predictionCol="prediction"),
                                  numFolds=3,
                                  parallelism=2)
        bestALSModel = crossval.fit(training)
        print('regParam:{}'.format(bestALSModel.bestModel._java_obj.parent().getRegParam()))
        print('rank:{}'.format(bestALSModel.bestModel.rank))

        #print('avgMetrics:' + str(bestALSModel.avgMetrics))

        predictions = bestALSModel.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rate",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square for test error = " + str(rmse))

        bestALSModel.save(self.modelFilePath)

        self.spark.stop()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvFilePath')
    parser.add_argument('modelFilePath')

    args = parser.parse_args()
    trainer = ALSTrainer(args.csvFilePath, args.modelFilePath)
    trainer.train()




if __name__ == '__main__':
    main()






