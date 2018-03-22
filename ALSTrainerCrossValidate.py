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

        print('rateFilePath:' + rateFilePath)
        print('modelFilePath' + modelFilePath)

        self.spark = SparkSession.builder.appName("ALSModelTraining").getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    def train(self):
        rateSchema = StructType([StructField("userIndex", IntegerType(), True),
                                 StructField("businessIndex", IntegerType(), True),
                                 StructField("city", StringType(), True),
                                 StructField("rate", FloatType(), True)])
        rateDF = self.spark.read.csv(path=self.rateFilePath, schema=rateSchema).select(
            'userIndex', 'businessIndex', 'rate')
        (training, test) = rateDF.randomSplit([0.8, 0.2], seed=0)
        training.cache()
        print('training count:{}'.format(training.count()))
        test.cache()
        print('test count:{}'.format(test.count()))

        bestALSModel = self.trainALSModelWithException(training)
        print('regParam:{}'.format(bestALSModel.bestModel._java_obj.parent().getRegParam()))
        print('rank:{}'.format(bestALSModel.bestModel.rank))

        #print('avgMetrics:' + str(bestALSModel.avgMetrics))

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rate",
                                        predictionCol="prediction")

        predictions = bestALSModel.transform(training)
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square for training error = " + str(rmse))

        predictions = bestALSModel.transform(test)
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square for test error = " + str(rmse))

        print('save model modelFilePath:' + self.modelFilePath)
        bestALSModel.write().overwrite().save(self.modelFilePath)

        self.spark.stop()

    def trainALSModelWithException(self, training):
        try:
            return self.trainALSMode('drop', training)
        except:
            print('fit exception happen. use non drop')
            return self.trainALSMode('nan', training)

    def trainALSMode(self, strategy, training):
        als = ALS(maxIter=20, userCol="userIndex", itemCol="businessIndex", ratingCol="rate",
                  coldStartStrategy=strategy)
        paramGrid = ParamGridBuilder() \
            .addGrid(als.rank, [50, 100, 200]) \
            .addGrid(als.regParam, [0.1, 0.02]) \
            .build()
        crossval = CrossValidator(estimator=als, estimatorParamMaps=paramGrid,
                                  evaluator=RegressionEvaluator(metricName="rmse", labelCol="rate",
                                                                predictionCol="prediction"), numFolds=3,
                                  parallelism=4)
        return crossval.fit(training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvFilePath')
    parser.add_argument('modelFilePath')

    args = parser.parse_args()
    trainer = ALSTrainer(args.csvFilePath, args.modelFilePath)
    trainer.train()




if __name__ == '__main__':
    main()






