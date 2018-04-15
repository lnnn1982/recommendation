# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from pyspark.sql import SparkSession
#from pyspark.sql.types import *
from pyspark.ml.fpm import FPGrowth
import pymysql.cursors
import random

class FrequentPatternMineTrainer(object):
    def __init__(self, minSupport, minConfidence):
        self.spark = SparkSession.builder.appName("FrequentPatternMineTrainer").getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.minSupport = minSupport
        self.minConfidence = minConfidence
        self.dbData = []

    def readFromDB(self):
        db = pymysql.connect("localhost", "root", "XXXXXXX", "yelp_db", charset ='utf8')
        cursor = db.cursor()
        sql = "select A.business_id, A.user_id from review A join business B " \
              "on A.business_id = B.id where B. city = 'Montréal' and B.is_open = 1 " \
              " group by A.business_id, A.user_id "

        count = cursor.execute(sql)
        print('db row count:' + str(count))

        intUsrId = 0
        intBusinessId = 0

        userIdMap = {}
        businessIdMap = {}
        while True:
            bulkDbData = cursor.fetchmany(1000)
            if bulkDbData == ():
                break

            for onerow in bulkDbData:
                oneRowList = []
                businessID = onerow[0]
                userId = onerow[1]
                if userId in userIdMap:
                    oneRowList.append(userIdMap[userId])
                else:
                    userIdMap[userId] = intUsrId
                    oneRowList.append(intUsrId)
                    intUsrId += 1

                if businessID in businessIdMap:
                    oneRowList.append(businessIdMap[businessID])
                else:
                    businessIdMap[businessID] = intBusinessId
                    oneRowList.append(intBusinessId)
                    intBusinessId += 1

                self.dbData.append(oneRowList)

        print("dbData size:{}, user size:{}, business size:{}".format(len(self.dbData),
                                                    len(userIdMap), len(businessIdMap)))
        #print(self.dbData)
        db.close()

    def train(self):
        trainDataList, testDataList = self.doRandomSplitData(self.dbData)
        print("random split. input list size:{}, train size:{}, test size:{}".format(len(self.dbData),
                                                                len(trainDataList), len(testDataList)))
        trainUsrItemMap = self.getItemsForUsr(trainDataList)
        testUsrItemMap = self.getItemsForUsr(testDataList)
        print('trainUsrItemMap len:' + str(len(trainUsrItemMap)) + ", testUsrItemMap:" + str(len(testUsrItemMap)))

        trainDf = self.spark.createDataFrame(trainUsrItemMap.items(), ["id", "items"])
        trainDf.cache()

        fpGrowth = FPGrowth(itemsCol="items", minSupport=self.minSupport, minConfidence=self.minConfidence)
        fgModel = fpGrowth.fit(trainDf)

        associateRules = fgModel.associationRules.collect()
        antecedentTmpList = [value['antecedent'] for value in associateRules]
        antecedentList = []
        [antecedentList.append(i) for i in antecedentTmpList if not i in antecedentList]
        print('associateRules len:', len(associateRules), ', antecedentList len:', len(antecedentList))

        freqItemsets = fgModel.freqItemsets.collect()
        print('freqItemsets len:', len(freqItemsets))

        antecedentPredictionList = self.transformAllAntecdents(antecedentList, fgModel)
        print('antecedentPredictionList size:', len(antecedentPredictionList))
        usrPredictionMap = self.predictForUsers(antecedentPredictionList, trainUsrItemMap)
        print('usrPredictionMap len:' + str(len(usrPredictionMap)))

        totalTP, totalFP, totalFN = self.getTestPrecionAndRecall(usrPredictionMap, testUsrItemMap)
        precision = float(totalTP)/float(totalTP + totalFP)
        recall = float(totalTP) / float(totalTP + totalFN)
        print('precision:', precision, ", recall:", recall)

    def transformAllAntecdents(self, antecedentList, fgModel):
        antecedentPredictionList = []

        for antecdent in antecedentList:
            df = self.spark.createDataFrame([(0, antecdent)], ["id", "items"])
            fitRows = fgModel.transform(df).collect()
            predictions = [value['prediction'] for value in fitRows if value['prediction'] != []]
            if predictions == []:
                continue
            predictionElems = reduce(lambda x, y: x + y, predictions)
            antecedentPredictionList.append((antecdent, predictionElems))

        return antecedentPredictionList

    def predictForUsers(self, antecedentPredictionList, trainUsrItemMap):
        usrPredictionMap = {}
        for usrId, items in trainUsrItemMap.iteritems():
            predictionElems = self.getPredictItemsForOneUser(items, antecedentPredictionList)
            predictItems = list(set(predictionElems).difference(set(items)))
            if predictItems != []:
                usrPredictionMap[usrId] = predictItems
        return usrPredictionMap

    def getPredictItemsForOneUser(self, orgItemList, antecedentPredictionList):
        predictionElems = []
        for antecedentPrediction in antecedentPredictionList:
            if set(antecedentPrediction[0]).intersection(set(orgItemList)) == set(antecedentPrediction[0]):
                predictionElems += antecedentPrediction[1]

        return predictionElems

    def getTestPrecionAndRecall(self, usrPredictionMap, testUsrItemMap):
        maxMatchNum = 0
        totalTP = 0
        totalFP = 0
        totalFN = 0
        for usr, items in testUsrItemMap.iteritems():
            if usr not in usrPredictionMap:
                continue

            predictions = usrPredictionMap[usr]
            numTP = len(set(predictions).intersection(set(items)))
            numFP = len(set(predictions).difference(set(items)))
            numFN = len(set(items).difference(set(predictions)))

            totalTP += numTP
            totalFP += numFP
            totalFN += numFN

            maxMatchNum += 1

        print ('maxMatchNum:', maxMatchNum, ", totalTP:", totalTP, ", totalFP:", totalFP, ", totalFN:", totalFN)
        return totalTP, totalFP, totalFN

    def getItemsForUsr(self, inputDataList):
        usrItemsMap = {}
        maxLen = 0
        for oneList in inputDataList:
            usrId = oneList[0]
            businessId = oneList[1]
            if usrId in usrItemsMap:
                usrItemsMap[usrId].append(businessId)
                if(len(usrItemsMap[usrId]) > maxLen):
                    maxLen = len(usrItemsMap[usrId])
            else:
                usrItemsMap[usrId] = [businessId]
        #print('maxLen:' + str(maxLen))
        return usrItemsMap

    def doRandomSplitData(self, inputList):
        #return inputList, inputList

        allIds = [i for i in range(len(inputList))]
        test_ids = random.sample(allIds, int(len(allIds) * 0.1))
        testData = []
        trainData = []
        for j in range(len(inputList)):
            if j in test_ids:
                testData.append(inputList[j])
            else:
                trainData.append(inputList[j])
        return trainData, testData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('minSupport')
    parser.add_argument('minConfidence')

    args = parser.parse_args()
    trainer = FrequentPatternMineTrainer((float)(args.minSupport), (float)(args.minConfidence))
    trainer.readFromDB()
    trainer.train()


if __name__ == '__main__':
    main()
