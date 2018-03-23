# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from sklearn.metrics.pairwise import cosine_similarity

#((1, 2), ([(1, 1.0), (2, 5.0)], [(2, 2.0), (1, 3.0)]))
#((1, 2), ([(1, 1.0), (2, 5.0), (3, 4.0)], [(2, 2.0), (1, 3.0), (4, 5.0)]))
def calculateSimilarity(x):
    itemRateInfo = x[1]
    #print(itemRateInfo)
    item1Map = {}
    item2Map = {}

    map(lambda x1: item1Map.setdefault(x1[0], x1[1]),  itemRateInfo[0])
    map(lambda x1: item2Map.setdefault(x1[0], x1[1]), itemRateInfo[1])

    #print(item1Map)
    #print(item2Map)

    commonUsers = list(set(item1Map.keys()).intersection(set(item2Map.keys())))
    rowsItem1 = []
    rowsItem2 = []
    for user in commonUsers:
        rowsItem1.append(item1Map[user])
        rowsItem2.append(item2Map[user])

    simRows1 = []
    simRows1.append(rowsItem1)
    simRows2 = []
    simRows2.append(rowsItem2)

    #print(simRows1)
    #print(simRows2)

    return cosine_similarity(simRows1, simRows2)[0][0]

#[(2, 0.96536339302826624, 2.0), (1, 0.99860319520601903, 14.0), (3, 0.99970980846748225, 6.0)]
def calRateFromItem2(rateList):
    simRateSum = 0
    simSum = 0
    for oneRate in rateList:
        simRateSum = simRateSum + (oneRate[1]*oneRate[2])
        simSum = simSum + oneRate[1]

    if simSum != 0:
        return simRateSum / simSum
    else:
        print("calRateFromItem2 error")
        return 0

def cfItemTrain(rateFilePath):
    spark = SparkSession.builder.appName("CFItemModel").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    rateSchema = StructType([StructField("userIndex", IntegerType(), True),
                             StructField("businessIndex", IntegerType(), True),
                             StructField("city", StringType(), True),
                             StructField("rate", FloatType(), True)])
    rateDF = spark.read.csv(path=rateFilePath, schema=rateSchema).select(
        'userIndex', 'businessIndex', 'rate')
    (training, test) = rateDF.randomSplit([0.8, 0.2], seed=0)

    rddDf = training.rdd
    itemUserRateRdd = rddDf.map(lambda x: (x.itemID, (x.userID, x.rating)))
    groupItemRdd = itemUserRateRdd.groupByKey().mapValues(list)

    #use less than to get half the computation
    cartesianRdd = groupItemRdd.cartesian(groupItemRdd).filter(
        lambda x: x[0] < x[1]).map(lambda x: ((x[0][0], x[1][0]), (x[0][1], x[1][1])))

    cosSimilarity = cartesianRdd.map(lambda x: ((x[0][0], x[0][1]), calculateSimilarity(x)))

    #from (2,1,***) get (2,1,***) (1,2,***)
    cosSimilarityPair = cosSimilarity.map(lambda x: [(x[0], x[1]), (
        (x[0][1], x[0][0]), x[1])]).flatMap(lambda x:x)

    #get 10 neighbours
    itemSimilarityRdd =  cosSimilarityPair.map(lambda x:(x[0][0], (x[0][1], x[1]))).groupByKey(
                    ).mapValues(list).map(lambda x:(x[0], sorted(x[1], key=lambda d:d[0], reverse=True)[0:10]))

    #split item1,use item2 as key
    #item2 (item1 similarity)
    #(4, (1, 0.99860319520601903)), (3, (1, 0.99958619974434737))
    item2Item1RateRdd = itemSimilarityRdd.flatMapValues(lambda x:x).map(lambda x : (x[1][0], (x[0],x[1][1])))

    #item1 user item2 similarity rate2
    #((1, 2), (2, 0.95022954097349777, 2.0))
    item1UserItem2InfoRdd = item2Item1RateRdd.join(itemUserRateRdd).map(lambda x: ((x[1][0][0],x[1][1][0]),
                                                                    (x[0],x[1][0][1],x[1][1][1])))

    #item1 user rateInfo
    #((4, 2), [(2, 0.96536339302826624, 2.0), (1, 0.99860319520601903, 14.0), (3, 0.99970980846748225, 6.0)])
    item1UserGroupByitem2 = item1UserItem2InfoRdd.groupByKey().mapValues(list)

    #user item rate
    #(2, 4, 7.3926527181335917)
    userItemPredictRateRdd = item1UserGroupByitem2.map(lambda x:(x[0][1], x[0][0], calRateFromItem2(x[1])))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvFilePath')

    args = parser.parse_args()
    cfItemTrain(args.csvFilePath)




if __name__ == '__main__':
    main()
