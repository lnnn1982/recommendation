# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from sklearn.metrics.pairwise import cosine_similarity
import math

#((1, 2), ([(1, 1.0), (2, 5.0)], [(2, 2.0), (1, 3.0)]))
#((1, 2), ([(1, 1.0), (2, 5.0), (3, 4.0)], [(2, 2.0), (1, 3.0), (4, 5.0)]))
def calculateSimilarity(x):
    #print("calculateSimilarity x:" + str(x))
    itemRateInfo = x[1]

    item1Map = {}
    item2Map = {}

    map(lambda x1: item1Map.setdefault(x1[0], x1[1]),  itemRateInfo[0])
    map(lambda x1: item2Map.setdefault(x1[0], x1[1]), itemRateInfo[1])

    #print("item1Map:" + str(item1Map))
    #print("item2Map:" + str(item2Map))

    commonUsers = list(set(item1Map.keys()).intersection(set(item2Map.keys())))
    rowsItem1 = []
    rowsItem2 = []
    for user in commonUsers:
        rowsItem1.append(item1Map[user])
        rowsItem2.append(item2Map[user])

    if len(rowsItem1) == 0 or len(rowsItem2) == 0:
        return 0

    simRows1 = []
    simRows1.append(rowsItem1)
    simRows2 = []
    simRows2.append(rowsItem2)

    #print('simRows1:' + str(simRows1))
    #print('simRows1:' + str(simRows2))

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
        #print("calRateFromItem2 error. rateList:" + str(rateList))
        return 0

def cfItemTrain(rateFilePath, neighborNum, type):
    spark = SparkSession.builder.appName("CFItemModel").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    print('filePath:' + rateFilePath)
    print('neighbour number:' + str(neighborNum))
    print('type:' + type)

    rateSchema = StructType([StructField("userID", IntegerType(), True),
                             StructField("itemID", IntegerType(), True),
                             StructField("city", StringType(), True),
                             StructField("rating", FloatType(), True)])
    rateDF = spark.read.csv(path=rateFilePath, schema=rateSchema).select(
        'userID', 'itemID', 'rating')
    (training, test) = rateDF.randomSplit([0.8, 0.2], seed=0)

    rddDf = rateDF.rdd
    if (type == 'test'):
        rddDf = test.rdd
    elif (type == 'train'):
        rddDf = training.rdd

    print("count size:" + str(rddDf.count()))

    itemUserRateRdd = rddDf.map(lambda x: (x.itemID, (x.userID, x.rating)))
    #print("itemUserRateRdd:" + str(itemUserRateRdd.collect()))

    groupItemRdd = itemUserRateRdd.groupByKey().mapValues(list)
    #print("groupItemRdd:" + str(groupItemRdd.collect()))

    #use less than to get half the computation
    cartesianRdd = groupItemRdd.cartesian(groupItemRdd).filter(
        lambda x: x[0] < x[1]).map(lambda x: ((x[0][0], x[1][0]), (x[0][1], x[1][1])))
    #print("cartesianRdd:" + str(cartesianRdd.collect()))

    cosSimilarity = cartesianRdd.map(lambda x: ((x[0][0], x[0][1]), calculateSimilarity(x)))

    #from (2,1,***) get (2,1,***) (1,2,***)
    cosSimilarityPair = cosSimilarity.map(lambda x: [(x[0], x[1]), (
        (x[0][1], x[0][0]), x[1])]).flatMap(lambda x:x)

    #get 10 neighbours
    #(1, [(3, 0.99958619974434737), (4, 0.99860319520601903), (2, 0.95022954097349777)])
    itemSimilarityRdd = cosSimilarityPair.map(lambda x:(x[0][0], (x[0][1], x[1]))).groupByKey(
                    ).mapValues(list).map(lambda x:(x[0], sorted(x[1], key=lambda d:d[1],
                    reverse=True)[0:neighborNum]))
    #print('itemSimilarityRdd:' + str(itemSimilarityRdd.collect()))

    #split item1,use item2 as key
    #item2 (item1 similarity)
    #(4, (1, 0.99860319520601903)), (3, (1, 0.99958619974434737))
    item2Item1RateRdd = itemSimilarityRdd.flatMapValues(lambda x:x).map(lambda x : (x[1][0], (x[0],x[1][1])))
    #print('item2Item1RateRdd:' + str(item2Item1RateRdd.collect()))

    #item1 user item2 similarity rate2
    #((1, 2), (2, 0.95022954097349777, 2.0))
    item1UserItem2InfoRdd = item2Item1RateRdd.join(itemUserRateRdd).map(lambda x: ((x[1][0][0],x[1][1][0]),
                                                                    (x[0],x[1][0][1],x[1][1][1])))
    #print('item1UserItem2InfoRdd:' + str(item1UserItem2InfoRdd.collect()))

    #item1 user rateInfo
    #((4, 2), [(2, 0.96536339302826624, 2.0), (1, 0.99860319520601903, 14.0), (3, 0.99970980846748225, 6.0)])
    item1UserGroupByitem2 = item1UserItem2InfoRdd.groupByKey().mapValues(list)
    #print('item1UserGroupByitem2:' + str(item1UserGroupByitem2.collect()))

    #user item rate
    #((2, 4), 7.3926527181335917)
    userItemPredictRateRdd = item1UserGroupByitem2.map(lambda x:((x[0][1], x[0][0]), calRateFromItem2(x[1])))
    #print('userItemPredictRateRddInfo:' + str(userItemPredictRateRdd.collect()))

    usrUserRageRdd = rddDf.map(lambda x: ((x.userID, x.itemID), x.rating))
    #print("usrUserRageRdd:" + str(usrUserRageRdd.collect()))

    predictRateRdd = usrUserRageRdd.join(userItemPredictRateRdd)
    #print('predictRateRdd' + str(predictRateRdd.collect()))

    rmse = math.sqrt(predictRateRdd.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    print("rmse:" + str(rmse))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvFilePath')
    parser.add_argument('neighborNum')
    parser.add_argument('type')

    args = parser.parse_args()
    cfItemTrain(args.csvFilePath, (int)(args.neighborNum), args.type)

if __name__ == '__main__':
    main()
