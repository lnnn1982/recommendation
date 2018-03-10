# -*- coding: utf-8 -*-


from __future__ import print_function

import argparse
import json
import csv

def convertReviewFile(jsonFilePath, csvFilePath):
    with open(csvFilePath, 'w') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(['userID', 'businessID', 'rate'])

        with open(jsonFilePath, 'r') as fin:
            for line in fin:
                jsonContent = json.loads(line, strict=False)
                userID = jsonContent["user_id"]
                businessID = jsonContent["business_id"]
                rate = jsonContent["stars"]

                csv_file.writerow([userID, businessID, rate])

def converUserFile(jsonFilePath, csvFilePath):
    with open(csvFilePath, 'w') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(['userID', 'userIndex'])

        index = 0
        with open(jsonFilePath, 'r') as fin:
            for line in fin:
                jsonContent = json.loads(line, strict=False)
                userID = jsonContent["user_id"]

                csv_file.writerow([userID, index])
                index += 1

def converBusinessFile(jsonFilePath, csvFilePath):
    with open(csvFilePath, 'w') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(['businessID', 'businessIndex', 'city'])

        index = 0

        with open(jsonFilePath, 'r') as fin:
            for line in fin:
                jsonContent = json.loads(line, strict=False)
                businessID = jsonContent["business_id"]
                city = jsonContent["city"]

                csv_file.writerow([businessID, index, city.encode('utf-8')])
                index += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonFilePath')
    parser.add_argument('processType')
    args = parser.parse_args()

    jsonFilePath = args.jsonFilePath
    processType = args.processType

    if processType == 'review':
        convertReviewFile(jsonFilePath, 'review.csv')
    elif processType == 'user':
        converUserFile(jsonFilePath, 'user.csv')
    elif processType == 'business':
        converBusinessFile(jsonFilePath, 'business.csv')


if __name__ == '__main__':
    main()









