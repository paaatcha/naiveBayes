#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: André Pacheco
Email: pacheco.comp@gmail.com
"""

import sys

sys.path.append('/home/patcha/Dropbox/Doutorado/Codigos/Python/utils/')

import numpy as np
from dataManipulation import data
from naiveBayes import naiveBayes


# loading the data set
print 'Loading the dataset...'

irisAll = np.genfromtxt('/home/patcha/Datasets/Iris/iris.csv', delimiter=',')



iris = data (dataset=irisAll, percTrain=0.7, percVal=0, percTest=0.3, normType=None, shuf=True, posOut='last', outBin=False)

print iris

nb = naiveBayes(iris.trainIn, iris.trainOut, iris.nClass)
splited = nb.splitByLabel()

stats = nb.statsByLabel (splited)

nb.getResult (stats, iris.testIn, iris.testOut)

