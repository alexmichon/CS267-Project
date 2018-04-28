from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
from ridge_dual import BDCD, CABDCD

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

num_executors = 4
num_partitions = 4
conf = SparkConf().setAll([('spark.executor.cores', '32'), ('spark.executor.instances', str(num_executors))])
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sc.setCheckpointDir(os.getcwd())
M = 700
N = 200
A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=num_partitions, seed=1)
b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=num_partitions, seed=2)
l = 0.5
eps = 0.1
mu = 4
s = 6
x_CABCD, metrics = CABCD(sc, A_p, b_p, M, N, l, mu, s, eps)
print(metrics)