from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

def benchmark_nodes(M, N, l, mu, eps, node_num):
	legends = []

	ss = [2,4,8,16,32]
	execution_times = []
	iterationss = []
	appIDs = []
	appNames = []

	for s in ss:
		appName = 'cabcd breakdown test s='+str(s) +"nodes_num: "+str(node_num)
		conf = SparkConf().setAll([('spark.executor.cores', '16'), ('spark.executor.instances', str(node_num)), ('spark.app.name', appName)])
		sc = SparkContext(conf=conf)
		sc.setLogLevel("ERROR")
		appIDs.append(sc.applicationId)
		appNames.append(appName)
		sc.setCheckpointDir(os.getcwd())
		print(s)
		A = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=int(node_num), seed=1)
		b = RandomRDDs.uniformRDD(sc, M, numPartitions=int(node_num), seed=2)
		x_CABCD, metrics = CABCD(sc, A, b, M, N, l, mu, s, eps)
		sc.stop()
		print(metrics)
		execution_times.append(metrics['execution'])
		iterationss.append(metrics['iterations'])

	outputFile = open("cabcd_breakdown_logsID.txt","a") 
	
	outputDic = {"appIDs": appIDs, "appNames": appNames, "ss": ss, "iterationss": iterationss, "M": M, "N": N, "lambda": l, "mu": mu, "node_num": node_num}
	json.dump(outputDic, outputFile)
	outputFile.write("\n")
	outputFile.close()


if __name__ == "__main__":

	plt.switch_backend('agg')
	matplotlib.use('TkAgg')

	M = 20000
	N = 200
	l = 0.5
	eps = 0.1
	mu = 16
	node_num = 32
	benchmark_nodes(M, N, l, mu, eps, node_num)