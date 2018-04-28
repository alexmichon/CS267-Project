from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from lasso import res_norm, BCD, CABCD
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def benchmark_nodes(M, N, l, mu, s, eps):
	legends = []
	nodes_num = 2**np.arange(0,5,1)
	nodes_num = np.append(nodes_num, 31)

	execution_times = []


	for node_num in nodes_num:
		conf = SparkConf().setAll([('spark.executor.cores', '16'), ('spark.executor.instances', str(node_num))])
		sc = SparkContext(conf=conf)
		sc.setLogLevel("ERROR")
		sc.setCheckpointDir(os.getcwd())
		
		print(node_num)
		A = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=int(node_num), seed=1)
		b = RandomRDDs.uniformRDD(sc, M, numPartitions=int(node_num), seed=2)
		x_CABCD, metrics = BCD(sc, A, b, M, N, l, mu, eps)
		sc.stop()
		print("time", metrics['execution'])
		execution_times.append(metrics['execution'])
	
	output = open("strong_scaling_lasso.txt","a") 
	output.write("\n")
	output.write("s: " +str(s)+"\n" + "M: " +str(M)+"\n"+"N: " +str(N)+"\n" )
	output.write("BCD\n")
	output.write(', '.join(str(e) for e in nodes_num)+"\n")
	output.write(', '.join(str(e) for e in execution_times)+"\n")

	execution_times = []

	for node_num in nodes_num:
		conf = SparkConf().setAll([('spark.executor.cores', '16'), ('spark.executor.instances', str(node_num))])
		sc = SparkContext(conf=conf)
		sc.setLogLevel("ERROR")
		sc.setCheckpointDir(os.getcwd())
		print(node_num)
		A = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=int(node_num), seed=1)
		b = RandomRDDs.uniformRDD(sc, M, numPartitions=int(node_num), seed=2)
		x_CABCD, metrics = CABCD(sc, A, b, M, N, l, mu, s, eps)
		sc.stop()
		print(metrics)
		execution_times.append(metrics['execution'])

	output.write("CABCD\n")
	output.write(', '.join(str(e) for e in nodes_num)+"\n")
	output.write(', '.join(str(e) for e in execution_times)+"\n")
	output.close()


if __name__ == "__main__":

	plt.switch_backend('agg')
	matplotlib.use('TkAgg')

	M = 40000
	N = 200
	l = 0.5
	eps = 0.1
	mu = 16
	s = 16
	benchmark_nodes(M, N, l, mu, s, eps)