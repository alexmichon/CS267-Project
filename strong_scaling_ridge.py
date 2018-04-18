from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def benchmark_nodes(sc, M, N, l, mu, s, eps):
	legends = []
	nodes_num = 2**np.arange(0,5,1)
	execution_times = []
	for node_num in nodes_num:

		print(node_num)
		A = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=int(node_num), seed=1)
		b = RandomRDDs.uniformRDD(sc, M, numPartitions=int(node_num), seed=2)
		x_CABCD, metrics = CABCD(sc, A, b, M, N, l, mu, s, eps)
		print(metrics)
		execution_times.append(metrics['execution'])

	for node_num in nodes_num:

		print(node_num)
		A = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=int(node_num), seed=1)
		b = RandomRDDs.uniformRDD(sc, M, numPartitions=int(node_num), seed=2)
		x_CABCD, metrics = BCD(sc, A, b, M, N, l, mu, eps)
		print("time", metrics['execution'])
		execution_times.append(metrics['execution'])
	
	ax = plt.axes()   
	plt.plot(nodes_num, execution_times)
	plt.title("number of nodes versus execution time")     
	ax.legend(["CABCD with s = 10","BCD"])
	plt.yscale('log')
	#plt.xscale('log')
	plt.xlabel('node number')
	plt.ylabel('execution time')
	plt.ioff()
	plt.savefig('cabcd_node_time.png')




if __name__ == "__main__":
	plt.switch_backend('agg')
	matplotlib.use('TkAgg')

	numExecutors = 1
	numPartitions = 1
	conf = SparkConf().setAll([('spark.executor.cores', '1'), ('spark.executor.instances', str(numExecutors))])
	sc = SparkContext('local', conf=conf)
	sc.setCheckpointDir(os.getcwd())
	output_file = open("benchmark_mu.csv", "w")
	output_file.write("mu,execution,sequential,remote,iterations\n")
	output_file.close()
	M = 2000
	N = 100
	l = 0.5
	eps = 0.1
	mu = 4
	Ss = [3,6,9]
	s = 10
	sc.setLogLevel("ERROR")
	benchmark_nodes(sc, M, N, l, mu, s, eps)