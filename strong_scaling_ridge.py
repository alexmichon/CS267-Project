from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def benchmark_nodes(M, N, l, mu, s, eps):
	legends = []
	nodes_num = 2**np.arange(0,5,1)
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

	# for node_num in nodes_num:
	# 	conf = SparkConf().setAll([('spark.executor.cores', '1'), ('spark.executor.instances', str(node_num))])
	# 	sc = SparkContext(conf=conf)
	# 	sc.setLogLevel("ERROR")
	# 	sc.setCheckpointDir(os.getcwd())
		
	# 	print(node_num)
	# 	A = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=int(node_num), seed=1)
	# 	b = RandomRDDs.uniformRDD(sc, M, numPartitions=int(node_num), seed=2)
	# 	x_CABCD, metrics = BCD(sc, A, b, M, N, l, mu, eps)
	# 	sc.stop()
	# 	print("time", metrics['execution'])
	# 	execution_times.append(metrics['execution'])
	
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

	M = 10000
	N = 500
	l = 0.5
	eps = 0.1
	mu = 4
	s = 10
	benchmark_nodes(M, N, l, mu, s, eps)