from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def benchmark_S(sc, A, b, M, N, l, mu, eps, Ss):
	legends = []
	x_BCD, metrics = BCD(sc, A, b, M, N, l, mu, eps)
	residuals = metrics['residual']
	total_iteration = len(residuals)*5
	xs = np.arange(5,total_iteration+1, 5)
	legends.append("BCD")
	plt.plot(xs, residuals)
	for s in Ss:
		x_CABCD, metrics = CABCD(sc, A, b, M, N, l, mu, s, eps)
		print(metrics)
		residuals = metrics['residual']
		total_iteration = len(residuals)*5
		xs = np.arange(5,total_iteration+1, 5)
		legends.append("CABCD s = "+str(s))
		total_iteration = len(residuals)*5*s
		xs = np.arange(5*s,total_iteration+1, 5*s)
		plt.plot(xs, residuals)
	ax = plt.axes()   
	plt.title("number of iteration versus residuals for different unrolling steps")     
	ax.legend(legends)
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Iterations')
	plt.ylabel('Residuals')
	plt.ioff()
	plt.savefig('cabcd_iteration_residuals_s.png')




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
	A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=numPartitions, seed=1)
	b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=numPartitions, seed=2)
	l = 0.5
	eps = 0.1
	mu = 4
	Ss = [3,6,9]
	sc.setLogLevel("ERROR")
	benchmark_S(sc, A_p, b_p, M, N, l, mu, eps, Ss)