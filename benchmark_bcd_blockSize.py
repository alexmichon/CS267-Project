from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def benchmark_blockSize(sc, A_p, b_p, M, N, l, eps, mus):
	legends = []
	for mu in mus:
		x_BCD, metrics = BCD(sc, A_p, b_p, M, N, l, mu, eps, mu)
		print(metrics)
		residuals = metrics['residual']
		total_iteration = len(residuals)*5
		xs = np.arange(5,total_iteration+1, 5)
		legends.append("line for block size of "+ str(mu))
		plt.plot(xs, residuals)
	ax = plt.axes()   
	plt.title("number of iteration versus residuals for different block size")     
	ax.legend(legends)
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Iterations')
	plt.ylabel('Residuals')
	plt.ioff()
	plt.savefig('iteration_residuals.png')




if __name__ == "__main__":
	plt.switch_backend('agg')
	matplotlib.use('TkAgg')

	numExecutors = 1
	numPartitions = 1
	conf = SparkConf().setAll([('spark.executor.cores', '1'), ('spark.executor.instances', str(numExecutors))])
	print("??")
	sc = SparkContext('local', conf=conf)
	print("hello world")
	sc.setCheckpointDir(os.getcwd())
	print(os.getcwd())
	output_file = open("benchmark_mu.csv", "w")
	output_file.write("mu,execution,sequential,remote,iterations\n")
	output_file.close()
	M = 700
	N = 40
	A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=numPartitions, seed=1)
	b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=numPartitions, seed=2)
	l = 0.5
	eps = 0.1
	mus = [4,8,16,32]
	sc.setLogLevel("ERROR")
	benchmark_blockSize(sc, A_p, b_p, M, N, l, eps, mus)

