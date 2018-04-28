from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import BCD, CABCD
from ridge_dual import BDCD
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

def benchmark_blockSize(sc, A_p, b_p, M, N, l, eps, mus, use_dual, num_partitions):
	legends = []
	for mu in mus:
		if use_dual:
			b = np.array(b_p.collect())
			y_BDCD, x_BDCD, metrics = BDCD(sc, A_p, b, M, N, l, mu, eps)
		else:
			x_BCD, metrics = BCD(sc, A_p, b_p, M, N, l, mu, eps)
		print(metrics)
		residuals = metrics['residual']
		total_iteration = len(residuals)*5
		xs = np.arange(5,total_iteration+1, 5)
		legends.append("block size of "+ str(mu))
		plt.plot(xs, residuals)
	ax = plt.axes()   
	alg = "BCD" if not use_dual else "BDCD"
	plt.title("number of iteration versus residuals for different block size for "+alg)     
	ax.legend(legends)
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Iterations')
	plt.ylabel('Residuals')
	plt.ioff()
	
	plt.savefig(alg+'_iteration_residuals_blocksize.png')




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--use-dual-version", 	required=False, 	default=0, 		help="whether use dual algorithm")
	args = vars(ap.parse_args())

	use_dual = bool(args["use_dual_version"])
	print(use_dual)
	#backend plot
	plt.switch_backend('agg')
	matplotlib.use('TkAgg')

	num_executors = 1
	num_partitions = 1
	conf = SparkConf().setAll([('spark.executor.cores', '1'), ('spark.executor.instances', str(num_executors))])
	sc = SparkContext(conf=conf)
	sc.setCheckpointDir(os.getcwd())
	M = 200
	N = 700
	if use_dual:
		A_p = RandomRDDs.uniformVectorRDD(sc, N, M, numPartitions=num_partitions, seed=1)
	else:
		A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=num_partitions, seed=1)
	b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=num_partitions, seed=2)
	l = 0.5
	eps = 0.1
	mus = [4,8,16,32]
	sc.setLogLevel("ERROR")
	benchmark_blockSize(sc, A_p, b_p, M, N, l, eps, mus, use_dual, num_partitions)

