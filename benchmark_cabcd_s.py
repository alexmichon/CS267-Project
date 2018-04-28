from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
from ridge_dual import BDCD, CABDCD

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse


def benchmark_S(sc, A, b, M, N, l, mu, eps, Ss, use_dual, num_partitions):
	legends = []
	if use_dual:
		b = np.array(b_p.collect())
		y_BDCD, x_BDCD, metrics = BDCD(sc, A, b, M, N, l, mu, eps)
	else:
		x_BDCD, metrics = BCD(sc, A, b, M, N, l, mu, eps)
	residuals = metrics['residual']
	total_iteration = len(residuals)*5
	xs = np.arange(5,total_iteration+1, 5)
	alg = "BCD" if not use_dual else "BDCD"
	legends.append(alg)
	plt.plot(xs, residuals)
	for s in Ss:
		if use_dual:
			b = np.array(b_p.collect())
			y_BDCD, x_BDCD, metrics = CABDCD(sc, A, b, M, N, l, mu, s, eps)
		else:
			x_CABCD, metrics = CABCD(sc, A, b, M, N, l, mu, s, eps)
		print(metrics)
		residuals = metrics['residual']
		total_iteration = len(residuals)*5
		xs = np.arange(5,total_iteration+1, 5)
		legends.append(alg+" s = "+str(s))
		total_iteration = len(residuals)*5*s
		xs = np.arange(5*s,total_iteration+1, 5*s)
		plt.plot(xs, residuals)
	ax = plt.axes()   
	plt.title("number of iteration versus residuals for different unrolling steps for "+alg)     
	ax.legend(legends)
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Iterations')
	plt.ylabel('Residuals')
	plt.ioff()
	plt.savefig(alg+'_iteration_residuals_s.png')




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--use-dual-version", 	required=False, 	default=0, 		help="whether use dual algorithm")
	args = vars(ap.parse_args())

	use_dual = bool(args["use_dual_version"])

	plt.switch_backend('agg')
	matplotlib.use('TkAgg')

	num_executors = 1
	num_partitions = 1
	conf = SparkConf().setAll([('spark.executor.cores', '1'), ('spark.executor.instances', str(num_executors))])
	sc = SparkContext('local', conf=conf)
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
	mu = 4
	Ss = [3,6,9]
	sc.setLogLevel("ERROR")
	benchmark_S(sc, A_p, b_p, M, N, l, mu, eps, Ss, use_dual, num_partitions)