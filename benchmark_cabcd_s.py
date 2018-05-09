from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import os
from ridge import direct, res_norm, BCD, CABCD
from ridge_dual import BDCD, CABDCD

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json


def benchmark_S(sc, A, b, M, N, l, mu, eps, Ss, use_dual, num_partitions):
	legends = []
	xss = []
	residualss = []
	alg = "BCD" if not use_dual else "BDCD"
	outputFile = open(alg+"_s_residual_random_data.txt", 'a')

	if use_dual:
		b = np.array(b_p.collect())
		y_BDCD, x_BDCD, metrics = BDCD(sc, A, b, M, N, l, mu, eps)
	else:
		x_BDCD, metrics = BCD(sc, A, b, M, N, l, mu, eps)
	residuals = metrics['residual']
	total_iteration = len(residuals)*5
	xs = np.arange(0,total_iteration, 5).tolist()
	legends.append(alg)
	xss.append(xs)
	residualss.append(residuals)
	#plt.plot(xs, residuals)
	for s in Ss:
		if use_dual:
			b = np.array(b_p.collect())
			y_BDCD, x_BDCD, metrics = CABDCD(sc, A, b, M, N, l, mu, s, eps)
		else:
			x_CABCD, metrics = CABCD(sc, A, b, M, N, l, mu, s, eps)
		print(metrics)
		residuals = metrics['residual']
		total_iteration = len(residuals)*5
		alg = "CABCD" if not use_dual else "CABDCD"
		legends.append(alg+" s = "+str(s))
		total_iteration = len(residuals)*5*s
		xs = np.arange(0,total_iteration, 5*s).tolist()
		xss.append(xs)
		residualss.append(residuals)

		#plt.plot(xs, residuals)
	title = "number of iteration versus residuals for different unrolling steps for "+alg
	filename = alg+'_iteration_residuals_s.png'
	xlabel = "Iterations"
	ylabel = "Residuals"
	outDic = {"legends":legends, "xs":xss, "ys": residualss, "title": title, "filename": filename, "xlabel": xlabel, "ylabel":ylabel, "M":M, "N": N, "eps": eps, "lambda": l, "num_executors":4}
	json.dump(outDic, outputFile)
	outputFile.close()
	# ax = plt.axes()   
	# plt.title("number of iteration versus residuals for different unrolling steps for "+alg)     
	# ax.legend(legends)
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.xlabel('Iterations')
	# plt.ylabel('Residuals')
	# plt.ioff()
	# plt.savefig(alg+'_iteration_residuals_s.png')




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--use-dual-version", 	required=False, 	default=0, 		help="whether use dual algorithm")
	args = vars(ap.parse_args())

	use_dual = bool(args["use_dual_version"])

	# plt.switch_backend('agg')
	# matplotlib.use('TkAgg')

	num_executors = 1
	num_partitions = 1
	conf = SparkConf().setAll([('spark.executor.cores', '1'), ('spark.executor.instances', str(num_executors))])
	sc = SparkContext('local', conf=conf)
	sc.setCheckpointDir(os.getcwd())
	M = 200
	N = 20000
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