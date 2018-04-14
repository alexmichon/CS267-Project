from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import argparse
import math
import numpy as np
import os
import sys
import time




def BCD(sc, A, b, M, N, l, mu, eps, max_iters=10):
	count = 0

	A = A.cache()
	b = b.cache()

	y = np.zeros(N)
	yt = A.map(lambda row: np.dot(row, y)).cache()
	yt.collect()

	z = np.zeros(N)
	zt = A.zip(b).map(lambda t: np.dot(t[0], z)-t[1]).cache()
	zt.collect()

	metrics = {}

	time_execution = 0
	time_sequential = 0
	time_remote = 0

	AA = A.map(lambda row: (0, N, mu, row)).cache()

	theta = float(mu)/N
	q = int(N/mu)

	while count < max_iters:

		iteration_total = 0
		iteration_remote = 0
		iteration_sequential = 0

		iteration_start = time.time()


		iteration_remote_start = time.time()

		# Sub matrix of A
		Ah = AA.map(lambda e: e[3][np.random.RandomState(seed=count).permutation(e[1])[:e[2]]]).cache()
		# AA = AA.map(lambda e: (e[0]+1, e[1], e[2], e[3]))

		# Ah T * Ah
		rDotAh = Ah.map(lambda row: np.outer(row, row))
		rRh = Ah.zip(yt).zip(zt).map(lambda t: t[0][0]*(theta*theta*t[0][1]+t[1]))

		# aux = dotAh.zip(rh).map(lambda t: np.concatenate((t[0],np.reshape(t[1], (mu, 1))), axis=1)).sum()
		dotAh, rh = rDotAh.zip(rRh).reduce(lambda t0, t1: (t0[0]+t1[0], t0[1]+t1[1]))

		iteration_remote += time.time() - iteration_remote_start
		
		iteration_sequential_start = time.time()

		# Select indices
		idx = np.random.RandomState(seed=count).permutation(N)[:mu]

		v = max(np.linalg.eigvals(dotAh))
		n = 1/(q*theta*v)
		g = z[idx] - n*rh
		dz = np.sign(g)*np.array([max(abs(e)-l*n, 0) for e in g]) - z[idx]
		
		z[idx] = z[idx] + dz
		y[idx] = y[idx] - (1-q*theta)/theta/theta*dz

		iteration_sequential += time.time() - iteration_sequential_start

		iteration_remote_start = time.time()

		dz_bcast = sc.broadcast(dz)


		dzt = Ah.map(lambda row: np.dot(row, dz_bcast.value)).cache()
		zt = zt.zip(dzt).map(lambda t: t[0] + t[1]).cache()
		dyt = Ah.map(lambda row: (1-q*theta)/theta/theta*np.dot(row, dz_bcast.value)).cache()
		yt = yt.zip(dyt).map(lambda t: t[0] - t[1]).cache()

		theta = 0.5*(math.sqrt(math.pow(theta,4)+4*math.pow(theta, 2)) - math.pow(theta, 2))


		# Break lineage
		if count % 50:
			zt.checkpoint()
			zt.count()
			yt.checkpoint()
			yt.count()

		iteration_remote += time.time() - iteration_remote_start

		iteration_total = time.time() - iteration_start

		Residue

		# if count % 5 == 0:
		# 	residual_start = time.time()

		# 	# A*x
		# 	Ax = A.map(lambda row: np.dot(x, row))

		# 	# AT*A*x
		# 	ATAx = A.zip(Ax).map(lambda t: np.dot(t[0], t[1])).sum()

		# 	# AT*b
		# 	ATb = A.zip(b).map(lambda t: np.dot(t[0], t[1])).sum()

		# 	# r = AT*b - AT*A*x
		# 	r = ATb-ATAx-l*x

		# 	norm_r = np.linalg.norm(r)
		# 	if norm_r < eps:
		# 		break	


		time_sequential += iteration_sequential
		time_remote += iteration_remote
		time_execution += iteration_total
		count += 1

		# print("Iteration ", count, " took ", iteration_total, "s (sequential: ", iteration_sequential, "s remote: ", iteration_remote, "s)")

	metrics['execution'] = time_execution
	metrics['sequential'] = time_sequential
	metrics['remote'] = time_remote
	metrics['iterations'] = count

	return theta*theta*y+z, metrics






def res_norm(A, b, x, l):
	return np.linalg.norm(np.dot(A, x)-b,2)+l*np.linalg.norm(x,1)




if __name__ == "__main__":


	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--num-partitions", 	required=False, 	default=1, 		help="number of partitions")
	ap.add_argument("-e", "--num-executors", 	required=False, 	default=1, 		help="number of executor instances")
	ap.add_argument("-M", "--matrix-rows", 		required=False, 	default=10, 	help="number of rows in matrix")
	ap.add_argument("-N", "--matrix-cols", 		required=False, 	default=10, 	help="number of cols in matrix")	
	ap.add_argument("-u", "--block-size", 		required=False, 	default=1, 		help="block size")
	ap.add_argument("-S", "--group-iterations", required=False, 	default=1, 		help="number of grouped iterations")
	ap.add_argument("-p", "--precision", 		required=False, 	default=0.1,	help="precision")
	args = vars(ap.parse_args())


	numPartitions = int(args["num_partitions"])
	numExecutors = int(args["num_executors"])
	M = int(args["matrix_rows"])
	N = int(args["matrix_cols"])
	mu = int(args["block_size"])
	S = int(args["group_iterations"])
	eps = float(args["precision"])


	conf = SparkConf().setAll([('spark.executor.cores', '1'), ('spark.executor.instances', str(numExecutors))])
	sc = SparkContext(conf=conf)
	sc.setCheckpointDir(os.getcwd())


	A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=numPartitions, seed=1)
	b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=numPartitions, seed=2)

	A = np.array(A_p.collect())
	b = np.array(b_p.collect())

	l = 0.5

	x_BCD, metrics_BCD = BCD(sc, A_p, b_p, M, N, l, mu, eps)
	print("BCD")
	print(x_BCD)
	print("Resnorm: ", res_norm(A, b, x_BCD, l))
	print("Execution Time: ", metrics_BCD['execution'])
	print("Sequential Time: ", metrics_BCD['sequential'])
	print("Remote Time: ", metrics_BCD['remote'])
	print("Iterations: ", metrics_BCD['iterations'])
	print("\n")

