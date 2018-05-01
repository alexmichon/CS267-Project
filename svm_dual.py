from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import argparse
import math
import numpy as np
import os
import sys
import time




def BDCD(sc, A, b, M, N, l, eps, type=0, max_iters=50):
	count = 0

	A = A.cache()

	x = A.map(lambda e: 0)

	alpha = np.zeros(M)

	if type == 0:
		gamma = 0
		nu = l
	else:
		gamma = 0.5/l
		nu = np.inf

	metrics = {}

	time_execution = 0
	time_sequential = 0
	time_remote = 0

	while count < max_iters:

		iteration_total = 0
		iteration_remote = 0
		iteration_sequential = 0

		iteration_start = time.time()


		iteration_remote_start = time.time()

		# Sub matrix of A
		Ah = A.map(lambda col: col[np.random.RandomState(seed=count).randint(M)]).cache()
		# AA = AA.map(lambda e: (e[0]+1, e[1], e[2], e[3]))

		# Ah T * Ah
		rDotAh = Ah.map(lambda col: np.outer(col, col))
		rAx = Ah.zip(x).map(lambda t: t[0]*t[1])

		# aux = dotAh.zip(rh).map(lambda t: np.concatenate((t[0],np.reshape(t[1], (mu, 1))), axis=1)).sum()
		n, Ax = rDotAh.zip(rAx).reduce(lambda t0, t1: (t0[0]+t1[0], t0[1]+t1[1]))

		iteration_remote += time.time() - iteration_remote_start
		
		iteration_sequential_start = time.time()

		# Select indices
		idx = np.random.RandomState(seed=count).randint(M)

		n += gamma
		g = b[idx]*Ax-1+gamma*alpha[idx]

		gt = abs(min(max(alpha[idx]-g, 0), nu) - alpha[idx])
		
		if gt != 0:
			theta = min(max(alpha[idx] - g/n, 0), nu) - alpha[idx]
		else:
			theta = 0

		alpha[idx] += theta

		iteration_sequential += time.time() - iteration_sequential_start

		iteration_remote_start = time.time()

		dx = Ah.map(lambda col: theta*b[idx]*col)
		x = x.zip(dx).map(lambda t: t[0] + t[1]).cache()

		# Break lineage
		if count % 50:
			x.checkpoint()
			x.count()

		iteration_remote += time.time() - iteration_remote_start

		iteration_total = time.time() - iteration_start

		# Residue

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

	return x.collect(), alpha, metrics






def CABDCD(sc, A, b, M, N, l, S, eps, type=0, max_iters=20):
	count = 0

	A = A.cache()

	x = A.map(lambda e: 0)

	alpha = np.zeros(M)
	thetas = np.zeros(S)

	if type == 0:
		gamma = 0
		nu = l
	else:
		gamma = 0.5/l
		nu = np.inf

	metrics = {}

	time_execution = 0
	time_sequential = 0
	time_remote = 0


	while count < max_iters:

		iteration_total = 0
		iteration_remote = 0
		iteration_sequential = 0

		iteration_start = time.time()


		iteration_remote_start = time.time()

		# Y = A.map(lambda row: row[idx])
		Y = A.map(lambda col: col[np.random.RandomState(seed=count).randint(M, size=S)]).cache()
		# AA = AA.map(lambda e: (e[0]+e[3], e[1], e[2], e[3], e[4])).cache()

		# aux = Y.zip(alpha).map(lambda t: np.concatenate((np.outer(t[0], t[0]), np.reshape(t[0]*t[1], (S*mu, 1))), axis=1)).sum()
		G = Y.map(lambda col: np.outer(col, col))
		Yx = Y.zip(x).map(lambda t: t[0]*t[1])
		G, Yx = G.zip(Yx).reduce(lambda t0, t1: (t0[0]+t1[0], t0[1]+t1[1]))

		iteration_remote += time.time() - iteration_remote_start
		
		iteration_sequential_start = time.time()

		idx = np.random.RandomState(seed=count).randint(M, size=S)

		
		n = np.diag(G)+gamma

		for j in range(S):
			beta = alpha[idx[j]]
			for t in range(j):
				if idx[t] == idx[j]:
					beta += thetas[t]

			g = b[idx[j]]*Yx[j]-1+gamma*beta
			for t in range(j):
				g += thetas[t]*b[idx[j]]*b[idx[t]]*G[j,t]

			gt = abs(min(max(beta-g, 0), nu) - beta)
			if gt != 0:
				thetas[j] = min(max(beta-g/n[j], 0), nu)-beta
			else:
				thetas[j] = 0

		for j in range(S):
			alpha[idx[j]] += thetas[j]

		dx = Y.map(lambda row: np.dot(row, np.multiply(thetas, b[idx])))
		x = x.zip(dx).map(lambda t: t[0]+t[1])


		# Break lineage
		if count % 50:
			x.checkpoint()
			x.count()

		iteration_remote += time.time() - iteration_remote_start

		iteration_total = time.time() - iteration_start

		# Residue

		# if count/S % 5 == 0:

		# 	residual_start = time.time()

		# 	# A*x
		# 	Ax = A.map(lambda row: np.dot(x, row))

		# 	# AT*A*x
		# 	ATAx = A.zip(Ax).map(lambda t: t[0] * t[1]).sum()

		# 	# AT*b
		# 	ATb = A.zip(b).map(lambda t: t[0]*t[1]).sum()

		# 	# r = AT*b - AT*A*x
		# 	r = ATb-ATAx-l*x

		# 	norm_r = np.linalg.norm(r)
		# 	if norm_r < eps:
		# 		break

		time_sequential += iteration_sequential
		time_remote += iteration_remote
		time_execution += iteration_total
		count += S

		# print("Iteration ", count, " took ", iteration_total, "s (sequential: ", iteration_sequential, "s remote: ", iteration_remote, "s)")

	metrics['execution'] = time_execution
	metrics['sequential'] = time_sequential
	metrics['remote'] = time_remote
	metrics['iterations'] = count

	return x.collect(), alpha, metrics









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


	A_p = RandomRDDs.uniformVectorRDD(sc, N, M, numPartitions=numPartitions, seed=1)
	b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=numPartitions, seed=2)

	A = np.array(A_p.collect()).T
	b = np.array(b_p.collect())

	l = 0.5

	x_BDCD, alpha_BDCD, metrics_BDCD = BDCD(sc, A_p, b, M, N, l, eps)
	print("BDCD")
	print(x_BDCD)
	# print("Resnorm: ", res_norm(A, b, x_BDCD, l))
	print("Execution Time: ", metrics_BDCD['execution'])
	print("Sequential Time: ", metrics_BDCD['sequential'])
	print("Remote Time: ", metrics_BDCD['remote'])
	print("Iterations: ", metrics_BDCD['iterations'])
	print("\n")

	x_CABDCD, alpha_CABDCD, metrics_CABDCD = CABDCD(sc, A_p, b, M, N, l, S, eps)
	print("CABDCD")
	print(x_CABDCD)
	# print("Resnorm: ", res_norm(A, b, x_CABCD, l))
	print("Execution Time: ", metrics_CABDCD['execution'])
	print("Sequential Time: ", metrics_CABDCD['sequential'])
	print("Remote Time: ", metrics_CABDCD['remote'])
	print("Iterations: ", metrics_CABDCD['iterations'])
	print("\n")

