from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import argparse
import numpy as np
import os
import sys
import time






def direct(A, b, M, N, l):
	Z = np.linalg.inv((1/l)*np.dot(A, A.T) + np.eye(M))
	return np.dot(Z, b)


def BDCD(sc, A, b, M, N, l, mu, eps, max_iters=50):
	count = 0

	A = A.cache()

	x = sc.parallelize(np.zeros(N), numSlices=A.getNumPartitions()).cache()
	alpha = np.zeros(M)

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
		Ah = A.map(lambda col: col[np.random.RandomState(seed=count).permutation(M)[:mu]]).cache()

		# Ah T * Ah
		dotAh = Ah.map(lambda col: np.outer(col, col))
		rh = Ah.zip(x).map(lambda t: t[0]*t[1])

		# aux = dotAh.zip(rh).map(lambda t: np.concatenate((t[0],np.reshape(t[1], (mu, 1))), axis=1)).sum()
		dotAh, rh = dotAh.zip(rh).reduce(lambda t0, t1: (t0[0]+t1[0], t0[1]+t1[1]))

		iteration_remote += time.time() - iteration_remote_start
		
		iteration_sequential_start = time.time()

		# Select indices
		idx = np.random.RandomState(seed=count).permutation(M)[:mu]


		dotAh = (1/l)*dotAh + np.eye(mu)
		rh = -rh + alpha[idx] - b[idx]


		# (Ah T * Ah)^-1
		invDotAh = np.linalg.inv(dotAh)
		# for i in range(mu):
		# 	rh[i] = rh[i] - l*x[idx[i]]

		dalphah = -np.dot(invDotAh,rh)

		# x = x + xhat
		# for i in range(mu):
		# 	alpha[idx[i]] += dalphah[i]
		alpha[idx] += dalphah

		iteration_sequential += time.time() - iteration_sequential_start

		iteration_remote_start = time.time()

		dalphah_bcast = sc.broadcast(dalphah)

		dx = Ah.map(lambda col: np.dot(col, dalphah_bcast.value))
		x = x.zip(dx).map(lambda t: t[0] - (1/l)*t[1])

		# Break lineage
		if count % 50:
			x.checkpoint()
			x.count()


		iteration_remote += time.time() - iteration_remote_start

		iteration_total = time.time() - iteration_start

		# Residue

		if count % 5 == 0:
			residual_start = time.time()

			# A*x
			ATalpha = A.map(lambda col: np.dot(alpha, col))

			# AT*A*x
			AATalpha = A.zip(ATalpha).map(lambda t: np.dot(t[0], t[1])).sum()

			# r = AT*b - AT*A*x
			r = (1/l)*AATalpha+alpha-b

			norm_r = np.linalg.norm(r)
			if norm_r < eps:
				break

		time_sequential += iteration_sequential
		time_remote += iteration_remote
		time_execution += iteration_total
		count += 1

	metrics['execution'] = time_execution
	metrics['sequential'] = time_sequential
	metrics['remote'] = time_remote
	metrics['iterations'] = count

	return alpha, x, metrics






def CABDCD(sc, A, b, M, N, l, mu, S, eps, max_iters=1000):
	count = 0

	A = A.cache()
	

	x = sc.parallelize(np.zeros(N), numSlices=A.getNumPartitions()).cache()
	alpha = np.zeros(M)

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


		idx = np.zeros(S*mu)
		for j in range(S):
			# Select indices
			coords = np.arange(M)
			np.random.shuffle(coords)
			idx[mu*j:mu*(j+1)] = coords[:mu]

		idx = idx.astype(int)


		YT = A.map(lambda col: col[np.array([np.random.RandomState(seed=count+i).permutation(M)[:mu] for i in range(S)]).flatten()]).cache()


		G = YT.map(lambda col: np.outer(col, col))
		Y_x = YT.zip(x).map(lambda t: t[0]*t[1])
		G, Y_x = G.zip(Y_x).reduce(lambda t0, t1: (t0[0]+t1[0], t0[1]+t1[1]))

		iteration_remote += time.time() - iteration_remote_start
		
		iteration_sequential_start = time.time()

		# aux = YT.zip(x).map(lambda t: np.concatenate((np.outer(t[0], t[0]), np.reshape(t[0]*t[1], (S*mu, 1))), axis=1)).sum()


		idx = np.array([np.random.RandomState(seed=count+i).permutation(M)[:mu] for i in range(S)]).flatten()

		# G = (1/l) * YT.map(lambda col: np.outer(col, col)).sum() + np.eye(S*mu)
		G = (1/l)*G + np.eye(S*mu)


		delta_a = np.zeros(S*mu)
		for j in range(S):
			Mj = G[mu*j:mu*(j+1), mu*j:mu*(j+1)]

			sum_x = np.zeros(mu)
			for t in range(j):
				sum_x += np.dot(G[mu*j:mu*(j+1), mu*t:mu*(t+1)], delta_a[mu*t:mu*(t+1)])
			
			rj = -Y_x[mu*j:mu*(j+1)] + (1/l)*sum_x + alpha[idx[mu*j:mu*(j+1)]]-b[idx[mu*j:mu*(j+1)]]
			delta_a[mu*j:mu*(j+1)] = -np.linalg.solve(Mj, rj)



		for i in range(S*mu):
			alpha[idx[i]] = alpha[idx[i]] + delta_a[i]


		iteration_sequential += time.time() - iteration_sequential_start

		iteration_remote_start = time.time()

		delta_a_bcast = sc.broadcast(delta_a)


		d_x = YT.map(lambda Y_row: np.dot(Y_row, delta_a_bcast.value))
		x = x.zip(d_x).map(lambda t: t[0]-(1/l)*t[1])

		# Break lineage
		if count % 50:
			x.checkpoint()
			x.count()

		iteration_remote += time.time() - iteration_remote_start

		iteration_total = time.time() - iteration_start

		# Residue

		if count % 5 == 0:
			residual_start = time.time()

			# A*x
			ATalpha = A.map(lambda col: np.dot(alpha, col))

			# AT*A*x
			AATalpha = A.zip(ATalpha).map(lambda t: np.dot(t[0], t[1])).sum()

			# r = AT*b - AT*A*x
			r = (1/l)*AATalpha+alpha-b

			norm_r = np.linalg.norm(r)
			if norm_r < eps:
				break

		time_sequential += iteration_sequential
		time_remote += iteration_remote
		time_execution += iteration_total
		count += S

	metrics['execution'] = time_execution
	metrics['sequential'] = time_sequential
	metrics['remote'] = time_remote
	metrics['iterations'] = count

	return alpha, x, metrics








def res_norm(A, b, y, l):
	return np.linalg.norm((1/l)*np.dot(A, np.dot(A.T, y))-b+y)




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


	A_p = RandomRDDs.uniformVectorRDD(sc, N, M, seed=1, numPartitions=numPartitions)
	b_p = RandomRDDs.uniformRDD(sc, M, seed=2, numPartitions=numPartitions)

	A = np.array(A_p.collect()).T
	b = np.array(b_p.collect())

	l = 0.5

	y_direct = direct(A, b, M, N, l)
	print("Direct")
	print(y_direct)
	print("Resnorm: ", res_norm(A, b, y_direct, l))
	print("\n")

	y_BDCD, x_BDCD, metrics_BDCD = BDCD(sc, A_p, b, M, N, l, mu, eps)
	print("BDCD")
	print(y_BDCD)
	print("Resnorm: ", res_norm(A, b, y_BDCD, l))
	print("Execution Time: ", metrics_BDCD['execution'])
	print("Sequential Time: ", metrics_BDCD['sequential'])
	print("Remote Time: ", metrics_BDCD['remote'])
	print("Iterations: ", metrics_BDCD['iterations'])
	print("\n")

	y_CABDCD, x_CABDCD, metrics_CABDCD = CABDCD(sc, A_p, b, M, N, l, mu, S, eps)
	print("CABDCD")
	print(y_CABDCD)
	print("Resnorm: ", res_norm(A, b, y_CABDCD, l))
	print("Execution Time: ", metrics_CABDCD['execution'])
	print("Sequential Time: ", metrics_CABDCD['sequential'])
	print("Remote Time: ", metrics_CABDCD['remote'])
	print("Iterations: ", metrics_CABDCD['iterations'])
	print("\n")