from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
import argparse
import numpy as np
import os
import sys
import time



def direct(A, b, M, N, l):
	# (A T * A + l*I)^-1
	ATA = np.linalg.inv(np.dot(np.transpose(A), A) + l*np.eye(N))
	# A * b
	ATb = np.dot(np.transpose(A), b)
	# x = (A T * A)^-1 * A*b
	return np.dot(ATA, ATb)


def BCD(sc, A, b, M, N, l, mu, eps, max_iters=500):
	count = 0

	A = A.cache()
	b = b.cache()

	x = np.zeros(N)
	z = b.map(lambda e: -e).cache()

	metrics = {}
	metrics['residual'] = []
	time_execution = 0
	time_sequential = 0
	time_remote = 0

	AA = A.map(lambda row: (0, N, mu, row)).cache()

	while True:

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
		rRh = Ah.zip(z).map(lambda t: t[0]*t[1])

		# aux = dotAh.zip(rh).map(lambda t: np.concatenate((t[0],np.reshape(t[1], (mu, 1))), axis=1)).sum()
		dotAh, rh = rDotAh.zip(rRh).reduce(lambda t0, t1: (t0[0]+t1[0], t0[1]+t1[1]))

		iteration_remote += time.time() - iteration_remote_start
		
		iteration_sequential_start = time.time()

		# Select indices
		idx = np.random.RandomState(seed=count).permutation(N)[:mu]

		dotAh += l*np.eye(mu)
		rh = -rh


		# (Ah T * Ah)^-1
		invDotAh = np.linalg.inv(dotAh)
		for i in range(mu):
			rh[i] = rh[i] - l*x[idx[i]]

		dxh = np.dot(invDotAh,rh)

		# x = x + xhat
		for i in range(mu):
			x[idx[i]] += dxh[i]

		iteration_sequential += time.time() - iteration_sequential_start

		iteration_remote_start = time.time()

		dxh_bcast = sc.broadcast(dxh)

		dz = Ah.map(lambda row: np.dot(row, dxh_bcast.value))
		z = z.zip(dz).map(lambda t: t[0] + t[1]).cache()

		# Break lineage
		if count % 50:
			z.checkpoint()
			z.count()

		iteration_remote += time.time() - iteration_remote_start

		iteration_total = time.time() - iteration_start

		# Residue

		if count % 5 == 0:
			residual_start = time.time()

			# A*x
			Ax = A.map(lambda row: np.dot(x, row))

			# AT*A*x
			ATAx = A.zip(Ax).map(lambda t: np.dot(t[0], t[1])).sum()

			# AT*b
			ATb = A.zip(b).map(lambda t: np.dot(t[0], t[1])).sum()

			# r = AT*b - AT*A*x
			r = ATb-ATAx-l*x

			norm_r = np.linalg.norm(r)
			metrics['residual'].append(norm_r)
			if norm_r < eps:
				break	


		time_sequential += iteration_sequential
		time_remote += iteration_remote
		time_execution += iteration_total
		count += 1

		# print("Iteration ", count, " took ", iteration_total, "s (sequential: ", iteration_sequential, "s remote: ", iteration_remote, "s)")

	metrics['execution'] = time_execution
	metrics['sequential'] = time_sequential
	metrics['remote'] = time_remote
	metrics['iterations'] = count

	return x, metrics



def CABCD(sc, A, b, M, N, l, mu, S, eps, max_iters=500):
	count = 0

	A = A.cache()
	b = b.cache()

	x = np.zeros(N)
	alpha = b.map(lambda e: -e).cache()

	metrics = {}
	metrics['residual'] = []

	time_execution = 0
	time_sequential = 0
	time_remote = 0

	AA = A.map(lambda row: (0, N, mu, S, row)).cache()

	while True:

		iteration_total = 0
		iteration_remote = 0
		iteration_sequential = 0

		iteration_start = time.time()


		iteration_remote_start = time.time()


		# Y = A.map(lambda row: row[idx])
		Y = AA.map(lambda e: e[4][np.array([np.random.RandomState(seed=count+i).permutation(e[1])[:e[2]] for i in range(e[3])]).flatten()]).cache()
		# AA = AA.map(lambda e: (e[0]+e[3], e[1], e[2], e[3], e[4])).cache()

		# aux = Y.zip(alpha).map(lambda t: np.concatenate((np.outer(t[0], t[0]), np.reshape(t[0]*t[1], (S*mu, 1))), axis=1)).sum()
		G = Y.map(lambda row: np.outer(row, row))
		Y_alpha = Y.zip(alpha).map(lambda t: t[0]*t[1])
		G, Y_alpha = G.zip(Y_alpha).reduce(lambda t0, t1: (t0[0]+t1[0], t0[1]+t1[1]))

		iteration_remote += time.time() - iteration_remote_start
		
		iteration_sequential_start = time.time()

		idx = np.array([np.random.RandomState(seed=count+i).permutation(N)[:mu] for i in range(S)]).flatten()

		G += l*np.eye(S*mu)
		Y_alpha = -Y_alpha


		delta_x = np.zeros(S*mu)
		for j in range(S):
			Mj = G[mu*j:mu*(j+1), mu*j:mu*(j+1)]

			sum_x = np.zeros(mu)
			for t in range(j):
				sum_x -= np.dot(G[mu*j:mu*(j+1), mu*t:mu*(t+1)], delta_x[mu*t:mu*(t+1)])
			
			rj = -l*x[idx[mu*j:mu*(j+1)]] + sum_x + Y_alpha[mu*j:mu*(j+1)]
			delta_x[mu*j:mu*(j+1)] = np.linalg.solve(Mj, rj)



		for i in range(S*mu):
			x[idx[i]] = x[idx[i]] + delta_x[i]

		iteration_sequential += time.time() - iteration_sequential_start

		iteration_remote_start = time.time()

		delta_x_bcast = sc.broadcast(delta_x)


		d_alpha = Y.map(lambda Y_row: np.dot(Y_row, delta_x_bcast.value))
		alpha = alpha.zip(d_alpha).map(lambda t: t[0]+t[1]).cache()

		# Break lineage
		if count % 50:
			alpha.checkpoint()
			alpha.count()

		iteration_remote += time.time() - iteration_remote_start

		iteration_total = time.time() - iteration_start

		# Residue

		if count/S % 5 == 0:

			residual_start = time.time()

			# A*x
			Ax = A.map(lambda row: np.dot(x, row))

			# AT*A*x
			ATAx = A.zip(Ax).map(lambda t: t[0] * t[1]).sum()

			# AT*b
			ATb = A.zip(b).map(lambda t: t[0]*t[1]).sum()

			# r = AT*b - AT*A*x
			r = ATb-ATAx-l*x

			norm_r = np.linalg.norm(r)

			metrics['residual'].append(norm_r)
			if norm_r < eps:
				break

		time_sequential += iteration_sequential
		time_remote += iteration_remote
		time_execution += iteration_total
		count += S

		# print("Iteration ", count, " took ", iteration_total, "s (sequential: ", iteration_sequential, "s remote: ", iteration_remote, "s)")

	metrics['execution'] = time_execution
	metrics['sequential'] = time_sequential
	metrics['remote'] = time_remote
	metrics['iterations'] = count

	return x, metrics






def res_norm(A, b, x, l):
	return np.linalg.norm(np.dot(np.transpose(A), np.dot(A, x)-b)+l*x)




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
	sc.setLogLevel("ERROR")


	A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=numPartitions, seed=1)
	b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=numPartitions, seed=2)

	A = np.array(A_p.collect())
	b = np.array(b_p.collect())

	l = 0.5

	x_direct = direct(A, b, M, N, l)
	print("Direct")
	print(x_direct)
	print("Resnorm: ", res_norm(A, b, x_direct, l))
	print("\n")

	x_BCD, metrics_BCD = BCD(sc, A_p, b_p, M, N, l, mu, eps)
	print("BCD")
	print(x_BCD)
	print("Resnorm: ", res_norm(A, b, x_BCD, l))
	print("Execution Time: ", metrics_BCD['execution'])
	print("Sequential Time: ", metrics_BCD['sequential'])
	print("Remote Time: ", metrics_BCD['remote'])
	print("Iterations: ", metrics_BCD['iterations'])
	print("\n")

	x_CABCD, metrics_CABCD = CABCD(sc, A_p, b_p, M, N, l, mu, S, eps)
	print("CABCD")
	print(x_CABCD)
	print("Resnorm: ", res_norm(A, b, x_CABCD, l))
	print("Execution Time: ", metrics_CABCD['execution'])
	print("Sequential Time: ", metrics_CABCD['sequential'])
	print("Remote Time: ", metrics_CABCD['remote'])
	print("Iterations: ", metrics_CABCD['iterations'])
	print("\n")



