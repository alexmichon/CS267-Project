from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
import numpy as np
import time





def direct(A, b, M, N, l):
	Z = np.linalg.inv((1/l)*np.dot(A, A.T) + np.eye(M))
	return np.dot(Z, b)


def BDCD(sc, A, b, M, N, l, mu, eps, max_iters=50):
	count = 0

	A.cache()

	x = sc.parallelize(np.zeros(N))
	alpha = np.zeros(M)

	metrics = {
		'iterations': 0,
		'execution': 0,
		'residual': 0
	}

	execution_start = time.time()

	while count < max_iters:

		# Select indices
		idx = np.arange(M)
		np.random.shuffle(idx)
		idx = idx[:mu]

		# Sub matrix of A
		Ah = A.map(lambda col: col[idx])

		# Ah T * Ah
		dotAh = Ah.map(lambda col: np.outer(col, col))
		rh = Ah.zip(x).map(lambda t: t[0]*t[1])

		aux = dotAh.zip(rh).map(lambda t: np.concatenate((t[0],np.reshape(t[1], (mu, 1))), axis=1)).sum()

		dotAh = np.eye(mu) + (1/l)*aux[:, :mu]
		rh = -aux[:,-1] + alpha[idx] - b[idx]


		# (Ah T * Ah)^-1
		invDotAh = np.linalg.inv(dotAh)
		# for i in range(mu):
		# 	rh[i] = rh[i] - l*x[idx[i]]

		dalphah = -np.dot(invDotAh,rh)

		# x = x + xhat
		# for i in range(mu):
		# 	alpha[idx[i]] += dalphah[i]
		alpha[idx] += dalphah

		dx = Ah.map(lambda col: np.dot(col, dalphah))
		x = x.zip(dx).map(lambda t: t[0] - (1/l)*t[1])

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
			metrics['residual'] += time.time() - residual_start
			if norm_r < eps:
				break

		count += 1

	metrics['execution'] += time.time() - execution_start
	metrics['iterations'] = count

	return alpha, x, metrics






def CABDCD(sc, A, b, M, N, l, mu, S, eps, max_iters=1000):
	count = 0

	A.cache()
	

	x = sc.parallelize(np.zeros(N))
	alpha = np.zeros(M)

	metrics = {
		'iterations': 0,
		'execution': 0,
		'residual': 0
	}

	execution_start = time.time()

	while count < max_iters:

		idx = np.zeros(S*mu)
		for j in range(S):
			# Select indices
			coords = np.arange(M)
			np.random.shuffle(coords)
			idx[mu*j:mu*(j+1)] = coords[:mu]

		idx = idx.astype(int)


		YT = A.map(lambda col: col[idx])

		aux = YT.zip(x).map(lambda t: np.concatenate((np.outer(t[0], t[0]), np.reshape(t[0]*t[1], (S*mu, 1))), axis=1)).sum()

		# G = (1/l) * YT.map(lambda col: np.outer(col, col)).sum() + np.eye(S*mu)
		G = (1/l)*aux[:, :S*mu] + np.eye(S*mu)
		Y_x = aux[:, -1]


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


		d_x = YT.map(lambda Y_row: np.dot(Y_row, delta_a))
		x = x.zip(d_x).map(lambda t: t[0]-(1/l)*t[1])

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
			metrics['residual'] += time.time() - residual_start
			if norm_r < eps:
				break

		count += 1

	metrics['execution'] += time.time() - execution_start
	metrics['iterations'] = count

	return alpha, x, metrics








def res_norm(A, b, y, l):
	return np.linalg.norm((1/l)*np.dot(A, np.dot(A.T, y))-b+y)




if __name__ == "__main__":

	spark = SparkSession.builder \
	  .master('local') \
	  .appName("Linear Regression Model") \
	  .config("spark.executor.memory", "1gb") \
	  .getOrCreate()

	sc = spark.sparkContext
	sc.setLogLevel("ERROR")

	M = 5
	N = 10


	A_p = RandomRDDs.uniformVectorRDD(sc, N, M, seed=1)
	b_p = RandomRDDs.uniformRDD(sc, M, seed=2)

	A = np.array(A_p.collect()).T
	b = np.array(b_p.collect())


	mu = 2
	l = 0.5

	S = 5

	eps = 0.1

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
	print("Residual Time: ", metrics_BDCD['residual'])
	print("Iterations: ", metrics_BDCD['iterations'])
	print("\n")

	y_CABDCD, x_CABDCD, metrics_CABDCD = CABDCD(sc, A_p, b, M, N, l, mu, S, eps)
	print("CABDCD")
	print(y_CABDCD)
	print("Resnorm: ", res_norm(A, b, y_CABDCD, l))
	print("Execution Time: ", metrics_CABDCD['execution'])
	print("Residual Time: ", metrics_CABDCD['residual'])
	print("Iterations: ", metrics_CABDCD['iterations'])
	print("\n")