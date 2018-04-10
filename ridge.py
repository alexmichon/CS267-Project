from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
import numpy as np
import time



def direct(A, b, M, N, l):
	# (A T * A + l*I)^-1
	ATA = np.linalg.inv(np.dot(A.T, A) + l*np.eye(N))
	# A * b
	ATb = np.dot(A.T, b)
	# x = (A T * A)^-1 * A*b
	return np.dot(ATA, ATb)


def BCD(A, b, M, N, l, mu, eps, max_iters=1000):
	count = 0

	A.cache()
	b.cache()

	x = np.zeros(N)
	z = b.map(lambda e: -e)

	metrics = {
		'iterations': 0,
		'execution': 0,
		'residual': 0
	}

	execution_start = time.time()

	while count < max_iters:

		# Select indices
		idx = np.arange(N)
		np.random.shuffle(idx)
		idx = idx[:mu]

		# Sub matrix of A
		Ah = A.map(lambda row: row[idx])

		# Ah T * Ah
		dotAh = Ah.map(lambda row: np.outer(row, row))
		rh = Ah.zip(z).map(lambda t: t[0]*t[1])

		aux = dotAh.zip(rh).map(lambda t: np.concatenate((t[0],np.reshape(t[1], (mu.size, 1))), axis=1)).sum()

		dotAh = l*np.eye(mu) + aux[:, 0:mu]
		rh = -aux[:,-1]


		# (Ah T * Ah)^-1
		invDotAh = np.linalg.inv(dotAh)
		for i in range(mu):
			rh[i] = rh[i] - l*x[idx[i]]

		dxh = np.dot(invDotAh,rh)

		# x = x + xhat
		for i in range(mu):
			x[idx[i]] += dxh[i]

		dz = Ah.map(lambda row: np.dot(row, dxh))
		z = z.zip(dz).map(lambda t: t[0] + t[1])

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
			metrics['residual'] += time.time() - residual_start
			if norm_r < eps:
				break

		count += 1

	metrics['execution'] += time.time() - execution_start
	metrics['iterations'] = count

	return x, metrics



def CABCD(A, b, M, N, l, mu, S, eps, max_iters=1000):
	converged = False
	count = 0

	A.cache()
	b.cache()

	x = np.zeros(N)
	alpha = b.map(lambda e: -e)

	metrics = {
		'iterations': 0,
		'execution': 0,
		'residual': 0
	}

	execution_start = time.time()

	while not converged and count < max_iters:

		idx = np.zeros(S*mu)
		for j in range(S):
			# Select indices
			coords = np.arange(N)
			np.random.shuffle(coords)
			idx[mu*j:mu*(j+1)] = coords[:mu]

		idx = idx.astype(int)


		Y = A.map(lambda row: row[idx])

		aux = Y.zip(alpha).map(lambda t: np.concatenate((np.outer(t[0], t[0]), np.reshape(t[0]*t[1], (S*mu, 1))), axis=1)).sum()

		G = aux[:, :S*mu] + l*np.eye(S*mu)
		Y_alpha = -aux[:, -1]


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


		d_alpha = Y.map(lambda Y_row: np.dot(Y_row, delta_x))
		alpha = alpha.zip(Y_alpha).map(lambda t: t[0]+t[1])

		# Residue

		if count % 5 == 0:

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

			metrics['residual'] += time.time() - residual_start

			if norm_r < eps:
				break

		count += 1

	metrics['execution'] += time.time() - execution_start
	metrics['iterations'] = count

	return x, metrics






def res_norm(A, b, x, l):
	return np.linalg.norm(np.dot(A.T, np.dot(A, x)-b)+l*x)




if __name__ == "__main__":

	spark = SparkSession.builder \
	  .master('local') \
	  .appName("Linear Regression Model") \
	  .config("spark.executor.memory", "1gb") \
	  .getOrCreate()

	sc = spark.sparkContext

	M = 10
	N = 4


	A_p = RandomRDDs.uniformVectorRDD(sc, M, N)
	b_p = RandomRDDs.uniformRDD(sc, M)

	A = A_p.collect()
	b = b_p.collect()


	mu = 1
	l = 0.5

	S = 5

	eps = 0.1

	# x_direct = direct(A, b, M, N, l)
	# print("Direct")
	# print(x_direct)
	# print("Resnorm: ", res_norm(A, b, x_direct, l))
	# print("\n")

	# x_BCD, metrics_BCD = BCD(A_p, b_p, M, N, l, mu, eps)
	# print("BCD")
	# print(x_BCD)
	# print("Resnorm: ", res_norm(A, b, x_BCD, l))
	# print("Execution Time: ", metrics_BCD['execution'])
	# print("Residual Time: ", metrics_BCD['residual'])
	# print("Iterations: ", metrics_BCD['iterations'])
	# print("\n")

	# x_CABCD, metrics_CABCD = CABCD(A_p, b_p, M, N, l, mu, S, eps)
	# print("CABCD")
	# print(x_CABCD)
	# print("Resnorm: ", res_norm(A, b, x_CABCD, l))
	# print("Execution Time: ", metrics_CABCD['execution'])
	# print("Residual Time: ", metrics_CABCD['residual'])
	# print("Iterations: ", metrics_CABCD['iterations'])
	# print("\n")



