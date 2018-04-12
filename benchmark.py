from ridge import direct, res_norm, BCD, CABCD
from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
import numpy as np

import numpy as np
if __name__ == "__main__":
	print("test")
	
	M = 2000
	N = 100
	mu = 2
	l = 0.5
	S = 5
	eps = 0.05
	spark = SparkSession.builder \
		  .master('local') \
		  .appName("Linear Regression Model Benchmark") \
		  .config("spark.executor.memory", "16gb") \
		  .config("spark.executor.instances", "1") \
		  .getOrCreate()
	sc = spark.sparkContext
	sc.setLogLevel("ERROR")
	A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=1, seed=16)
	b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=1, seed=20)

	A = A_p.collect()
	b = b_p.collect()
	x_direct = direct(A, b, M, N, l)

	print("Direct: ", x_direct)
	print("Resnorm: ", res_norm(A, b, x_direct, l))
	print("\n")

	nodes_num = np.arange(31,1,-1)
	for node_num in nodes_num:
		print("node_num:", node_num)
		spark = SparkSession.builder \
		  .master('local') \
		  .appName("Linear Regression Model Benchmark") \
		  .config("spark.executor.memory", "1gb") \
		  .config("spark.executor.instances", str(node_num)) \
		  .getOrCreate()

		sc = spark.sparkContext

		sc.setLogLevel("ERROR")

		A_p = RandomRDDs.uniformVectorRDD(sc, M, N, numPartitions=int(node_num), seed=16)
		b_p = RandomRDDs.uniformRDD(sc, M, numPartitions=int(node_num), seed=20)

		A = A_p.collect()
		b = b_p.collect()

		x_BCD, metrics_BCD = BCD(A_p, b_p, M, N, l, mu, eps)
		#print("BCD: ", x_BCD)
		#print("Resnorm: ", res_norm(A, b, x_BCD, l))
		print("BCD Execution Time: ", metrics_BCD['execution'])
		#print("Residual Time: ", metrics_BCD['residual'])
		print("Iterations: ", metrics_BCD['iterations'])
		x_CABCD, metrics_CABCD = CABCD(A_p, b_p, M, N, l, mu, S, eps)
		#print("CABCD: ", x_CABCD)
		#print("Resnorm: ", res_norm(A, b, x_CABCD, l))
		print("CABCD: Execution Time: ", metrics_CABCD['execution'])
		#print("Residual Time: ", metrics_CABCD['residual'])
		print("Iterations: ", metrics_CABCD['iterations'])
		print("\n")

	