from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs

import numpy as np
import argparse
import os
import requests
import sys
from datetime import datetime

from ridge import direct, res_norm, BCD, CABCD



def benchmark_mu(sc, A_p, b_p, M, N, l, eps):
	mus = np.linspace(1, N, num=min(N, 20), endpoint=True).astype(int)

	output_file = open("benchmark_mu.csv", "w")
	output_file.write("mu,execution,sequential,remote,iterations\n")

	for mu in mus:
		print("mu:", mu)

		x_BCD, metrics_BCD = BCD(sc, A_p, b_p, M, N, l, mu, eps)
		#print("BCD: ", x_BCD)
		#print("Resnorm: ", res_norm(A, b, x_BCD, l))
		print("Execution Time: ", metrics_BCD['execution'])
		print("Sequential Time: ", metrics_BCD['sequential'])
		print("Remote Time: ", metrics_BCD['remote'])
		print("Iterations: ", metrics_BCD['iterations'])

		output_file.write(','.join([
			str(mu), 
			str(metrics_BCD['execution']), 
			str(metrics_BCD['sequential']), 
			str(metrics_BCD['remote']),
			str(metrics_BCD['iterations'])
		]))
		output_file.write('\n')

	output_file.close()





def benchmark_S(sc, A_p, b_p, M, N, l, eps):
	Ss = [1,2,3,5,10,15,20,30]

	output_file = open("benchmark_S.csv", "w")
	output_file.write("S,mu,execution,sequential,remote,iterations\n")

	for S in Ss:
		print("S:", S)

		mus = np.linspace(1, N, num=min(N, 10), endpoint=True).astype(int)
		for mu in mus:
			print("mu:", mu)

			x_CABCD, metrics_CABCD = CABCD(sc, A_p, b_p, M, N, l, mu, S, eps)
			#print("CABCD: ", x_CABCD)
			#print("Resnorm: ", res_norm(A, b, x_CABCD, l))
			print("Execution Time: ", metrics_CABCD['execution'])
			print("Sequential Time: ", metrics_CABCD['sequential'])
			print("Remote Time: ", metrics_CABCD['remote'])
			print("Iterations: ", metrics_CABCD['iterations'])
			print("\n")

			output_file.write(','.join([
				str(S), 
				str(mu), 
				str(metrics_BCD['execution']), 
				str(metrics_BCD['sequential']), 
				str(metrics_BCD['remote']),
				str(metrics_BCD['iterations'])
			]))
			output_file.write('\n')

	output_file.close()



def communication_time(sc):
	appId = sc.applicationId
	url = sc.uiWebUrl + '/api/v1/applications/' + appId
	resp = requests.get(url=url + '/jobs')
	jobs = resp.json()

	comm_time = 0
	for j in jobs:
		t1 = datetime.strptime(j['submissionTime'], '%Y-%m-%dT%H:%M:%S.%f%Z')
		t2 = datetime.strptime(j['completionTime'], '%Y-%m-%dT%H:%M:%S.%f%Z')
		dt = t2-t1
		comm_time += dt.microseconds/1000

	return comm_time










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
	

	l = 0.5

	benchmark_mu(sc, A_p, b_p, M, N, l, eps)
	benchmark_S(sc, A_p, b_p, M, N, l, eps)