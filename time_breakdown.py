import json
def get_metrics(log_file):
	#log_file = open("/Users/jimmy/Downloads/app-20180503124706-0004", "r")
	log_file = open(log_file, "r")
	lines = log_file.readlines()
	task_times = []
	des_time = 0
	com_time = 0
	for line in lines:
		json_object = json.loads(line)
		event = json_object["Event"]
		if event=="SparkListenerTaskEnd":
			launch_time = json_object["Task Info"]["Launch Time"]
			finish_time = json_object["Task Info"]["Finish Time"]
			task_times.append(finish_time-launch_time)
		if event=="SparkListenerApplicationStart":
			app_start_time = json_object["Timestamp"]
		if event=="SparkListenerApplicationEnd":
			app_end_time = json_object["Timestamp"]
		if event=="SparkListenerEnvironmentUpdate":
			node_num = int(json_object["Spark Properties"]["spark.executor.instances"])
			appName = json_object["Spark Properties"]["spark.app.name"]
		if event=="SparkListenerTaskEnd":
			des_time += int(json_object["Task Metrics"]["Executor Deserialize Time"])
			com_time += int(json_object["Task Metrics"]["Executor Run Time"])

	total_time = app_end_time - app_start_time
	comm_time = (total_time * node_num - sum(task_times))/node_num
	within_node_total = total_time - comm_time
	com_time_total = within_node_total * (com_time/(des_time+com_time))
	des_time_total = within_node_total * (des_time/(des_time+com_time))
	result = {"node number": node_num,
			"communication_time": comm_time,
			"com_time_total": com_time_total,
			"des_time_total": des_time_total,
			"appName": appName
			}

	return result



#print(task_times)
# data = json.load(open('/Users/jimmy/Downloads/app-20180503004928-0002.json'))
