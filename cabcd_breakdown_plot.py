from time_breakdown import get_metrics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
log_files = ["app-20180505161246-0000", "app-20180505161412-0001", "app-20180505161509-0002", "app-20180505161545-0003", "app-20180505161619-0004"]
path = "/global/cscratch1/sd/wjm9696/spark/event_logs/"
comm_times = []
comp_times = []
for log_file in log_files:
	metrics = get_metrics(path + log_file)
	comm_time = metrics["communication_time"] + metrics["des_time_total"]
	comp_time = metrics["com_time_total"]
	app_name = metrics["appName"]
	comm_times.append(comm_time)
	comp_times.append(comp_time)
	
	print("appName:", app_name)
	print("compute_time", comp_time)
	print("comm_time", comm_time)
	print("total_time", comp_time+comm_time)
	print("communication_percentage", comm_time/(comp_time+comm_time))
N = len(log_files)
ind = np.arange(N)
width = 0.35
plt.switch_backend('agg')
matplotlib.use('TkAgg')
comm_times = np.array(comm_times)/1000
comp_times = np.array(comp_times)/1000
p1 = plt.bar(ind, comm_times, width)
p2 = plt.bar(ind, comp_times, width,
             bottom=comm_times)
plt.legend((p1[0], p2[0]), ('Communication time', 'Computation time'))
plt.xlabel('Unrolling steps')
plt.ylabel('Running time breakdown (seconds)')
plt.title('Running time breakdown of different s for CABCD')
plt.xticks(ind, ('s=2', 's=4', 's=8', 's=16', 's=32'))
plt.savefig("cabcd_breakdown.png")
