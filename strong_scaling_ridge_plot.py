import matplotlib
import matplotlib.pyplot as plt
nodes_num = [1, 2, 4]
execution_times = [10, 10, 10]
ax = plt.axes()   
plt.plot(nodes_num, execution_times)
plt.title("number of nodes versus execution time")     
ax.legend(["BCD", "CABCD with s = 10"])
plt.yscale('log')
plt.xscale('log')
plt.xlabel('node number')
plt.ylabel('execution time')
plt.ioff()
plt.savefig('cabcd_node_time_test.png')