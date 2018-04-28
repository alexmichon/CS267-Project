import matplotlib
import matplotlib.pyplot as plt
nodes_num = [1, 2, 4, 8, 16, 31]
execution_times = [43.12389039993286, 28.69337224960327, 23.147878885269165, 24.82949924468994, 30.237916946411133, 34.74588751792908]
plt.plot(nodes_num, execution_times)
execution_times = [130.08639001846313, 75.11723232269287, 39.97679686546326, 28.586194038391113, 21.461585998535156, 12.507391929626465]
plt.plot(nodes_num, execution_times)
plt.title("Number of nodes versus execution time")
ax = plt.axes()     
ax.legend(["BCD", "CABCD with s = 10"])
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Node number')
plt.ylabel('Execution time')
plt.ioff()
plt.savefig('cabcd_node_time_test.png')