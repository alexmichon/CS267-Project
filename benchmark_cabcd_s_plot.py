import matplotlib
import matplotlib.pyplot as plt
import json
datafilename = "BCD_s_residual_random_data.txt"
dataFile = open(datafilename)
data = json.load(dataFile)
title = data["title"]
xlabel = data['xlabel']
ylabel = data['ylabel']
xss = data['xs']
residualss = data['ys']
filename = data['filename']
legends = data['legends']
for xs,residuals in zip(xss,residualss):
	plt.plot(xs, residuals)
plt.title(title)  
ax = plt.axes()      
ax.legend(legends)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ioff()
plt.savefig(filename)
dataFile.close()