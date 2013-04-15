import matplotlib.pyplot as plt
import numpy

delta = 1e-10

sample_data = open("results_1D_parallel", "r").readlines()
my_data = open("results_1D_serial", "r").readlines()

sample = []
result = []


for line in sample_data:
  sline = str.split(line)
  aline=[]
  for item in sline:
    aline.append(float(item))
  sample.append(aline)

for line in my_data:
	sline = str.split(line)
	aline = []
	for item in sline:
		aline.append(float(item))
	result.append(aline)

sample_array = numpy.array(sample)
result_array = numpy.array(result)

diff = sample_array - result_array

count = 0

for i in range(diff.size):
	if diff[i] != 0:
		if (result_array[i] / diff[i]) < delta:
			count += 1

print "Suspicious no. of elements: %d" % count

plt.plot(sample_array, diff, 'o', markerfacecolor='r')

#f1 = plt.figure()
#plt.plot(sample_array[:,0], sample_array[:,1], 'o', markerfacecolor='r')
#f2 = plt.figure()
#plt.plot(result_array[:,0], result_array[:,1], 'o', markerfacecolor='b')
#f3 = plt.figure()
#plt.plot(x_diff, y_diff, 'o', markerfacecolor='g')

plt.show()

