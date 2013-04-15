from isomap_parallel import *
import numpy
import time

data = open("song1000", "r").readlines()

matrix = []

print("Importing data from file...")
current_time = time.clock()
for line in data:
	sline = line.split('\t')
	aline = []
	for item in line:
		aline.append(item)
	matrix.append(sline)
print "Time to import files: %6.2f s" % (time.clock() - current_time)


M = numpy.float32(numpy.array(matrix)[:,4:8])
print M.shape
A = isodata()
A.load_data(M)
A.apply_isomap(e=200)

print("Writing data to file...")
current_time = time.clock()
results = open("song_reduced_1000", "w")
for line in A.outdata:
	for item in line:
		results.write(" %f" % item)
	results.write("\n")
results.close()
print "Time to write files: %6.2f s" % (time.clock() - current_time)
