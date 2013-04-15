from isomap_parallel import *
import numpy
import time

data = open("out10000.dat", "r").readlines()

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

M = numpy.array(matrix)[:,4:7]
print M.shape
A = isodata()
A.load_data(M)
A.apply_isomap_parallel(e=0.5)

print("Writing data to file...")
current_time = time.clock()
results = open("song_reduced_100000_no_tempo", "w")
for item in A.outdata:
	results.write(" %f" % item)
	results.write("\n")
results.close()
print "Time to write files: %6.2f s" % (time.clock() - current_time)
