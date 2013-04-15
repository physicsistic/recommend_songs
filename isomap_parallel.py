# Computation modules
from math import *
import sys, time
import numpy as np
import numpy.linalg

# PyCuda Modules
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.autoinit

# This is the maximum possible distance that any two objects can have on a distance matrix
MAX = 99999999.

_kernel_source = """
#include <math.h>

#define GETIN(x, y)	(in[(y) * N + (x)])
#define SETOUT(x, y, v)	(out[(y) * N + (x)] = v)
#define GETDXY(x, y) (abs(score_current[x] - score_current[y]))
#define MAX 99999999.

__global__ void distance_kernel(float *distmat, float *indata, int M, int N) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x > N - 1 || y > N - 1)
		return;

	for (int i = 0; i < M; i++) {
		distmat[y * N + x] += pow((indata[y * M + i] - indata[x * M + i]), 2);
	}
	__syncthreads();

	distmat[y * N + x] = sqrt(distmat[y * N + x]);
}

__global__ void graph_kernel(float *dismat, int N, float epsilon) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x > N - 1 || y > N - 1)
		return;

	if (dismat[y * N + x] > epsilon)
		dismat[y * N + x] = MAX;	
}

__global__ void floyd_kernel(float *in, float *out, int k, int N) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x > N - 1 || y > N - 1)
		return;

	if (GETIN(x, y) > GETIN(x, k) + GETIN(k, y))
		SETOUT(x, y, GETIN(x, k) + GETIN(k, y));
}

__global__ void B_matrix_kernel(float *D_matrix, float *B_matrix, float *score_current, int N) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x > N - 1 || y > N - 1)
		return;

	B_matrix[y * N + x] = 0;
	__syncthreads();

	if (GETDXY(x, y) != 0)
		B_matrix[y * N + x] = (D_matrix[y * N + x] / GETDXY(x, y)) * (- 1.);

	__syncthreads();

	if (x != y)
		B_matrix[x * N + x] -= B_matrix[x * N + y];
	__syncthreads(); 
}

__global__ void mds_kernel(float *D_matrix, float *B_matrix, float *score_current, float *score_next, float *sigma, int N) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x > N - 1 || y > N - 1)
		return;

	score_next[x] = 0;
	__syncthreads();
	score_next[x] += B_matrix[x * N + y] * score_current[y];

	sigma[0] = 0;

	if (x < y)
		sigma[0] += abs(GETDXY(x, y) - D_matrix[y * N + x]) / N;

}

__global__ void mds2_kernel(float *graph, float *row_sum, float *score_current, float *score_next, float *sigma, float *delta, int N) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x > N - 1)
		return;

	row_sum[x] = 0;
	__syncthreads();

	for (int j = 0; j < N; j++) {
		float Bij = 0.;
		if (j != x) {
			Bij = graph[x * N + j] / abs(score_current[x] - score_current[j]);
			row_sum[x] += Bij;
			score_next[x] -= Bij * score_current[j]; 
		}
		
	}

	score_next[x] += row_sum[x] * score_current[x];
	score_next[x] = (score_next[x] / N);

	__syncthreads();

	sigma[x] = 0;
	__syncthreads();
	for (int j = 0; j < N; j++) {
		if (j != x)
			sigma[x] += abs(graph[x * N + j] - abs(score_next[x] - score_next[j]));
			delta[x] += graph[x * N + j];
	}
}
"""

def cuda_compile(source_string, function_name):
	source_module = nvcc.SourceModule(source_string)
	return source_module.get_function(function_name)

# Class for computing iso data
class isodata:

	# load input data
	def load_data(self, indata):
		print("Loading data...")
		self.indata = np.array(indata)
		self.N, self.M = np.shape(self.indata)

	# calculate distrance matrix on the gpu in parallel
	def get_distance_matrix_parallel(self):
		print("Calculating distance matrix in parallel...")
		current_time = time.clock()
		dismat_d = gpu.to_gpu(np.float32(np.zeros([self.N, self.N])))
		indata_d = gpu.to_gpu(np.float32(self.indata))
		distance_kernel = cuda_compile(_kernel_source, 'distance_kernel')
		distance_kernel(
			dismat_d,
			indata_d,
			np.int32(self.M),
			np.int32(self.N),
			block=(32, 32, 1), 
			grid=(int(self.N / 32 + 1), int(self.N / 32 + 1))
		)
		self.dismat = dismat_d.get()
		print "Time to calculate distance matrix in parallel: %6.2f s" % (time.clock() - current_time)

	# calculate distance matrix
	def get_distance_matrix(self):
		print("Calculating distance matrix...")
		current_time = time.clock()
		dismat = np.zeros((self.N, self.N), dtype=np.float32)
		for i in range(self.N):
			for j in range(i + 1, self.N):
				difference = self.indata[i] - self.indata[j]
				dismat[i][j] = sqrt(np.dot(difference, difference))
				dismat[j][i] = dismat[i][j]
			self.dismat = dismat
		print "Time to calculate distance matrix: %6.2f s" % (time.clock() - current_time)

	# calculate graph matrix on the gpu in parallel
	def get_graph_matrix_parallel(self):
		print("Constructing graph matrix in parallel...")
		current_time = time.clock()
		graph_d = gpu.to_gpu(np.float32(self.dismat))
		graph_kernel = cuda_compile(_kernel_source, "graph_kernel")
		graph_kernel(
			graph_d,
			np.int32(self.N),
			np.float32(self.e),
			block=(32, 32, 1), 
			grid=(int(self.N / 32 + 1), int(self.N / 32 + 1))
		)
		self.graph = graph_d.get()
		print "Time to get graph matrix in parallel: %6.2f s" % (time.clock() - current_time)

	# calculate graph matrix 
	def get_graph_matrix(self):
		print("Constructing graph matrix...")
		current_time = time.clock()
		graph = np.zeros((self.N, self.N), dtype=np.float32)
		for i in range(self.N):
			for j in range(self.N):
				graph[i][j] = MAX
		counter = np.zeros(self.N, dtype=np.int32)
		for i in range(self.N):
			for j in range(i, self.N):
				if (self.dismat[i][j] < self.e):
					graph[i][j] = self.dismat[i][j]
				graph[j][i] = graph[i][j]
		self.graph = graph
		print "Time to get graph matrix: %6.2f s" % (time.clock() - current_time)

	# calculate shortest path using floyd algorithm
	def apply_floyd_parallel(self):
		print("Applying Floyd's algorithm in parallel...")
		current_time = time.clock()
		input_d = gpu.to_gpu(np.float32(self.graph))
		output_d = gpu.to_gpu(np.float32(self.graph))
		floyd_kernel = cuda_compile(_kernel_source, 'floyd_kernel')
		for k in range(self.N):
			floyd_kernel(
				input_d, 
				output_d, 
				np.int32(k),
				np.int32(self.N),
				block=(32, 32, 1), 
				grid=(int(self.N / 32 + 1), int(self.N / 32 + 1))
			)
			input_d = output_d
			k += 1
		self.graph = output_d.get()
		print "Time to apply the Floyd's algorithm in parallel: %6.2f s" %(time.clock() - current_time)
		print self.graph

	# Apply MDS via SMACOF
	def apply_mds_parallel2(self):
		print("Applying parallel MDS via SMACOF...")
		current_time = time.clock()
		graph_d = gpu.to_gpu(np.float32(self.graph))
		row_sum_d = gpu.to_gpu(np.float32(np.zeros(self.N)))
		score_current_d = gpu.to_gpu(np.float32(np.random.uniform(0, 10, size=self.N)))
		score_next_d = gpu.to_gpu(np.float32(np.zeros(self.N)))
		sigma_d = gpu.to_gpu(np.float32(np.zeros(self.N)))
		delta_d = gpu.to_gpu(np.float32(np.zeros(self.N)))
		mds2_kernel = cuda_compile(_kernel_source, 'mds2_kernel')
		stress  = 1
		while (stress > 0.001):
			mds2_kernel(
				graph_d,
				row_sum_d,
				score_current_d,
				score_next_d,
				sigma_d,
				delta_d,
				np.int32(self.N),
				block=(1024, 1, 1), 
				grid=(int(self.N / 1024 + 1), int(1))
			)
			score_current_d = score_next_d
			score_next_d = gpu.to_gpu(np.float32(np.zeros(self.N)))
			stress = gpu.sum(sigma_d).get() / gpu.sum(delta_d).get()
		self.outdata = score_current_d.get()
		print "Time to apply parallel MDS: %6.2f s" % (time.clock() - current_time)
		
	def apply_mds_parallel(self):
		print("Applying parallel MDS via SMACOF...")
		current_time = time.clock()
		D_matrix_d = gpu.to_gpu(np.float32(self.graph))
		B_matrix_d = gpu.to_gpu(np.float32(np.zeros([self.N, self.N])))
		score_current_d = gpu.to_gpu(np.float32(np.random.uniform(0, 10, size=self.N)))
		score_next_d = gpu.to_gpu(np.float32(np.zeros(self.N)))
		sigma_d = gpu.to_gpu(np.float32(np.array(1.0)))
		B_matrix_kernel = cuda_compile(_kernel_source, 'B_matrix_kernel')
		mds_kernel = cuda_compile(_kernel_source, 'mds_kernel')
		k = 0
		print sigma_d.get()
		for k in range(2):
			B_matrix_kernel(
				D_matrix_d,
				B_matrix_d,
				score_current_d,
				np.int32(self.N),
				block=(16, 16, 1), 
				grid=(int(self.N / 16 + 1), int(self.N / 16 + 1))
			)
			print B_matrix_d.get()
			mds_kernel(
				D_matrix_d,
				B_matrix_d,
				score_current_d,
				score_next_d,
				sigma_d,
				np.int32(self.N),
				block=(16, 16, 1), 
				grid=(int(self.N / 16 + 1), int(self.N / 16 + 1))
			)
			score_current_d = score_next_d
			print score_current_d.get()
			print sigma_d.get()
			k += 1

		self.outdata = score_next_d.get()
		print "Time to apply parallel MDS: %6.2f s" % (time.clock() - current_time)

	# get lower dimension matrix using MDS
	def apply_mds(self):
		print("Applying MDS...")
		current_time = time.clock()
		A = np.zeros((self.N, self.N), dtype=np.float32)
		for i in range(self.N):
			for j in range(self.N):
				A[i][j] = -self.graph[i][j] * self.graph[i][j] / 2

		a = np.zeros(self.N, dtype=np.float32)
		for i in range(self.N):
			a[i] = 0.0
			for j in range(self.N):
				a[i] = a[i] + A[i][j] / self.N

		b = 0.0
		for i in range(self.N):
			for j in range(self.N):
				b = b + A[i][j] / (self.N * self.N)

		B = np.zeros((self.N, self.N), dtype=np.float32)
		for i in range(self.N):
			for j in range(self.N):
				B[i][j] = A[i][j] - a[i] - a[j] + b
		eigen_time_start = time.clock()
		eigenvals, eigenvecs = numpy.linalg.eig(B)
		eigenvecs = eigenvecs.real
		eigenvals = eigenvals.real
		eigen_time_total = time.clock() - eigen_time_start
		self.outdata = np.zeros((self.N, self.O), dtype=np.float32)
		for i in range(self.N):
			for j in range(self.O):
				self.outdata[i][j] = eigenvecs[i][j] * sqrt(eigenvals[j])

		print "Time to apply MDS: %6.2f s" % (time.clock() - current_time)
		print "Time to compute eigensystem: %6.2f s" % eigen_time_total
	# execute isomap of input data
	def apply_isomap(self, e=0.5, O=1):
		self.e = e
		self.O = O
		self.get_distance_matrix()
		self.get_graph_matrix()
		self.apply_floyd()
		self.apply_mds()

	# execute isomap of input data in parallel
	def apply_isomap_parallel(self, e=0.5, O=1):
		self.e = e
		self.O = O
		self.get_distance_matrix_parallel()
		self.get_graph_matrix_parallel()
		self.apply_floyd_parallel()
		self.apply_mds_parallel2()






