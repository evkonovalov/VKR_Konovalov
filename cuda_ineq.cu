#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <chrono>

#define BLOCKSIZE 128
#define GRIDSIZE 128
#define N 128
#define VECTORSIZE 2
#define FUNCTIONQUANTITY 2
#define ENDCRITERIA 0.01

typedef double(*func)(double*);

typedef struct {
	double mins[VECTORSIZE];
	double maxs[VECTORSIZE];
} Box;

double diam(Box &box) {
	double diam = 0;
	for (int i = 0; i < VECTORSIZE; i++)
		diam = fmax(diam, abs(box.maxs[i] - box.mins[i]));
	return diam;
}

std::pair<Box, Box> split(Box &box) {
	std::pair<Box, Box> r;
	int maxIndex = 0;
	double max = abs(box.maxs[0] - box.mins[0]);
	for (int i = 0; i < VECTORSIZE; i++) {
		double cur = abs(box.maxs[i] - box.mins[i]);
		if (cur > max){
			maxIndex = i;
			max = cur;
		}
		r.first.mins[i] = box.mins[i];
		r.first.maxs[i] = box.maxs[i];
		r.second.mins[i] = box.mins[i];
		r.second.maxs[i] = box.maxs[i];
	}
	double border = box.mins[maxIndex] + abs(box.maxs[maxIndex] - box.mins[maxIndex]) / 2;
	r.first.maxs[maxIndex] = border;
	r.second.mins[maxIndex] = border;
	return r;
}
//Function description
__device__ double f1(double* vector) {
	return (vector[0] - 15)*(vector[0] - 15) + (vector[1] - 15)*(vector[1] - 15) - 100;
}

__device__ double f2(double* vector) {
	return sinf(vector[0]) + cosf(vector[1]);
}

__device__ double f3(double* vector) {
	return (vector[0] - 5)*(vector[0] - 5) + (vector[2] - 5)*(vector[2] - 5) - 100;
}

__device__ double f4(double* vector) {
	return vector[0] + vector[1] + vector[2] - 100;
}

template <func... Functions>
__global__ void boostKernel(double* boxes, int len, double* retmax, double* retmin)
{
	//For every block we create two shared arrays that contain values of every function on grid. 
	//In the beginning they duplicate each other, but smax aggregates with max function and smin with min function.
	__shared__ double smax[FUNCTIONQUANTITY*BLOCKSIZE];
	__shared__ double smin[FUNCTIONQUANTITY*BLOCKSIZE];

	constexpr func table[] = { Functions... };
	double split_len = pow(double(N),VECTORSIZE) / BLOCKSIZE + 1;
	double split_min[FUNCTIONQUANTITY];
	double split_max[FUNCTIONQUANTITY];
	int box_num = blockIdx.x < len ? blockIdx.x : len - 1;
	Box p;
	for (int i = 0; i < VECTORSIZE; i++) {
		p.mins[i] = boxes[box_num * 2 * VECTORSIZE + i * 2];
		p.maxs[i] = boxes[box_num * 2 * VECTORSIZE + i * 2 + 1];
	}
	for (int k = 0; k < split_len; k++){
		double vec[VECTORSIZE];
		int cell_num = split_len*threadIdx.x + k;
		for (int i = 0; i < VECTORSIZE; i++) {
			vec[i] = p.mins[i] + (fabs(p.maxs[i] - p.mins[i]) / N) * (cell_num % N);
			cell_num /= N;
		}
		for (int i = 0; i < FUNCTIONQUANTITY; i++) {
			func fun = table[i];
			double res = fun(vec);
			if (k == 0 || split_max[i] < res)
				split_max[i] = res;
			if (k == 0 || split_min[i] > res)
				split_min[i] = res;
		}
	}
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		smax[i*BLOCKSIZE + threadIdx.x] = split_max[i];
		smin[i*BLOCKSIZE + threadIdx.x] = split_min[i];
	}
	__syncthreads();
	int s = blockDim.x >> 1;
	while (s != 0) {
		if (threadIdx.x < s) {
			int su = threadIdx.x + s;
			for (int i = 0; i < FUNCTIONQUANTITY; i++) {
				smax[i*blockDim.x + threadIdx.x] = fmax(smax[i*blockDim.x + threadIdx.x], smax[i*blockDim.x + su]);
				smin[i*blockDim.x + threadIdx.x] = fmin(smin[i*blockDim.x + threadIdx.x], smin[i*blockDim.x + su]);
			}
		}
		__syncthreads();
		s >>= 1;
	}
	__syncthreads();
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		retmax[blockIdx.x*FUNCTIONQUANTITY + i] = smax[i*blockDim.x];
		retmin[blockIdx.x*FUNCTIONQUANTITY + i] = smin[i*blockDim.x];
	}
	__syncthreads();
}

template <func... Functions>
__global__ void addKernel(Box box, double* maxout, double* minout, double* retmax, double* retmin)
{
	//For every block we create two shared arrays that contain values of every function on grid. 
	//In the beginning they duplicate each other, but smax aggregates with max function and smin with min function.
	__shared__ double smax[FUNCTIONQUANTITY*BLOCKSIZE];
	__shared__ double smin[FUNCTIONQUANTITY*BLOCKSIZE];

	int dim = pow(double(BLOCKSIZE * GRIDSIZE), 1 / double(VECTORSIZE));
	constexpr func table[] = { Functions... };
	double vec[VECTORSIZE];
	int threadNum = (threadIdx.x + blockIdx.x * blockDim.x);
	for (int i = 0; i < VECTORSIZE; i++) {
		vec[i] = box.mins[i] + (abs(box.maxs[i] - box.mins[i]) / dim) * (threadNum % dim);
		threadNum /= dim;
	}
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		func fun = table[i];
		smax[i*blockDim.x + threadIdx.x] = fun(vec);
		smin[i*blockDim.x + threadIdx.x] = fun(vec);
	}
	__syncthreads();
	int s = blockDim.x >> 1;
	while (s != 0) {
		if (threadIdx.x < s) {
			int su = threadIdx.x + s;
			for (int i = 0; i < FUNCTIONQUANTITY; i++) {
				smax[i*blockDim.x + threadIdx.x] = fmax(smax[i*blockDim.x + threadIdx.x], smax[i*blockDim.x + su]);
				smin[i*blockDim.x + threadIdx.x] = fmin(smin[i*blockDim.x + threadIdx.x], smin[i*blockDim.x + su]);
			}
		}
		__syncthreads();
		s >>= 1;
	}
	__syncthreads();
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		maxout[i*gridDim.x + blockIdx.x] = smax[i*blockDim.x];
		minout[i*gridDim.x + blockIdx.x] = smin[i*blockDim.x];
	}
	__syncthreads();
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		if (threadIdx.x < gridDim.x) {
			smax[i*blockDim.x + threadIdx.x] = maxout[i*gridDim.x + threadIdx.x];
			smin[i*blockDim.x + threadIdx.x] = minout[i*gridDim.x + threadIdx.x];
		}
		else {
			smax[i*blockDim.x + threadIdx.x] = maxout[i*gridDim.x];
			smin[i*blockDim.x + threadIdx.x] = minout[i*gridDim.x];
		}
	}
	__syncthreads();
	s = blockDim.x >> 1;
	while (s != 0) {
		if (threadIdx.x < s) {
			int su = threadIdx.x + s;
			for (int i = 0; i < FUNCTIONQUANTITY; i++) {
				smax[i*blockDim.x + threadIdx.x] = fmax(smax[i*blockDim.x + threadIdx.x], smax[i*blockDim.x + su]);
				smin[i*blockDim.x + threadIdx.x] = fmin(smin[i*blockDim.x + threadIdx.x], smin[i*blockDim.x + su]);
			}
		}
		__syncthreads();
		s >>= 1;
	}
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		retmax[i] = smax[i*blockDim.x];
		retmin[i] = smin[i*blockDim.x];
	}
	__syncthreads();
}

bool checkAxes(double* minsMaxes, Box& p, int numberOfAxes, int*axes) {
	bool flag = true;
	for (int i = 0; i < numberOfAxes; i++) {
		if (!((p.mins[axes[i] - 1] > minsMaxes[i * 2]) && (p.maxs[axes[i] - 1] < minsMaxes[i * 2 + 1]))) {
			flag = false;
			break;
		}
	}
	return flag;
}

int main(int argc, char** argv)
{
	auto begin = std::chrono::high_resolution_clock::now();
	unsigned int start_time = clock();
	dim3 dimblock(BLOCKSIZE);
	dim3 dimgrid(GRIDSIZE);
	//Main box borders
	double mins[VECTORSIZE] = { 0,0};
	double maxs[VECTORSIZE] = { 30,30};

	Box box;
	for (int i = 0; i < VECTORSIZE; i++) {
		box.mins[i] = mins[i];
		box.maxs[i] = maxs[i];
	}
	std::vector<Box> temp;
	std::vector<Box> main;
	std::vector<Box> I;
	std::vector<Box> E;
	main.push_back(box);
	double curD = diam(box);
	double* retmin, *retmax, *rmin, *rmax, *rsmax, *crsmax;
	double* boxes;
	double* cuda_boxes;
	cudaMalloc((void**)&retmin, FUNCTIONQUANTITY*GRIDSIZE* sizeof(double));
	cudaMalloc((void**)&retmax, FUNCTIONQUANTITY*GRIDSIZE* sizeof(double));
	cudaMalloc((void**)&crsmax, FUNCTIONQUANTITY*BLOCKSIZE * sizeof(double));
	cudaMalloc((void**)&cuda_boxes, GRIDSIZE * 2 * VECTORSIZE * sizeof(double));
	rmin = (double*)malloc(sizeof(double) * FUNCTIONQUANTITY*GRIDSIZE);
	rmax = (double*)malloc(sizeof(double) * FUNCTIONQUANTITY*GRIDSIZE);
	boxes = (double*)malloc(GRIDSIZE * 2 * VECTORSIZE * sizeof(double));
	int iter = 0;
	while (curD > ENDCRITERIA && main.size() > 0) {
		while (iter < main.size()) {
			int len = 0;
			for (int i = 0; iter < main.size() && i < GRIDSIZE; i++, iter++) {
				for(int j = 0; j < VECTORSIZE; j++){
					boxes[i*2*VECTORSIZE + j*2] = main[iter].mins[j];
					boxes[i*2*VECTORSIZE + j*2 + 1] = main[iter].maxs[j];
				}
				len++;
			}
			cudaMemcpy(cuda_boxes, boxes, GRIDSIZE * 2 * VECTORSIZE * sizeof(double),cudaMemcpyHostToDevice);
			boostKernel <f1, f2> << <dimgrid, dimblock >> > (cuda_boxes, len, retmax, retmin);
			cudaMemcpy(rmax, retmax, GRIDSIZE*FUNCTIONQUANTITY * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(rmin, retmin, GRIDSIZE*FUNCTIONQUANTITY * sizeof(double), cudaMemcpyDeviceToHost);
			for (int k = 0; k < len; k++) {
				double max = rmax[k * FUNCTIONQUANTITY];
				double min = rmin[k * FUNCTIONQUANTITY];
				for (int i = 1; i < FUNCTIONQUANTITY; i++) {
					max = fmax(rmax[k * FUNCTIONQUANTITY + i - 1], rmax[k*FUNCTIONQUANTITY + i]);
					min = fmax(rmin[k * FUNCTIONQUANTITY + i - 1], rmin[k*FUNCTIONQUANTITY + i]);
				}
				Box t;
				for (int i = 0; i < VECTORSIZE; i++) {
					t.mins[i] = boxes[k * 2 * VECTORSIZE + i * 2];
					t.maxs[i] = boxes[k * 2 * VECTORSIZE + i * 2 + 1];
				}
				if (min > 0) {
					E.push_back(t);
					continue;
				}
				if (max < 0) {
					I.push_back(t);
					continue;
				}
				std::pair<Box, Box> sp = split(t);
				temp.push_back(sp.first);
				temp.push_back(sp.second);
				curD = diam(sp.first);
			}
		}
		iter = 0;
		std::cout << "Main size: " << main.size() << " Cur diam: " << curD << "\n";
		main.clear();
		main.insert(main.begin(), temp.begin(), temp.end());
		temp.clear();
	}
	cudaFree(retmin);
	cudaFree(retmax);
	cudaFree(cuda_boxes);
	free(rmax);
	free(rmin);
	free(boxes);
	unsigned int end_time = clock();
	unsigned int search_time = end_time - start_time;
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns\n";
	std::cout << search_time << "\n";
	std::ofstream myfile;
	myfile.open("out.txt");
	myfile << VECTORSIZE << "\n";
	for (int i = 0; i < VECTORSIZE; i++) {
	    myfile << mins[i] << " " << maxs[i] << "\n";
	}
	myfile << main.size() << "\n";
	for (auto p : main) {
	    for (int i = 0; i < VECTORSIZE; i++) {
	        myfile << p.mins[i] << " " << p.maxs[i] << " ";
	    }
	    myfile << "\n";
	}
	myfile << I.size() << "\n";
	for (auto p : I) {
	    for (int i = 0; i < VECTORSIZE; i++) {
	        myfile << p.mins[i] << " " << p.maxs[i] << " ";
	    }
	    myfile << "\n";
	}
	myfile << E.size() << "\n";
	for (auto p : E) {
	    for (int i = 0; i < VECTORSIZE; i++) {
	        myfile << p.mins[i] << " " << p.maxs[i] << " ";
	    }
	    myfile << "\n";
	}
	myfile.close();
    return 0;
}
