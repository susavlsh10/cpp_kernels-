#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <chrono>

#define DATA_SIZE 8388608
#define INCR_VALUE 2

void print_state (const std::ios& stream) {
  std::cout << " good()=" << stream.good();
  std::cout << " eof()=" << stream.eof();
  std::cout << " fail()=" << stream.fail();
  std::cout << " bad()=" << stream.bad();
  std::cout<<std::endl;
}

int gen()
{
    static int i = 0;
    return ++i;
}

int main(){	
	
	int vec_size[] = {1024, 2048, 4096, 8192, 16384, 32768, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456};
	
	
	char filename[100];
	char filename1[100];
	
	//for (int i=0; i<15; i++){
	
	int data_size = vec_size[14];
	sprintf(filename, "float_%d.dat", data_size);
	sprintf(filename1, "float_%d_a.dat", data_size);
	//std::cout<< "Filename: " <<filename<<std::endl;
	
	std::ifstream fin;

	fin.open(filename, std::ios::in | std::ios::binary);

	if (!fin){
		std::cout<<"File cannot be opened."<<std::endl;
		return 1;
	}
	
	std::ifstream fin1;

	fin1.open(filename1, std::ios::in | std::ios::binary);

	if (!fin1){
		std::cout<<"File cannot be opened."<<std::endl;
		return 1;
	}	
	
	//std::streampos size;
	int* read_buffer = new int[data_size];
	int* buffer_copy = new int[data_size];
	int* out_buffer = new int[data_size];
	
	//std::generate(buffer_copy, buffer_copy+data_size, gen);
	
	auto start = std::chrono::high_resolution_clock::now();
	
	fin.read((char*)read_buffer,sizeof(int)*data_size);
	fin1.read((char*)buffer_copy,sizeof(int)*data_size);
	
	if(!fin.good()) {
      std::cout << "Error occurred at reading time!" << std::endl;
      return 1;
   }	
  	if(!fin1.good()) {
      std::cout << "Error occurred at reading time!" << std::endl;
      return 1;
   } 
   
	auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start);
    std::cout << "SSD->CPU time, " << data_size << ","<<duration1.count() << ", microseconds" << std::endl;	

	
	
	//operation
	/*
	for (int i=0; i<data_size; i++){
		out_buffer[i] = read_buffer[i] + buffer_copy[i];
	}
	auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "CPU execution Time, " << data_size << ","<<duration.count() << ", microseconds" << std::endl;
	*/
	
	fin.close();
	

	
	
	delete[] read_buffer;
	delete[] buffer_copy;
	//}

	
	return 0;
	
}