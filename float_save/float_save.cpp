#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

int gen()
{
    static float i = 0;
    return ++i;
}

int main(int argc, char** argv){
	int vec_size[] = {1024, 2048, 4096, 8192, 16384, 32768, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456};
	
	//write the data to the file
	for (int i=0; i<15; i++){
		//std::cout<<vec_size[i] << ",";
		
		int data_size = vec_size[i];
		//create a int vector of size vec_size
		float* data = new float[vec_size[i]];
		
		//fill the vector randomly
		std::generate(data, data+data_size, gen);
		std::cout<<"Data written:" <<std::endl;
		/*
		for (int j=0; j<data_size; j++){
			std::cout<<data[j]<<" ";
		}
		std::cout<<std::endl;
		*/
		
		//char* write_buffer = new char[sizeof(int) * data_size];
		
		std::ofstream fout;
		char filename[100];
		sprintf(filename, "float_%d.dat", data_size);

		fout.open(filename, std::ios::out | std::ios::binary);

		if (!fout){
			std::cout<< "Cannot open file." <<std::endl;
			return 1;
		}
		fout.write((char*)data, sizeof(float) * data_size);
		fout.close();		
		
		
		delete[] data;
		//delete[] write_buffer;

	}
	//std::cout<<std::endl;
	
	//------------------------------------------------------------------//
	//now read the data back 
}