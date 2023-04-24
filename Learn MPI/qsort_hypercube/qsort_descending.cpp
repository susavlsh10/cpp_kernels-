#include <cmath>
#include <cstdlib>
#include <cstdio>

#define LIST_SIZE 20

int compare_int(const void *a0, const void *b0) {
    int a = *(int *)a0;
    int b = *(int *)b0;
    if (a < b) {
	return 1; 
    } else if (a > b) {
	return -1;
    } else {
	return 0;
    }
}

int main(int argc, char*argv[]){

    int *arr = new int[LIST_SIZE];

    for(int i=0; i<LIST_SIZE; i++){
        arr[i] = i;
    }

    //print here
    printf("Original List \n");
    for(int i =0; i<LIST_SIZE; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
    //sort here 
    qsort(arr, LIST_SIZE, sizeof(int), compare_int);

    //print here
    printf("Sorted List \n");
    for(int i =0; i<LIST_SIZE; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
    //print here 
    return 0;
}