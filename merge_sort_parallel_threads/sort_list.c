//
// Sorts a list using multiple threads
//

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#define MAX_THREADS     65536
#define MAX_LIST_SIZE   1000000000

#define DEBUG 0

// Thread variables
//
// VS: ... declare thread variables, mutexes, condition varables, etc.,
// VS: ... as needed for this assignment 
//
pthread_t *THREADS;
pthread_mutex_t COUNT_MUTEX;
pthread_cond_t count_threshold;
pthread_attr_t attr;
int count;

typedef struct 
{
    /* Thread data */
    int ptr_my_id;
    int my_list_size;
} LocalSortData;


typedef struct 
{
    /* Thread data */
    int level_t;
    int my_id_t;
    int *ptr_t;
    int np_t;

} MergeSortData;


// Global variables
int num_threads;		// Number of threads to create - user input 
int list_size;			// List size
int *list;			// List of values
int *work;			// Work array
int *list_orig;			// Original list of values, used for error checking

LocalSortData *LSD;
MergeSortData *MSD;

// Comparison routine for qsort (stdlib.h) which is used to sort
// a thread's sub-list at the start of the algorithm
int compare_int(const void *a0, const void *b0) {
    int a = *(int *)a0;
    int b = *(int *)b0;
    if (a < b) {
        return -1;
    } else if (a > b) {
        return 1;
    } else {
        return 0;
    }
}

// Print list - for debugging
void print_list(int *list, int list_size) {
    int i;
    for (i = 0; i < list_size; i++) {
        printf("[%d] \t %16d\n", i, list[i]); 
    }
    printf("--------------------------------------------------------------------\n"); 
}



// Return index of first element larger than or equal to v in sorted list
// ... return last if all elements are smaller than v
// ... elements in list[first], list[first+1], ... list[last-1]
//
//   int idx = first; while ((v > list[idx]) && (idx < last)) idx++;
//
int binary_search_lt(int v, int *list, int first, int last) {
   
    // Linear search code
    // int idx = first; while ((v > list[idx]) && (idx < last)) idx++; return idx;

    int left = first; 
    int right = last-1; 

    if (list[left] >= v) return left;
    if (list[right] < v) return right+1;
    int mid = (left+right)/2; 
    while (mid > left) {
        if (list[mid] < v) {
	    left = mid; 
	} else {
	    right = mid;
	}
	mid = (left+right)/2;
    }
    return right;
}
// Return index of first element larger than v in sorted list
// ... return last if all elements are smaller than or equal to v
// ... elements in list[first], list[first+1], ... list[last-1]
//
//   int idx = first; while ((v >= list[idx]) && (idx < last)) idx++;
//
int binary_search_le(int v, int *list, int first, int last) {

    // Linear search code
    // int idx = first; while ((v >= list[idx]) && (idx < last)) idx++; return idx;
 
    int left = first; 
    int right = last-1; 

    if (list[left] > v) return left; 
    if (list[right] <= v) return right+1;
    int mid = (left+right)/2; 
    while (mid > left) {
        if (list[mid] <= v) {
	    left = mid; 
	} else {
	    right = mid;
	}
	mid = (left+right)/2;
    }
    return right;
}

void *SortLocalList(void *arg){
    /* Thread method to sort the local list using qsort */
    int ptr_my_id_t;
    int my_list_size_t;
    
    LocalSortData* data = (LocalSortData *) arg;
    ptr_my_id_t = data->ptr_my_id;
    my_list_size_t = data->my_list_size;
    //int list_t[my_list_size_t];

    /*
        int *list_t = malloc(my_list_size_t*sizeof(int));

    // copy the values of the local list to a temporary buffer 
    for (int i = 0; i< my_list_size_t; i++){ 
        list_t[i] = list[ptr_my_id_t + i]; 
    }
    
    */


    qsort(&list[ptr_my_id_t], my_list_size_t, sizeof(int), compare_int);

    //copy the sorted list back to the original list
    /*
        for (int i = 0; i< my_list_size_t; i++){ 
        list[ptr_my_id_t + i] = list_t[i]; 
    }
    
    */

    pthread_exit(NULL);
    
}

void *ParallelMergeSort(void *args)
{
/* Thread method to perform merge sort in parallel. */
    int level;
    int my_id;
    int *ptr;
    int np;

    MergeSortData* data = (MergeSortData *) args;
    level = data->level_t;
    my_id = data->my_id_t;
    ptr = data->ptr_t;
    np = data->np_t;

    int my_blk_size = np * (1 << level); 

    int my_own_blk = ((my_id >> level) << level);
    int my_own_idx = ptr[my_own_blk];

    int my_search_blk = ((my_id >> level) << level) ^ (1 << level);
    int my_search_idx = ptr[my_search_blk];
    int my_search_idx_max = my_search_idx+my_blk_size;

    int my_write_blk = ((my_id >> (level+1)) << (level+1));
    int my_write_idx = ptr[my_write_blk];

    int idx = my_search_idx;
    
    int my_search_count = 0;


    // Binary search for 1st element
    if (my_search_blk > my_own_blk) {
        idx = binary_search_lt(list[ptr[my_id]], list, my_search_idx, my_search_idx_max); 
    } else {
        idx = binary_search_le(list[ptr[my_id]], list, my_search_idx, my_search_idx_max); 
    }
    my_search_count = idx - my_search_idx;
    int i_write = my_write_idx + my_search_count + (ptr[my_id]-my_own_idx); 

    work[i_write] = list[ptr[my_id]];


    // Linear search for 2nd element onwards
    for (int i = ptr[my_id]+1; i < ptr[my_id+1]; i++) {
        if (my_search_blk > my_own_blk) {
        while ((list[i] > list[idx]) && (idx < my_search_idx_max)) {
            idx++; my_search_count++;
        }
    } else {
        while ((list[i] >= list[idx]) && (idx < my_search_idx_max)) {
            idx++; my_search_count++;
        }
    }
    i_write = my_write_idx + my_search_count + (i-my_own_idx); 
    work[i_write] = list[i];
    }

    //Wait for all the threads to finish computation
    pthread_mutex_lock (&COUNT_MUTEX);                //lock
    count++;

    while (count < num_threads){    //rechecking predicate
        pthread_cond_wait(&count_threshold, &COUNT_MUTEX);
    }
    if (count == num_threads){
        pthread_cond_broadcast(&count_threshold);
        count++;
    }
    pthread_mutex_unlock (&COUNT_MUTEX);              //unlock 
    

    //Update the list in parallel
    for (int i = ptr[my_id]; i < ptr[my_id+1]; i++) {
        list[i] = work[i];
    }   

    pthread_exit(NULL);
}

// Sort list via parallel merge sort
//
// VS: ... to be parallelized using threads ...
//
void sort_list(int q) {

    int i, level, my_id; 
    int np, my_list_size; 
    int ptr[num_threads+1];

    int my_own_blk, my_own_idx;
    int my_blk_size, my_search_blk, my_search_idx, my_search_idx_max;
    int my_write_blk, my_write_idx;
    int my_search_count; 
    int idx, i_write; 
    
    np = list_size/num_threads; 	// Sub list size 

    // Initialize starting position for each sublist
    for (my_id = 0; my_id < num_threads; my_id++) {
        ptr[my_id] = my_id * np;
    }
    ptr[num_threads] = list_size;

    // Sort local lists
    

    for (my_id = 0; my_id < num_threads; my_id++) {
        /* Create threads to sort local lists in parallel */
        LSD[my_id].my_list_size = ptr[my_id+1]-ptr[my_id];
        LSD[my_id].ptr_my_id = ptr[my_id];
        
        pthread_create(&THREADS[my_id], &attr, SortLocalList, (void *) &LSD[my_id]);
        
        //my_list_size = ptr[my_id+1]-ptr[my_id];
        //qsort(&list[ptr[my_id]], my_list_size, sizeof(int), compare_int);
    }

    //wait for the threads to join
    int rc;
    long t;
    void *status;
    for(t=0; t<num_threads; t++) {
      rc = pthread_join(THREADS[t], &status);
      if (rc) {
         printf("ERROR; return code from pthread_join() is %d\n", rc);
         exit(-1);
         }
      //printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
      }

    if (DEBUG) {
        printf("List after local sort \n");
        print_list(list, list_size); 
        }

    // Sort list
    
    for (level = 0; level < q; level++) {
        count = 0;
        // Each thread scatters its sub_list into work array
        for (my_id = 0; my_id < num_threads; my_id++) {
            //create and branch threads to merge sort
            MSD[my_id].level_t = level;
            MSD[my_id].my_id_t = my_id;
            MSD[my_id].ptr_t = ptr;
            MSD[my_id].np_t = np;
            pthread_create(&THREADS[my_id], &attr, ParallelMergeSort, (void *) &MSD[my_id]);
        }

        // wait for all threads to finish
        for (t = 0; t < num_threads; t++) {
            rc = pthread_join(THREADS[t], &status);
            if (rc) {
                printf("ERROR; return code from pthread_join() is %d\n", rc);
                exit(-1);
                }
                //printf("Main: completed join with thread %ld having a status of %ld\n", t,(long)status);
            }
    }
    if (DEBUG){
        printf("List after Merge sort \n");
        print_list(list, list_size);
    } 
    /*  Free the dynamic thread arrays    */
    free(LSD);
    free(MSD);
}

// Main program - set up list of random integers and use threads to sort the list
//
// Input: 
//	k = log_2(list size), therefore list_size = 2^k
//	q = log_2(num_threads), therefore num_threads = 2^q
//
int main(int argc, char *argv[]) {

    struct timespec start, stop, stop_qsort;
    double total_time, time_res, total_time_qsort;
    int k, q, j, error; 

    // Read input, validate
    if (argc != 3) {
	printf("Need two integers as input \n"); 
	printf("Use: <executable_name> <log_2(list_size)> <log_2(num_threads)>\n"); 
	exit(0);
    }
    k = atoi(argv[argc-2]);
    if ((list_size = (1 << k)) > MAX_LIST_SIZE) {
	printf("Maximum list size allowed: %d.\n", MAX_LIST_SIZE);
	exit(0);
    }; 
    q = atoi(argv[argc-1]);
    if ((num_threads = (1 << q)) > MAX_THREADS) {
	printf("Maximum number of threads allowed: %d.\n", MAX_THREADS);
	exit(0);
    }; 
    if (num_threads > list_size) {
	printf("Number of threads (%d) < list_size (%d) not allowed.\n", 
	   num_threads, list_size);
	exit(0);
    }; 

    // Allocate list, list_orig, and work

    list = (int *) malloc(list_size * sizeof(int));
    list_orig = (int *) malloc(list_size * sizeof(int));
    work = (int *) malloc(list_size * sizeof(int));

//
// VS: ... May need to initialize mutexes, condition variables, 
// VS: ... and their attributes
    THREADS = (pthread_t *) malloc(num_threads * sizeof(pthread_t));

    pthread_mutex_init(&COUNT_MUTEX, NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_cond_init (&count_threshold, NULL);

    LSD= (LocalSortData *) malloc (num_threads* sizeof(LocalSortData));
    MSD = (MergeSortData *) malloc(num_threads* sizeof(MergeSortData));

    //Manage thread stack size
    /*
    size_t stacksize;
    pthread_attr_getstacksize (&attr, &stacksize);
    //printf("Default stack size = %li\n", stacksize);
    stacksize = list_size * sizeof(int) * 3;
    //printf("Amount of stack needed per thread = %li\n",stacksize);
    pthread_attr_setstacksize (&attr, stacksize);
    
    */



//

    // Initialize list of random integers; list will be sorted by 
    // multi-threaded parallel merge sort
    // Copy list to list_orig; list_orig will be sorted by qsort and used
    // to check correctness of multi-threaded parallel merge sort
    srand48(0); 	// seed the random number generator
    for (j = 0; j < list_size; j++) {
	list[j] = (int) lrand48();
	list_orig[j] = list[j];
    }
    // duplicate first value at last location to test for repeated values
    list[list_size-1] = list[0]; list_orig[list_size-1] = list_orig[0];

    if (DEBUG){
        printf("Original List\n");
        print_list(list_orig, list_size);
    } 
    // Create threads; each thread executes find_minimum
    clock_gettime(CLOCK_REALTIME, &start);

//
// VS: ... may need to initialize mutexes, condition variables, and their attributes
//

// Serial merge sort 
// VS: ... replace this call with multi-threaded parallel routine for merge sort
// VS: ... need to create threads and execute thread routine that implements 
// VS: ... parallel merge sort

    sort_list(q);

    // Compute time taken
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000000001*(stop.tv_nsec-start.tv_nsec);

    // Check answer
    qsort(list_orig, list_size, sizeof(int), compare_int);
    clock_gettime(CLOCK_REALTIME, &stop_qsort);
    total_time_qsort = (stop_qsort.tv_sec-stop.tv_sec)
	+0.000000001*(stop_qsort.tv_nsec-stop.tv_nsec);

    error = 0; 
    for (j = 1; j < list_size; j++) {
	if (list[j] != list_orig[j]) error = 1; 
    }

    if (error != 0) {
	printf("Houston, we have a problem!\n"); 
    }

    // Print time taken
    printf("List Size = %d, Threads = %d, error = %d, time (sec) = %8.4f, qsort_time = %8.4f\n", 
	    list_size, num_threads, error, total_time, total_time_qsort);

// VS: ... destroy mutex, condition variables, etc.
    
    pthread_exit(NULL);
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&COUNT_MUTEX);

    free(list); free(work); free(list_orig); 

}

