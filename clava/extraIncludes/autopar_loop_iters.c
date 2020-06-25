#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include "autopar_loop_iters.h"

struct autopar_loop_iters* autopar_loop_iters_array = NULL;
int autopar_loop_iters_array_size = 0;

void autopar_init_array(int n_loops) {
	int autopar_n_threads = omp_get_max_threads();
	autopar_loop_iters_array_size = autopar_n_threads * n_loops;
	
	autopar_loop_iters_array  = (struct autopar_loop_iters*) malloc(autopar_loop_iters_array_size * sizeof(struct autopar_loop_iters));
	if(autopar_loop_iters_array == NULL) {
	   	printf("autopar_loop_iters_array is NULL\n");
	   	exit(-1);
	}
	
	struct autopar_loop_iters autopar_loop_iters_default = { 0, 0, INT_MAX, 0, 0, 0 };
	for(int i = 0; i < autopar_loop_iters_array_size; i++) {
		autopar_loop_iters_array[i] = autopar_loop_iters_default;
	}
}

void autopar_array_exit_function() {	
	for(int i = 0; i < autopar_loop_iters_array_size; i++) {	
		printf("autopar_loop_iters[%d] = { %d, %d, %llu, %llu, %d }\n", i,
		 autopar_loop_iters_array[i].max_iters,
		 autopar_loop_iters_array[i].min_iters,
		 autopar_loop_iters_array[i].sum_iters,
		 autopar_loop_iters_array[i].sum_iters_squared,
		 autopar_loop_iters_array[i].total_calls);
	}
	
	free(autopar_loop_iters_array);
}

