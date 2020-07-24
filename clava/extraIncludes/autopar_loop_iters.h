#ifndef AUTOPAR_LOOP_ITERS_H
#define AUTOPAR_LOOP_ITERS_H

struct autopar_loop_iters {
	int max_iters;
	int min_iters;
	unsigned long long sum_iters;
	unsigned long long sum_iters_squared;
	int total_calls;
};

extern struct autopar_loop_iters* autopar_loop_iters_array;
extern int autopar_loop_iters_array_size;

void autopar_init_array(int n_loops);
void autopar_array_exit_function();

#endif
