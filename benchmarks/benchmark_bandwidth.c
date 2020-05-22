/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif


#include "../include/blasfeo.h"
#include "benchmark_x_common.h"



int main()
	{

#if !defined(BENCHMARKS_MODE)
	printf("\n\n Recompile BLASFEO with BENCHMARKS_MODE=1 to run this benchmark.\n");
	printf("On CMake use -DBLASFEO_BENCHMARKS=ON .\n\n");
	return 0;
#endif

	int nn, ii, rep, nrep;

	blasfeo_timer timer;
	double tmp_time, bandwidth_set_sequential, bandwidth_copy_sequential, bandwidth_set_parallel, bandwidth_copy_parallel;

	int size[] = {16, 32, 48, 64, 80, 96, 112, 128, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 8192, 12288, 16384, 24576, 32768, 40960, 49152, 65536, 81920, 98304, 131072};
	int nnrep[] = {4000000, 4000000, 4000000, 2000000, 2000000, 1000000, 1000000, 1000000, 400000, 400000, 400000, 200000, 200000, 200000, 100000, 100000, 100000, 40000, 40000, 40000, 20000, 20000, 20000, 10000, 10000, 10000, 10000, 4000, 4000, 4000, 2000, 2000, 2000};
	int n_size = 33;

#ifdef _OPENMP
int num_threads = 2;
omp_set_num_threads(num_threads);
printf("\nusing %d threads\n", num_threads);
#endif


	printf("\nsize (KB)\tset sequential\tset parrallel\tspeedup\t\tcopy sequential\tcopy parrallel\tspeedup\n");
	for(nn=0; nn<n_size; nn++)
		{

		nrep = nnrep[nn];

		double *x = malloc(size[nn]*sizeof(double));
		double *y = malloc(size[nn]*sizeof(double));

int id = -1;

		/* set benchmark sequential */

		// set to zero
		blasfeo_tic(&timer);

		for(rep=0; rep<nrep; rep++)
			{

			for(ii=0; ii<size[nn]; ii++)
				{
				x[ii] = 0.0;
				}

			}

		tmp_time = blasfeo_toc(&timer) / nrep;

		bandwidth_set_sequential = size[nn] * 8.0 / tmp_time;


		/* set benchmark parallel */

#pragma omp parallel private(id, rep, ii)
{
//id = omp_get_thread_num();
//printf("\nid %d\n", id);

#pragma omp master
{
		// set to zero
		blasfeo_tic(&timer);
} // master

#pragma omp for schedule(static)
		for(rep=0; rep<nrep; rep++)
			{

			for(ii=0; ii<size[nn]; ii++)
				{
				x[ii] = 0.0;
				}

			}

#pragma omp master
{
		tmp_time = blasfeo_toc(&timer) / nrep;
} // master
} // parallel

		bandwidth_set_parallel = size[nn] * 8.0 / tmp_time;


		/* copy benchmark sequential */

		// copy
		blasfeo_tic(&timer);

		for(rep=0; rep<nrep; rep++)
			{

			for(ii=0; ii<size[nn]; ii++)
				{
				y[ii] = x[ii];
				}

			}

		tmp_time = blasfeo_toc(&timer) / nrep;

		bandwidth_copy_sequential = size[nn] * 2 * 8.0 / tmp_time;


		/* copy benchmark parallel */

#pragma omp parallel private(id, rep, ii)
{
//id = omp_get_thread_num();
//printf("\nid %d\n", id);

#pragma omp master
{
		// copy
		blasfeo_tic(&timer);
} // master

#pragma omp for schedule(static)
		for(rep=0; rep<nrep; rep++)
			{

			for(ii=0; ii<size[nn]; ii++)
				{
				y[ii] = x[ii];
				}

			}

#pragma omp master
{
		tmp_time = blasfeo_toc(&timer) / nrep;
} // master
} // parallel

		bandwidth_copy_parallel = size[nn] * 2 * 8.0 / tmp_time;


		// print
		printf("%f\t%e\t%e\t%f\t%e\t%e\t%f\n", size[nn]*8.0/1024.0, bandwidth_set_sequential, bandwidth_set_parallel, bandwidth_set_parallel/bandwidth_set_sequential, bandwidth_copy_sequential, bandwidth_copy_parallel, bandwidth_copy_parallel/bandwidth_copy_sequential);

		free(x);
		free(y);

		}
	
	return 0;

	}
