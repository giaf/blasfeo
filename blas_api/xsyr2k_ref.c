/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS for embedded optimization.                                                      *
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

//#define TIME_INT


#ifdef TIME_INT
#include <blasfeo_timing.h>
#endif



void SYR2K(char *uplo, char *trans, int *pm, int *pk, REAL *palpha, REAL *A, int *plda, REAL *B, int *pldb, REAL *pbeta, REAL *C, int *pldc)
	{

#ifdef TIME_INT
    blasfeo_timer timer;
	blasfeo_tic(&timer);
#endif

#if defined(DIM_CHECK)
	if( !(*uplo=='l' | *uplo=='L' | *uplo=='u' | *uplo=='U') )
		{
#ifdef EXT_DEP
		printf("\nBLASFEO: syr2k: wrong value for uplo\n");
#endif
		return;
		}
	if( !(*trans=='c' | *trans=='C' | *trans=='n' | *trans=='N' | *trans=='t' | *trans=='T') )
		{
#ifdef EXT_DEP
		printf("\nBLASFEO: syr2k: wrong value for trans\n");
#endif
		return;
		}
#endif

#if defined(FALLBACK_TO_EXTERNAL_BLAS)
	// TODO
#endif

	struct MAT sA;
	sA.pA = A;
	sA.m = *plda;

	struct MAT sB;
	sB.pA = B;
	sB.m = *pldb;

	struct MAT sC;
	sC.pA = C;
	sC.m = *pldc;

	if(*uplo=='l' | *uplo=='L')
		{
		if(*trans=='n' | *trans=='N')
			{
			SYR2K_LN(*pm, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		else
			{
			SYR2K_LT(*pm, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		}
	else
		{
		if(*trans=='n' | *trans=='N')
			{
			SYR2K_UN(*pm, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		else
			{
			SYR2K_UT(*pm, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		}

#ifdef TIME_INT
	double flops = 2.0 * *pm * *pm * *pk;
	double time = blasfeo_toc(&timer);
	double Gflops = 1e-9 * flops / time;
	double Gflops_max = 3.4 * 16;
#ifdef EXT_DEP
    printf("\nblasfeo syrk\t%c\t%c\t%d\t%d\t%f\t%f\n", *uplo, *trans, *pm, *pk, Gflops, 100.0*Gflops/Gflops_max);
#endif
#endif

	return;

	}


