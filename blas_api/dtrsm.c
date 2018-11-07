/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_target.h"
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"



#if defined(FORTRAN_BLAS_API)
#define blasfeo_dtrsm dtrsm_
#endif



void blasfeo_dtrsm(char *side, char *uplo, char *transa, char *diag, int *pm, int *pn, double *alpha, double *A, int *plda, double *B, int *pldb)
	{

	int m = *pm;
	int n = *pn;
	int lda = *plda;
	int ldb = *pldb;

	int ii, jj;

	int ps = 4;

	if(m<=0 | n<=0)
		return;

// TODO visual studio alignment
#if defined(TARGET_GENERIC)
	double pd0[K_MAX_STACK];
#else
	double pd0[K_MAX_STACK] __attribute__ ((aligned (64)));
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	double pU0[3*4*K_MAX_STACK] __attribute__ ((aligned (64)));
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE) | defined(TARGET_ARMV8A_ARM_CORTEX_A57)
	double pU0[2*4*K_MAX_STACK] __attribute__ ((aligned (64)));
#elif defined(TARGET_GENERIC)
	double pU0[1*4*K_MAX_STACK];
#else
	double pU0[1*4*K_MAX_STACK] __attribute__ ((aligned (64)));
#endif

	int k0;
	// TODO update if necessary !!!!!
	if(*side=='l' | *side=='L')
		k0 = m;
	else
		k0 = n;

	int sdu0 = (k0+3)/4*4;
	sdu0 = sdu0<K_MAX_STACK ? sdu0 : K_MAX_STACK;

	struct blasfeo_dmat sA, sB;
	double *pU, *pB, *dA, *dB;
	int sda, sdb, sdu;
	int sA_size, sB_size;
	void *mem, *mem_align;
	int m1, n1;


	if(*side=='l' | *side=='L') // _l
		{
		if(*uplo=='l' | *uplo=='L') // _ll
			{
			if(*transa=='n' | *transa=='N') // _lln
				{
				if(*diag=='n' | *diag=='N') // _llnn
					{
					goto llnn;
					}
				else if(*diag=='u' | *diag=='U') // _llnu
					{
					printf("\nBLASFEO: dtrmm_llnu: not implemented yet\n");
					return;
					}
				}
			else if(*transa=='t' | *transa=='T' | *transa=='c' | *transa=='C') // _llt
				{
				if(*diag=='n' | *diag=='N') // _lltn
					{
					printf("\nBLASFEO: dtrmm_lltn: not implemented yet\n");
					return;
					}
				else if(*diag=='u' | *diag=='U') // _lltu
					{
					printf("\nBLASFEO: dtrmm_lltu: not implemented yet\n");
					return;
					}
				}
			else
				{
				printf("\nBLASFEO: dtrmm: wrong value for transa\n");
				return;
				}
			}
		else if(*uplo=='u' | *uplo=='U') // _lu
			{
			if(*transa=='n' | *transa=='N') // _lun
				{
				if(*diag=='n' | *diag=='N') // _lunn
					{
					printf("\nBLASFEO: dtrmm_lunn: not implemented yet\n");
					return;
					}
				else if(*diag=='u' | *diag=='U') // _lunu
					{
					printf("\nBLASFEO: dtrmm_lunu: not implemented yet\n");
					return;
					}
				}
			else if(*transa=='t' | *transa=='T' | *transa=='c' | *transa=='C') // _lut
				{
				if(*diag=='n' | *diag=='N') // _lutn
					{
					printf("\nBLASFEO: dtrmm_lutn: not implemented yet\n");
					return;
					}
				else if(*diag=='u' | *diag=='U') // _lutu
					{
					printf("\nBLASFEO: dtrmm_lutu: not implemented yet\n");
					return;
					}
				}
			else
				{
				printf("\nBLASFEO: dtrmm: wrong value for transa\n");
				return;
				}
			}
		else
			{
			printf("\nBLASFEO: dtrmm: wrong value for uplo\n");
			return;
			}
		}
	else if(*side=='r' | *side=='R') // _r
		{
		if(*uplo=='l' | *uplo=='L') // _rl
			{
			if(*transa=='n' | *transa=='N') // _rln
				{
				if(*diag=='n' | *diag=='N') // _rlnn
					{
					printf("\nBLASFEO: dtrmm_rlnn: not implemented yet\n");
					return;
					}
				else if(*diag=='u' | *diag=='U') // _rlnu
					{
					printf("\nBLASFEO: dtrmm_rlnu: not implemented yet\n");
					return;
					}
				}
			else if(*transa=='t' | *transa=='T' | *transa=='c' | *transa=='C') // _rlt
				{
				if(*diag=='n' | *diag=='N') // _rltn
					{
					goto rltn;
					}
				else if(*diag=='u' | *diag=='U') // _rltu
					{
					printf("\nBLASFEO: dtrmm_rltu: not implemented yet\n");
					return;
					}
				}
			else
				{
				printf("\nBLASFEO: dtrmm: wrong value for transa\n");
				return;
				}
			}
		else if(*uplo=='u' | *uplo=='U') // _ru
			{
			if(*transa=='n' | *transa=='N') // _run
				{
				if(*diag=='n' | *diag=='N') // _runn
					{
					printf("\nBLASFEO: dtrmm_runn: not implemented yet\n");
					return;
					}
				else if(*diag=='u' | *diag=='U') // _runu
					{
					printf("\nBLASFEO: dtrmm_runu: not implemented yet\n");
					return;
					}
				}
			else if(*transa=='t' | *transa=='T' | *transa=='c' | *transa=='C') // _rut
				{
				if(*diag=='n' | *diag=='N') // _rutn
					{
					printf("\nBLASFEO: dtrmm_rutn: not implemented yet\n");
					return;
					}
				else if(*diag=='u' | *diag=='U') // _rutu
					{
					printf("\nBLASFEO: dtrmm_rutu: not implemented yet\n");
					return;
					}
				}
			else
				{
				printf("\nBLASFEO: dtrmm: wrong value for transa\n");
				return;
				}
			}
		else
			{
			printf("\nBLASFEO: dtrmm: wrong value for uplo\n");
			return;
			}
		}
	else
		{
		printf("\nBLASFEO: dtrmm: wrong value for side\n");
		return;
		}


/***********************
* llnn
***********************/
llnn:
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>=128 | n>=128 | m>K_MAX_STACK) // XXX cond on m !!!!!
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if(m>=64 | n>=64 | m>K_MAX_STACK) // XXX cond on m !!!!!
#else
	if(m>=12 | n>=12 | m>K_MAX_STACK) // XXX cond on m !!!!!
#endif
		{
		goto llnn_1;
		}
	else
		{
		goto llnn_0;
		}

llnn_0:
	// XXX limits of ii and jj swapped !!!
	pU = pU0;
	sdu = sdu0;
	dA = pd0;

	for(ii=0; ii<m; ii++)
		dA[ii] = 1.0/A[ii+ii*lda];

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<n-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(m, B+(ii+4)*ldb, ldb, pU+4*sdu);
		kernel_dpack_tn_4_lib4(m, B+(ii+8)*ldb, ldb, pU+8*sdu);
		for(jj=0; jj<m-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4c4c(jj, pU, sdu, A+jj, lda, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, A+jj+jj*lda, lda, dA+jj);
			}
		if(jj<m)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib4c4c(jj, pU, sdu, A+jj, lda, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, A+jj+jj*lda, lda, dA+jj, n-ii, m-jj);
			}
		kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
		kernel_dunpack_nt_4_lib4(m, pU+4*sdu, B+(ii+4)*ldb, ldb);
		kernel_dunpack_nt_4_lib4(m, pU+8*sdu, B+(ii+8)*ldb, ldb);
		}
	if(ii<n)
		{
		if(n-ii<=4)
			{
			goto llnn_0_left_4;
			}
		if(n-ii<=8)
			{
			goto llnn_0_left_8;
			}
		else
			{
			goto llnn_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<n-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(m, B+(ii+4)*ldb, ldb, pU+ps*sdu);
		for(jj=0; jj<m-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4c4c(jj, pU, sdu, A+jj, lda, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, A+jj+jj*lda, lda, dA+jj);
			}
		if(jj<m)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib4c4c(jj, pU, sdu, A+jj, lda, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, A+jj+jj*lda, lda, dA+jj, n-ii, m-jj);
			}
		kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
		kernel_dunpack_nt_4_lib4(m, pU+ps*sdu, B+(ii+4)*ldb, ldb);
		}
	if(ii<n)
		{
		if(n-ii<=4)
			{
			goto llnn_0_left_4;
			}
		else
			{
			goto llnn_0_left_8;
			}
		}
#else
	for(; ii<n-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
		for(jj=0; jj<m-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib4c4c(jj, pU, A+jj, lda, alpha, pU+jj*ps, pU+jj*ps, A+jj+jj*lda, lda, dA+jj);
			}
		if(jj<m)
			{
			kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c(jj, pU, A+jj, lda, alpha, pU+jj*ps, pU+jj*ps, A+jj+jj*lda, lda, dA+jj, n-ii, m-jj);
			}
		kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
		}
	if(ii<m)
		{
		goto llnn_0_left_4;
		}
#endif
	goto llnn_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
llnn_0_left_12:
	kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
	kernel_dpack_tn_4_lib4(m, B+(ii+4)*ldb, ldb, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(m, B+(ii+8)*ldb, ldb, pU+8*sdu, n-(ii+8));
	for(jj=0; jj<m; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib4c4c(jj, pU, sdu, A+jj, lda, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, A+jj+jj*lda, lda, dA+jj, n-ii, m-jj);
		}
	kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
	kernel_dunpack_nt_4_lib4(m, pU+4*sdu, B+(ii+4)*ldb, ldb);
	kernel_dunpack_nt_4_vs_lib4(m, pU+8*sdu, B+(ii+8)*ldb, ldb, n-(ii+8));
goto llnn_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
llnn_0_left_8:
	kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
	kernel_dpack_tn_4_vs_lib4(m, B+(ii+4)*ldb, ldb, pU+ps*sdu, n-(ii+4));
	for(jj=0; jj<m; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib4c4c(jj, pU, sdu, A+jj, lda, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, A+jj+jj*lda, lda, dA+jj, n-ii, m-jj);
		}
	kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
	kernel_dunpack_nt_4_vs_lib4(m, pU+ps*sdu, B+(ii+4)*ldb, ldb, n-(ii+4));
goto llnn_0_return;
#endif

llnn_0_left_4:
	kernel_dpack_tn_4_vs_lib4(m, B+ii*ldb, ldb, pU, m-ii);
	for(jj=0; jj<m; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib4c4c(jj, pU, A+jj, lda, alpha, pU+jj*ps, pU+jj*ps, A+jj+jj*lda, lda, dA+jj, n-ii, m-jj);
		}
	kernel_dunpack_nt_4_vs_lib4(m, pU, B+ii*ldb, ldb, m-ii);
goto llnn_0_return;

llnn_0_return:
	return;



llnn_1:
	// XXX limits of ii and jj swapped !!!
	m1 = (m+128-1)/128*128;
	sA_size = blasfeo_memsize_dmat(12, m1);
	sB_size = blasfeo_memsize_dmat(m1, m1);
	mem = malloc(sA_size+sB_size+64);
	blasfeo_align_64_byte(mem, &mem_align);
	blasfeo_create_dmat(12, m, &sA, mem_align);
	blasfeo_create_dmat(m, m, &sB, mem_align+sA_size);

	// TODO pack using loops over kernels !!!!!!!!!!!!!!!!!!!!!!!!!
//	blasfeo_pack_dmat(n, n, A, lda, &sB, 0, 0);
	blasfeo_pack_l_dmat(m, m, A, lda, &sB, 0, 0);
//	blasfeo_print_dmat(n, n, &sB, 0, 0);

	pU = sA.pA;
	sdu = sA.cn;
	pB = sB.pA;
	sdb = sB.cn;
	dB = sB.dA;

	for(ii=0; ii<m; ii++)
		dB[ii] = 1.0/A[ii+ii*lda];

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(; ii<n-11; ii+=12)
		{
		kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(m, B+(ii+4)*ldb, ldb, pU+4*sdu);
		kernel_dpack_tn_4_lib4(m, B+(ii+8)*ldb, ldb, pU+8*sdu);
		for(jj=0; jj<m-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4(jj, pU, sdu, sB.pA+jj*sdb, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, sB.pA+jj*ps+jj*sdb, dB+jj);
			}
		if(jj<m)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(jj, pU, sdu, sB.pA+jj*sdb, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, sB.pA+jj*ps+jj*sdb, dB+jj, n-ii, m-jj);
			}
		kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
		kernel_dunpack_nt_4_lib4(m, pU+4*sdu, B+(ii+4)*ldb, ldb);
		kernel_dunpack_nt_4_lib4(m, pU+8*sdu, B+(ii+8)*ldb, ldb);
		}
	if(ii<n)
		{
		if(n-ii<=4)
			{
			goto llnn_1_left_4;
			}
		if(n-ii<=8)
			{
			goto llnn_1_left_8;
			}
		else
			{
			goto llnn_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(; ii<n-7; ii+=8)
		{
		kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
		kernel_dpack_tn_4_lib4(m, B+(ii+4)*ldb, ldb, pU+4*sdu);
		for(jj=0; jj<m-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4(jj, pU, sdu, sB.pA+jj*sdb, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, sB.pA+jj*ps+jj*sdb, dB+jj);
			}
		if(jj<m)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(jj, pU, sdu, sB.pA+jj*sdb, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, sB.pA+jj*ps+jj*sdb, dB+jj, n-ii, m-jj);
			}
		kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
		kernel_dunpack_nt_4_lib4(m, pU+4*sdu, B+(ii+4)*ldb, ldb);
		}
	if(ii<n)
		{
		if(n-ii<=4)
			{
			goto llnn_1_left_4;
			}
		else
			{
			goto llnn_1_left_8;
			}
		}
#else
	for(; ii<n-3; ii+=4)
		{
		kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
		for(jj=0; jj<m-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib4(jj, pU, sB.pA+jj*sdb, alpha, pU+jj*ps, pU+jj*ps, sB.pA+jj*ps+jj*sdb, dB+jj);
			}
		if(jj<m)
			{
			kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(jj, pU, sB.pA+jj*sdb, alpha, pU+jj*ps, pU+jj*ps, sB.pA+jj*ps+jj*sdb, dB+jj, n-ii, m-jj);
			}
		kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
		}
	if(ii<m)
		{
		goto llnn_1_left_4;
		}
#endif
	goto llnn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
llnn_1_left_12:
	kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
	kernel_dpack_tn_4_lib4(m, B+(ii+4)*ldb, ldb, pU+4*sdu);
	kernel_dpack_tn_4_vs_lib4(m, B+(ii+8)*ldb, ldb, pU+8*sdu, n-ii);
	for(jj=0; jj<m; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(jj, pU, sdu, sB.pA+jj*sdb, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, sB.pA+jj*ps+jj*sdb, dB+jj, n-ii, m-jj);
		}
	kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
	kernel_dunpack_nt_4_lib4(m, pU+4*sdu, B+(ii+4)*ldb, ldb);
	kernel_dunpack_nt_4_vs_lib4(m, pU+8*sdu, B+(ii+8)*ldb, ldb, n-ii);
goto llnn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
llnn_1_left_8:
	kernel_dpack_tn_4_lib4(m, B+ii*ldb, ldb, pU);
	kernel_dpack_tn_4_vs_lib4(m, B+(ii+4)*ldb, ldb, pU+4*sdu, n-ii);
	for(jj=0; jj<m; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(jj, pU, sdu, sB.pA+jj*sdb, alpha, pU+jj*ps, sdu, pU+jj*ps, sdu, sB.pA+jj*ps+jj*sdb, dB+jj, n-ii, m-jj);
		}
	kernel_dunpack_nt_4_lib4(m, pU, B+ii*ldb, ldb);
	kernel_dunpack_nt_4_vs_lib4(m, pU+4*sdu, B+(ii+4)*ldb, ldb, n-ii);
goto llnn_1_return;
#endif

llnn_1_left_4:
	kernel_dpack_tn_4_vs_lib4(m, B+ii*ldb, ldb, pU, n-ii);
	for(jj=0; jj<m; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib4(jj, pU, sB.pA+jj*sdb, alpha, pU+jj*ps, pU+jj*ps, sB.pA+jj*ps+jj*sdb, dB+jj, n-ii, m-jj);
		}
	kernel_dunpack_nt_4_vs_lib4(m, pU, B+ii*ldb, ldb, n-ii);
goto llnn_1_return;

llnn_1_return:
	free(mem);
	return;



/***********************
* rltn
***********************/
rltn:
#if defined(TARGET_X64_INTEL_HASWELL)
	if(m>120 | n>120 | m>K_MAX_STACK) // XXX cond on m !!!!!
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	if(m>=64 | n>=64 | m>K_MAX_STACK) // XXX cond on m !!!!!
#else
	if(m>=12 | n>=12 | m>K_MAX_STACK) // XXX cond on m !!!!!
#endif
		{
		goto rltn_1;
		}
	else
		{
		goto rltn_0;
		}

rltn_0:
	pU = pU0;
	sdu = sdu0;
	dA = pd0;

	for(ii=0; ii<n; ii++)
		dA[ii] = 1.0/A[ii+ii*lda];

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-11; ii+=12)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj);
			kernel_dpack_nn_12_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_0_left_4;
			}
		if(m-ii<=8)
			{
			goto rltn_0_left_8;
			}
		else
			{
			goto rltn_0_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(ii=0; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj);
			kernel_dpack_nn_8_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_0_left_4;
			}
		else
			{
			goto rltn_0_left_8;
			}
		}
#else
	for(ii=0; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib4ccc(jj, pU, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj);
			kernel_dpack_nn_4_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(jj, pU, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		goto rltn_0_left_4;
		}
#endif
	goto rltn_0_return;

#if defined(TARGET_X64_INTEL_HASWELL)
rltn_0_left_12:
		for(jj=0; jj<n; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
			kernel_dpack_nn_12_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
			}
	goto rltn_0_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
rltn_0_left_8:
		for(jj=0; jj<n; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib4ccc(jj, pU, sdu, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
			kernel_dpack_nn_8_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
			}
	goto rltn_0_return;
#endif

rltn_0_left_4:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib4ccc(jj, pU, A+jj, lda, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, A+jj+jj*lda, lda, dA+jj, m-ii, n-jj);
		kernel_dpack_nn_4_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, m-ii);
		}
goto rltn_0_return;

rltn_0_return:
	return;



rltn_1:
	n1 = (n+128-1)/128*128;
	sA_size = blasfeo_memsize_dmat(12, n1);
	sB_size = blasfeo_memsize_dmat(n1, n1);
	mem = malloc(sA_size+sB_size+64);
	blasfeo_align_64_byte(mem, &mem_align);
	blasfeo_create_dmat(12, n, &sA, mem_align);
	blasfeo_create_dmat(n, n, &sB, mem_align+sA_size);

	// TODO pack triangle
//	blasfeo_pack_dmat(n, n, A, lda, &sB, 0, 0);
	blasfeo_pack_l_dmat(n, n, A, lda, &sB, 0, 0);
//	blasfeo_print_dmat(n, n, &sB, 0, 0);

	pU = sA.pA;
	sdu = sA.cn;
	pB = sB.pA;
	sdb = sB.cn;
	dB = sB.dA;

	for(ii=0; ii<n; ii++)
		dB[ii] = 1.0/A[ii+ii*lda];

	ii = 0;
#if defined(TARGET_X64_INTEL_HASWELL)
	for(ii=0; ii<m-11; ii+=12)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_12x4_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj);
			kernel_dpack_nn_12_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_12x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_1_left_4;
			}
		if(m-ii<=8)
			{
			goto rltn_1_left_8;
			}
		else
			{
			goto rltn_1_left_12;
			}
		}
#elif defined(TARGET_X64_INTEL_SANDY_BRIDGE)
	for(ii=0; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_8x4_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj);
			kernel_dpack_nn_8_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps, sdu);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_8x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		if(m-ii<=4)
			{
			goto rltn_1_left_4;
			}
		else
			{
			goto rltn_1_left_8;
			}
		}
#else
	for(ii=0; ii<m-3; ii+=4)
		{
		for(jj=0; jj<n-3; jj+=4)
			{
			kernel_dtrsm_nt_rl_inv_4x4_lib44c4(jj, pU, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj);
			kernel_dpack_nn_4_lib4(4, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		if(jj<n)
			{
			kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4(jj, pU, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
//			kernel_dpack_nn_4_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps);
			}
		}
	if(ii<m)
		{
		goto rltn_1_left_4;
		}
#endif
	goto rltn_1_return;

#if defined(TARGET_X64_INTEL_HASWELL)
rltn_1_left_12:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_12x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
		kernel_dpack_nn_12_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
		}
goto rltn_1_return;
#endif

#if defined(TARGET_X64_INTEL_HASWELL) | defined(TARGET_X64_INTEL_SANDY_BRIDGE)
rltn_1_left_8:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_8x4_vs_lib44c4(jj, pU, sdu, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
		kernel_dpack_nn_8_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, sdu, m-ii);
		}
goto rltn_1_return;
#endif

rltn_1_left_4:
	for(jj=0; jj<n; jj+=4)
		{
		kernel_dtrsm_nt_rl_inv_4x4_vs_lib44c4(jj, pU, pB+jj*sdb, alpha, B+ii+jj*ldb, ldb, B+ii+jj*ldb, ldb, pB+jj*ps+jj*sdb, dB+jj, m-ii, n-jj);
		kernel_dpack_nn_4_vs_lib4(n-jj, B+ii+jj*ldb, ldb, pU+jj*ps, m-ii);
		}
goto rltn_1_return;

rltn_1_return:
	free(mem);
	return;

	}
