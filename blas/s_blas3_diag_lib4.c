/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2017 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                          gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_kernel.h"



#if defined(LA_HIGH_PERFORMANCE)



// dgemm with A diagonal matrix (stored as strvec)
void sgemm_l_diag_libstr(int m, int n, float alpha, struct blasfeo_svec *sA, int ai, struct blasfeo_smat *sB, int bi, int bj, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0 | n<=0)
		return;

	if(bi!=0 | ci!=0 | di!=0)
		{
		printf("\nsgemm_l_diag_libstr: feature not implemented yet: bi=%d, ci=%d, di=%d\n", bi, ci, di);
		exit(1);
		}

	const int bs = 4;

	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *dA = sA->pa + ai;
	float *pB = sB->pA + bj*bs;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

//	sgemm_diag_left_lib(m, n, alpha, dA, pB, sdb, beta, pC, sdc, pD, sdd);
	int ii;

	ii = 0;
	if(beta==0.0)
		{
		for( ; ii<m-3; ii+=4)
			{
			kernel_sgemm_diag_left_4_a0_lib4(n, &alpha, &dA[ii], &pB[ii*sdb], &pD[ii*sdd]);
			}
		}
	else
		{
		for( ; ii<m-3; ii+=4)
			{
			kernel_sgemm_diag_left_4_lib4(n, &alpha, &dA[ii], &pB[ii*sdb], &beta, &pC[ii*sdc], &pD[ii*sdd]);
			}
		}
	if(m-ii>0)
		{
		if(m-ii==1)
			kernel_sgemm_diag_left_1_lib4(n, &alpha, &dA[ii], &pB[ii*sdb], &beta, &pC[ii*sdc], &pD[ii*sdd]);
		else if(m-ii==2)
			kernel_sgemm_diag_left_2_lib4(n, &alpha, &dA[ii], &pB[ii*sdb], &beta, &pC[ii*sdc], &pD[ii*sdd]);
		else // if(m-ii==3)
			kernel_sgemm_diag_left_3_lib4(n, &alpha, &dA[ii], &pB[ii*sdb], &beta, &pC[ii*sdc], &pD[ii*sdd]);
		}
	
	return;

	}



// dgemm with B diagonal matrix (stored as strvec)
void sgemm_r_diag_libstr(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sB, int bi, float beta, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0 | n<=0)
		return;

	if(ai!=0 | ci!=0 | di!=0)
		{
		printf("\nsgemm_r_diag_libstr: feature not implemented yet: ai=%d, ci=%d, di=%d\n", ai, ci, di);
		exit(1);
		}

	const int bs = 4;

	int sda = sA->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*bs;
	float *dB = sB->pa + bi;
	float *pC = sC->pA + cj*bs;
	float *pD = sD->pA + dj*bs;

	int ii;

	ii = 0;
	if(beta==0.0)
		{
		for( ; ii<n-3; ii+=4)
			{
			kernel_sgemm_diag_right_4_a0_lib4(m, &alpha, &pA[ii*bs], sda, &dB[ii], &pD[ii*bs], sdd);
			}
		}
	else
		{
		for( ; ii<n-3; ii+=4)
			{
			kernel_sgemm_diag_right_4_lib4(m, &alpha, &pA[ii*bs], sda, &dB[ii], &beta, &pC[ii*bs], sdc, &pD[ii*bs], sdd);
			}
		}
	if(n-ii>0)
		{
		if(n-ii==1)
			kernel_sgemm_diag_right_1_lib4(m, &alpha, &pA[ii*bs], sda, &dB[ii], &beta, &pC[ii*bs], sdc, &pD[ii*bs], sdd);
		else if(n-ii==2)
			kernel_sgemm_diag_right_2_lib4(m, &alpha, &pA[ii*bs], sda, &dB[ii], &beta, &pC[ii*bs], sdc, &pD[ii*bs], sdd);
		else // if(n-ii==3)
			kernel_sgemm_diag_right_3_lib4(m, &alpha, &pA[ii*bs], sda, &dB[ii], &beta, &pC[ii*bs], sdc, &pD[ii*bs], sdd);
		}
		return;

	}



#else

#error : wrong LA choice

#endif




