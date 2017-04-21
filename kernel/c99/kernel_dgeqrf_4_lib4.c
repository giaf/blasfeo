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

#include <math.h>
#include <stdio.h>

#include "../../include/blasfeo_common.h"
#include "../../include/blasfeo_d_aux.h"



void kernel_dgeqrf_4_lib4(int m, double *pD, int sdd, double *dD)
	{
	int ii, jj, ll;
	double alpha, beta, tmp, w1, w2, w3;
	const int ps = 4;
	// first column
	beta = 0.0;
	tmp = pD[1+ps*0];
	beta += tmp*tmp;
	tmp = pD[2+ps*0];
	beta += tmp*tmp;
	tmp = pD[3+ps*0];
	beta += tmp*tmp;
	for(ii=4; ii<m-3; ii+=4)
		{
		tmp = pD[0+ii*sdd+ps*0];
		beta += tmp*tmp;
		tmp = pD[1+ii*sdd+ps*0];
		beta += tmp*tmp;
		tmp = pD[2+ii*sdd+ps*0];
		beta += tmp*tmp;
		tmp = pD[3+ii*sdd+ps*0];
		beta += tmp*tmp;
		}
	for(ll=0; ll<m-ii; ll++)
		{
		tmp = pD[ll+ii*sdd+ps*0];
		beta += tmp*tmp;
		}
	if(beta==0.0)
		{
		// tau
		dD[0] = 0.0;
		}
	else
		{
		alpha = pD[0+ps*0];
		beta += alpha*alpha;
		beta = sqrt(beta);
		if(alpha>0)
			beta = -beta;
		// tau0
		dD[0] = (beta-alpha) / beta;
		tmp = 1.0 / (alpha-beta);
		// compute v0
		pD[0+ps*0] = beta;
		pD[1+ps*0] *= tmp;
		pD[2+ps*0] *= tmp;
		pD[3+ps*0] *= tmp;
		for(ii=4; ii<m-3; ii+=4)
			{
			pD[0+ii*sdd+ps*0] *= tmp;
			pD[1+ii*sdd+ps*0] *= tmp;
			pD[2+ii*sdd+ps*0] *= tmp;
			pD[3+ii*sdd+ps*0] *= tmp;
			}
		for(ll=0; ll<m-ii; ll++)
			{
			pD[ll+ii*sdd+ps*0] *= tmp;
			}
		}
	// gemv_t & ger
	w1 = pD[0+ps*1];
	w2 = pD[0+ps*2];
	w3 = pD[0+ps*3];
	w1 += pD[1+ps*1] * pD[1+ps*0];
	w2 += pD[1+ps*2] * pD[1+ps*0];
	w3 += pD[1+ps*3] * pD[1+ps*0];
	w1 += pD[2+ps*1] * pD[2+ps*0];
	w2 += pD[2+ps*2] * pD[2+ps*0];
	w3 += pD[2+ps*3] * pD[2+ps*0];
	w1 += pD[3+ps*1] * pD[3+ps*0];
	w2 += pD[3+ps*2] * pD[3+ps*0];
	w3 += pD[3+ps*3] * pD[3+ps*0];
	for(ii=4; ii<m-3; ii+=4)
		{
		w1 += pD[0+ii*sdd+ps*1] * pD[0+ii*sdd+ps*0];
		w2 += pD[0+ii*sdd+ps*2] * pD[0+ii*sdd+ps*0];
		w3 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*0];
		w1 += pD[1+ii*sdd+ps*1] * pD[1+ii*sdd+ps*0];
		w2 += pD[1+ii*sdd+ps*2] * pD[1+ii*sdd+ps*0];
		w3 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*0];
		w1 += pD[2+ii*sdd+ps*1] * pD[2+ii*sdd+ps*0];
		w2 += pD[2+ii*sdd+ps*2] * pD[2+ii*sdd+ps*0];
		w3 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*0];
		w1 += pD[3+ii*sdd+ps*1] * pD[3+ii*sdd+ps*0];
		w2 += pD[3+ii*sdd+ps*2] * pD[3+ii*sdd+ps*0];
		w3 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*0];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		w1 += pD[ll+ii*sdd+ps*1] * pD[ll+ii*sdd+ps*0];
		w2 += pD[ll+ii*sdd+ps*2] * pD[ll+ii*sdd+ps*0];
		w3 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*0];
		}
	w1 = - dD[0] * w1;
	w2 = - dD[0] * w2;
	w3 = - dD[0] * w3;
	pD[0+ps*1] += w1;
	pD[0+ps*2] += w2;
	pD[0+ps*3] += w3;
	pD[1+ps*1] += w1 * pD[1+ps*0];
	pD[1+ps*2] += w2 * pD[1+ps*0];
	pD[1+ps*3] += w3 * pD[1+ps*0];
	pD[2+ps*1] += w1 * pD[2+ps*0];
	pD[2+ps*2] += w2 * pD[2+ps*0];
	pD[2+ps*3] += w3 * pD[2+ps*0];
	pD[3+ps*1] += w1 * pD[3+ps*0];
	pD[3+ps*2] += w2 * pD[3+ps*0];
	pD[3+ps*3] += w3 * pD[3+ps*0];
	for(ii=4; ii<m-3; ii+=4)
		{
		pD[0+ii*sdd+ps*1] += w1 * pD[0+ii*sdd+ps*0];
		pD[0+ii*sdd+ps*2] += w2 * pD[0+ii*sdd+ps*0];
		pD[0+ii*sdd+ps*3] += w3 * pD[0+ii*sdd+ps*0];
		pD[1+ii*sdd+ps*1] += w1 * pD[1+ii*sdd+ps*0];
		pD[1+ii*sdd+ps*2] += w2 * pD[1+ii*sdd+ps*0];
		pD[1+ii*sdd+ps*3] += w3 * pD[1+ii*sdd+ps*0];
		pD[2+ii*sdd+ps*1] += w1 * pD[2+ii*sdd+ps*0];
		pD[2+ii*sdd+ps*2] += w2 * pD[2+ii*sdd+ps*0];
		pD[2+ii*sdd+ps*3] += w3 * pD[2+ii*sdd+ps*0];
		pD[3+ii*sdd+ps*1] += w1 * pD[3+ii*sdd+ps*0];
		pD[3+ii*sdd+ps*2] += w2 * pD[3+ii*sdd+ps*0];
		pD[3+ii*sdd+ps*3] += w3 * pD[3+ii*sdd+ps*0];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		pD[ll+ii*sdd+ps*1] += w1 * pD[ll+ii*sdd+ps*0];
		pD[ll+ii*sdd+ps*2] += w2 * pD[ll+ii*sdd+ps*0];
		pD[ll+ii*sdd+ps*3] += w3 * pD[ll+ii*sdd+ps*0];
		}
	// second column
	beta = 0.0;
	tmp = pD[2+ps*1];
	beta += tmp*tmp;
	tmp = pD[3+ps*1];
	beta += tmp*tmp;
	for(ii=4; ii<m-3; ii+=4)
		{
		tmp = pD[0+ii*sdd+ps*1];
		beta += tmp*tmp;
		tmp = pD[1+ii*sdd+ps*1];
		beta += tmp*tmp;
		tmp = pD[2+ii*sdd+ps*1];
		beta += tmp*tmp;
		tmp = pD[3+ii*sdd+ps*1];
		beta += tmp*tmp;
		}
	for(ll=0; ll<m-ii; ll++)
		{
		tmp = pD[ll+ii*sdd+ps*1];
		beta += tmp*tmp;
		}
	if(beta==0.0)
		{
		// tau
		dD[1] = 0.0;
		}
	else
		{
		alpha = pD[1+ps*1];
		beta += alpha*alpha;
		beta = sqrt(beta);
		if(alpha>0)
			beta = -beta;
		// tau0
		dD[1] = (beta-alpha) / beta;
		tmp = 1.0 / (alpha-beta);
		// compute v0
		pD[1+ps*1] = beta;
		pD[2+ps*1] *= tmp;
		pD[3+ps*1] *= tmp;
		for(ii=4; ii<m-3; ii+=4)
			{
			pD[0+ii*sdd+ps*1] *= tmp;
			pD[1+ii*sdd+ps*1] *= tmp;
			pD[2+ii*sdd+ps*1] *= tmp;
			pD[3+ii*sdd+ps*1] *= tmp;
			}
		for(ll=0; ll<m-ii; ll++)
			{
			pD[ll+ii*sdd+ps*1] *= tmp;
			}
		}
	// gemv_t & ger
	w2 = pD[1+ps*2];
	w3 = pD[1+ps*3];
	w2 += pD[2+ps*2] * pD[2+ps*1];
	w3 += pD[2+ps*3] * pD[2+ps*1];
	w2 += pD[3+ps*2] * pD[3+ps*1];
	w3 += pD[3+ps*3] * pD[3+ps*1];
	for(ii=4; ii<m-3; ii+=4)
		{
		w2 += pD[0+ii*sdd+ps*2] * pD[0+ii*sdd+ps*1];
		w3 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*1];
		w2 += pD[1+ii*sdd+ps*2] * pD[1+ii*sdd+ps*1];
		w3 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*1];
		w2 += pD[2+ii*sdd+ps*2] * pD[2+ii*sdd+ps*1];
		w3 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*1];
		w2 += pD[3+ii*sdd+ps*2] * pD[3+ii*sdd+ps*1];
		w3 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*1];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		w2 += pD[ll+ii*sdd+ps*2] * pD[ll+ii*sdd+ps*1];
		w3 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*1];
		}
	w2 = - dD[1] * w2;
	w3 = - dD[1] * w3;
	pD[1+ps*2] += w2;
	pD[1+ps*3] += w3;
	pD[2+ps*2] += w2 * pD[2+ps*1];
	pD[2+ps*3] += w3 * pD[2+ps*1];
	pD[3+ps*2] += w2 * pD[3+ps*1];
	pD[3+ps*3] += w3 * pD[3+ps*1];
	for(ii=4; ii<m-3; ii+=4)
		{
		pD[0+ii*sdd+ps*2] += w2 * pD[0+ii*sdd+ps*1];
		pD[0+ii*sdd+ps*3] += w3 * pD[0+ii*sdd+ps*1];
		pD[1+ii*sdd+ps*2] += w2 * pD[1+ii*sdd+ps*1];
		pD[1+ii*sdd+ps*3] += w3 * pD[1+ii*sdd+ps*1];
		pD[2+ii*sdd+ps*2] += w2 * pD[2+ii*sdd+ps*1];
		pD[2+ii*sdd+ps*3] += w3 * pD[2+ii*sdd+ps*1];
		pD[3+ii*sdd+ps*2] += w2 * pD[3+ii*sdd+ps*1];
		pD[3+ii*sdd+ps*3] += w3 * pD[3+ii*sdd+ps*1];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		pD[ll+ii*sdd+ps*2] += w2 * pD[ll+ii*sdd+ps*1];
		pD[ll+ii*sdd+ps*3] += w3 * pD[ll+ii*sdd+ps*1];
		}
	// third column
	beta = 0.0;
	tmp = pD[3+ps*2];
	beta += tmp*tmp;
	for(ii=4; ii<m-3; ii+=4)
		{
		tmp = pD[0+ii*sdd+ps*2];
		beta += tmp*tmp;
		tmp = pD[1+ii*sdd+ps*2];
		beta += tmp*tmp;
		tmp = pD[2+ii*sdd+ps*2];
		beta += tmp*tmp;
		tmp = pD[3+ii*sdd+ps*2];
		beta += tmp*tmp;
		}
	for(ll=0; ll<m-ii; ll++)
		{
		tmp = pD[ll+ii*sdd+ps*2];
		beta += tmp*tmp;
		}
	if(beta==0.0)
		{
		// tau
		dD[2] = 0.0;
		}
	else
		{
		alpha = pD[2+ps*2];
		beta += alpha*alpha;
		beta = sqrt(beta);
		if(alpha>0)
			beta = -beta;
		// tau0
		dD[2] = (beta-alpha) / beta;
		tmp = 1.0 / (alpha-beta);
		// compute v0
		pD[2+ps*2] = beta;
		pD[3+ps*2] *= tmp;
		for(ii=4; ii<m-3; ii+=4)
			{
			pD[0+ii*sdd+ps*2] *= tmp;
			pD[1+ii*sdd+ps*2] *= tmp;
			pD[2+ii*sdd+ps*2] *= tmp;
			pD[3+ii*sdd+ps*2] *= tmp;
			}
		for(ll=0; ll<m-ii; ll++)
			{
			pD[ll+ii*sdd+ps*2] *= tmp;
			}
		}
	// gemv_t & ger
	w3 = pD[2+ps*3];
	w3 += pD[3+ps*3] * pD[3+ps*2];
	for(ii=4; ii<m-3; ii+=4)
		{
		w3 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*2];
		w3 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*2];
		w3 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*2];
		w3 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*2];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		w3 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*2];
		}
	w3 = - dD[2] * w3;
	pD[2+ps*3] += w3;
	pD[3+ps*3] += w3 * pD[3+ps*2];
	for(ii=4; ii<m-3; ii+=4)
		{
		pD[0+ii*sdd+ps*3] += w3 * pD[0+ii*sdd+ps*2];
		pD[1+ii*sdd+ps*3] += w3 * pD[1+ii*sdd+ps*2];
		pD[2+ii*sdd+ps*3] += w3 * pD[2+ii*sdd+ps*2];
		pD[3+ii*sdd+ps*3] += w3 * pD[3+ii*sdd+ps*2];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		pD[ll+ii*sdd+ps*3] += w3 * pD[ll+ii*sdd+ps*2];
		}
	// fourth column
	beta = 0.0;
	for(ii=4; ii<m-3; ii+=4)
		{
		tmp = pD[0+ii*sdd+ps*3];
		beta += tmp*tmp;
		tmp = pD[1+ii*sdd+ps*3];
		beta += tmp*tmp;
		tmp = pD[2+ii*sdd+ps*3];
		beta += tmp*tmp;
		tmp = pD[3+ii*sdd+ps*3];
		beta += tmp*tmp;
		}
	for(ll=0; ll<m-ii; ll++)
		{
		tmp = pD[ll+ii*sdd+ps*3];
		beta += tmp*tmp;
		}
	if(beta==0.0)
		{
		// tau
		dD[3] = 0.0;
		}
	else
		{
		alpha = pD[3+ps*3];
		beta += alpha*alpha;
		beta = sqrt(beta);
		if(alpha>0)
			beta = -beta;
		// tau0
		dD[3] = (beta-alpha) / beta;
		tmp = 1.0 / (alpha-beta);
		// compute v0
		pD[3+ps*3] = beta;
		for(ii=4; ii<m-3; ii+=4)
			{
			pD[0+ii*sdd+ps*3] *= tmp;
			pD[1+ii*sdd+ps*3] *= tmp;
			pD[2+ii*sdd+ps*3] *= tmp;
			pD[3+ii*sdd+ps*3] *= tmp;
			}
		for(ll=0; ll<m-ii; ll++)
			{
			pD[ll+ii*sdd+ps*3] *= tmp;
			}
		}
	return;
	}


void kernel_dlarf_4_lib4(int m, int n, double *pD, int sdd, double *dD, double *pC0, int sdc)
	{
	if(m<=0 | n<=0)
		return;
	int ii, jj, ll;
	const int ps = 4;
	double v10,
	       v20, v21,
		   v30, v31, v32;
	double tmp, d0, d1, d2, d3;
	double *pC;
	double pT[16];// = {};
	int ldt = 4;
	double pW[8];// = {};
	int ldw = 2;
	// dot product of v
	// ii=0
	// ii=1
	v10 = 1.0 * pD[1+ps*0];
	// ii=2
	v10 += pD[2+ps*1] * pD[2+ps*0];
	v20 = 1.0 * pD[2+ps*0];
	v21 = 1.0 * pD[2+ps*1];
	// ii=3
	v10 += pD[3+ps*1] * pD[3+ps*0];
	v20 += pD[3+ps*2] * pD[3+ps*0];
	v21 += pD[3+ps*2] * pD[3+ps*1];
	v30 = 1.0 * pD[3+ps*0];
	v31 = 1.0 * pD[3+ps*1];
	v32 = 1.0 * pD[3+ps*2];
	for(ii=4; ii<m-3; ii+=4)
		{
		v10 += pD[0+ii*sdd+ps*1] * pD[0+ii*sdd+ps*0];
		v20 += pD[0+ii*sdd+ps*2] * pD[0+ii*sdd+ps*0];
		v21 += pD[0+ii*sdd+ps*2] * pD[0+ii*sdd+ps*1];
		v30 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*0];
		v31 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*1];
		v32 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*2];
		v10 += pD[1+ii*sdd+ps*1] * pD[1+ii*sdd+ps*0];
		v20 += pD[1+ii*sdd+ps*2] * pD[1+ii*sdd+ps*0];
		v21 += pD[1+ii*sdd+ps*2] * pD[1+ii*sdd+ps*1];
		v30 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*0];
		v31 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*1];
		v32 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*2];
		v10 += pD[2+ii*sdd+ps*1] * pD[2+ii*sdd+ps*0];
		v20 += pD[2+ii*sdd+ps*2] * pD[2+ii*sdd+ps*0];
		v21 += pD[2+ii*sdd+ps*2] * pD[2+ii*sdd+ps*1];
		v30 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*0];
		v31 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*1];
		v32 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*2];
		v10 += pD[3+ii*sdd+ps*1] * pD[3+ii*sdd+ps*0];
		v20 += pD[3+ii*sdd+ps*2] * pD[3+ii*sdd+ps*0];
		v21 += pD[3+ii*sdd+ps*2] * pD[3+ii*sdd+ps*1];
		v30 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*0];
		v31 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*1];
		v32 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*2];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		v10 += pD[ll+ii*sdd+ps*1] * pD[ll+ii*sdd+ps*0];
		v20 += pD[ll+ii*sdd+ps*2] * pD[ll+ii*sdd+ps*0];
		v21 += pD[ll+ii*sdd+ps*2] * pD[ll+ii*sdd+ps*1];
		v30 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*0];
		v31 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*1];
		v32 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*2];
		}
	// compute lower triangular T containing tau for matrix update
	pT[0+ldt*0] = dD[0];
	pT[1+ldt*1] = dD[1];
	pT[2+ldt*2] = dD[2];
	pT[3+ldt*3] = dD[3];
	pT[1+ldt*0] = - dD[1] * (v10*pT[0+ldt*0]);
	pT[2+ldt*1] = - dD[2] * (v21*pT[1+ldt*1]);
	pT[3+ldt*2] = - dD[3] * (v32*pT[2+ldt*2]);
	pT[2+ldt*0] = - dD[2] * (v20*pT[0+ldt*0] + v21*pT[1+ldt*0]);
	pT[3+ldt*1] = - dD[3] * (v31*pT[1+ldt*1] + v32*pT[2+ldt*1]);
	pT[3+ldt*0] = - dD[3] * (v30*pT[0+ldt*0] + v31*pT[1+ldt*0] + v32*pT[2+ldt*0]);
	// downgrade matrix
	ii = 0;
	for( ; ii<n-1; ii+=2)
		{
		pC = pC0+ii*ps;
		// compute W^T = C^T * V
		// jj=0
		tmp = pC[0+ps*0];
		pW[0+ldw*0] = tmp;
		tmp = pC[0+ps*1];
		pW[1+ldw*0] = tmp;
		// jj=1
		d0 = pD[1+ps*0];
		tmp = pC[1+ps*0];
		pW[0+ldw*0] += tmp * d0;
		pW[0+ldw*1] = tmp;
		tmp = pC[1+ps*1];
		pW[1+ldw*0] += tmp * d0;
		pW[1+ldw*1] = tmp;
		// jj=2
		d0 = pD[2+ps*0];
		d1 = pD[2+ps*1];
		tmp = pC[2+ps*0];
		pW[0+ldw*0] += tmp * d0;
		pW[0+ldw*1] += tmp * d1;
		pW[0+ldw*2] = tmp;
		tmp = pC[2+ps*1];
		pW[1+ldw*0] += tmp * d0;
		pW[1+ldw*1] += tmp * d1;
		pW[1+ldw*2] = tmp;
		// jj=3
		d0 = pD[3+ps*0];
		d1 = pD[3+ps*1];
		d2 = pD[3+ps*2];
		tmp = pC[3+ps*0];
		pW[0+ldw*0] += tmp * d0;
		pW[0+ldw*1] += tmp * d1;
		pW[0+ldw*2] += tmp * d2;
		pW[0+ldw*3] = tmp;
		tmp = pC[3+ps*1];
		pW[1+ldw*0] += tmp * d0;
		pW[1+ldw*1] += tmp * d1;
		pW[1+ldw*2] += tmp * d2;
		pW[1+ldw*3] = tmp;
		for(jj=4; jj<m-3; jj+=4)
			{
			//
			d0 = pD[0+jj*sdd+ps*0];
			d1 = pD[0+jj*sdd+ps*1];
			d2 = pD[0+jj*sdd+ps*2];
			d3 = pD[0+jj*sdd+ps*3];
			tmp = pC[0+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * d0;
			pW[0+ldw*1] += tmp * d1;
			pW[0+ldw*2] += tmp * d2;
			pW[0+ldw*3] += tmp * d3;
			tmp = pC[0+jj*sdc+ps*1];
			pW[1+ldw*0] += tmp * d0;
			pW[1+ldw*1] += tmp * d1;
			pW[1+ldw*2] += tmp * d2;
			pW[1+ldw*3] += tmp * d3;
			//
			d0 = pD[1+jj*sdd+ps*0];
			d1 = pD[1+jj*sdd+ps*1];
			d2 = pD[1+jj*sdd+ps*2];
			d3 = pD[1+jj*sdd+ps*3];
			tmp = pC[1+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * d0;
			pW[0+ldw*1] += tmp * d1;
			pW[0+ldw*2] += tmp * d2;
			pW[0+ldw*3] += tmp * d3;
			tmp = pC[1+jj*sdc+ps*1];
			pW[1+ldw*0] += tmp * d0;
			pW[1+ldw*1] += tmp * d1;
			pW[1+ldw*2] += tmp * d2;
			pW[1+ldw*3] += tmp * d3;
			//
			d0 = pD[2+jj*sdd+ps*0];
			d1 = pD[2+jj*sdd+ps*1];
			d2 = pD[2+jj*sdd+ps*2];
			d3 = pD[2+jj*sdd+ps*3];
			tmp = pC[2+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * d0;
			pW[0+ldw*1] += tmp * d1;
			pW[0+ldw*2] += tmp * d2;
			pW[0+ldw*3] += tmp * d3;
			tmp = pC[2+jj*sdc+ps*1];
			pW[1+ldw*0] += tmp * d0;
			pW[1+ldw*1] += tmp * d1;
			pW[1+ldw*2] += tmp * d2;
			pW[1+ldw*3] += tmp * d3;
			//
			d0 = pD[3+jj*sdd+ps*0];
			d1 = pD[3+jj*sdd+ps*1];
			d2 = pD[3+jj*sdd+ps*2];
			d3 = pD[3+jj*sdd+ps*3];
			tmp = pC[3+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * d0;
			pW[0+ldw*1] += tmp * d1;
			pW[0+ldw*2] += tmp * d2;
			pW[0+ldw*3] += tmp * d3;
			tmp = pC[3+jj*sdc+ps*1];
			pW[1+ldw*0] += tmp * d0;
			pW[1+ldw*1] += tmp * d1;
			pW[1+ldw*2] += tmp * d2;
			pW[1+ldw*3] += tmp * d3;
			}
		for(ll=0; ll<m-jj; ll++)
			{
			d0 = pD[ll+jj*sdd+ps*0];
			d1 = pD[ll+jj*sdd+ps*1];
			d2 = pD[ll+jj*sdd+ps*2];
			d3 = pD[ll+jj*sdd+ps*3];
			tmp = pC[ll+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * d0;
			pW[0+ldw*1] += tmp * d1;
			pW[0+ldw*2] += tmp * d2;
			pW[0+ldw*3] += tmp * d3;
			tmp = pC[ll+jj*sdc+ps*1];
			pW[1+ldw*0] += tmp * d0;
			pW[1+ldw*1] += tmp * d1;
			pW[1+ldw*2] += tmp * d2;
			pW[1+ldw*3] += tmp * d3;
			}
		// compute W^T *= T
		pW[0+ldw*3] = pT[3+ldt*0]*pW[0+ldw*0] + pT[3+ldt*1]*pW[0+ldw*1] + pT[3+ldt*2]*pW[0+ldw*2] + pT[3+ldt*3]*pW[0+ldw*3];
		pW[1+ldw*3] = pT[3+ldt*0]*pW[1+ldw*0] + pT[3+ldt*1]*pW[1+ldw*1] + pT[3+ldt*2]*pW[1+ldw*2] + pT[3+ldt*3]*pW[1+ldw*3];
		pW[0+ldw*2] = pT[2+ldt*0]*pW[0+ldw*0] + pT[2+ldt*1]*pW[0+ldw*1] + pT[2+ldt*2]*pW[0+ldw*2];
		pW[1+ldw*2] = pT[2+ldt*0]*pW[1+ldw*0] + pT[2+ldt*1]*pW[1+ldw*1] + pT[2+ldt*2]*pW[1+ldw*2];
		pW[0+ldw*1] = pT[1+ldt*0]*pW[0+ldw*0] + pT[1+ldt*1]*pW[0+ldw*1];
		pW[1+ldw*1] = pT[1+ldt*0]*pW[1+ldw*0] + pT[1+ldt*1]*pW[1+ldw*1];
		pW[0+ldw*0] = pT[0+ldt*0]*pW[0+ldw*0];
		pW[1+ldw*0] = pT[0+ldt*0]*pW[1+ldw*0];
		// compute C -= V * W^T
		// jj=0
		pC[0+ps*0] -= pW[0+ldw*0];
		pC[0+ps*1] -= pW[1+ldw*0];
		// jj=1
		pC[1+ps*0] -= pD[1+ps*0]*pW[0+ldw*0] + pW[0+ldw*1];
		pC[1+ps*1] -= pD[1+ps*0]*pW[1+ldw*0] + pW[1+ldw*1];
		// jj=2
		pC[2+ps*0] -= pD[2+ps*0]*pW[0+ldw*0] + pD[2+ps*1]*pW[0+ldw*1] + pW[0+ldw*2];
		pC[2+ps*1] -= pD[2+ps*0]*pW[1+ldw*0] + pD[2+ps*1]*pW[1+ldw*1] + pW[1+ldw*2];
		// jj=3
		pC[3+ps*0] -= pD[3+ps*0]*pW[0+ldw*0] + pD[3+ps*1]*pW[0+ldw*1] + pD[3+ps*2]*pW[0+ldw*2] + pW[0+ldw*3];
		pC[3+ps*1] -= pD[3+ps*0]*pW[1+ldw*0] + pD[3+ps*1]*pW[1+ldw*1] + pD[3+ps*2]*pW[1+ldw*2] + pW[1+ldw*3];
		for(jj=4; jj<m-3; jj+=4)
			{
			//
			d0 = pD[0+jj*sdd+ps*0];
			d1 = pD[0+jj*sdd+ps*1];
			d2 = pD[0+jj*sdd+ps*2];
			d3 = pD[0+jj*sdd+ps*3];
			pC[0+jj*sdc+ps*0] -= d0*pW[0+ldw*0] + d1*pW[0+ldw*1] + d2*pW[0+ldw*2] + d3*pW[0+ldw*3];
			pC[0+jj*sdc+ps*1] -= d0*pW[1+ldw*0] + d1*pW[1+ldw*1] + d2*pW[1+ldw*2] + d3*pW[1+ldw*3];
			//
			d0 = pD[1+jj*sdd+ps*0];
			d1 = pD[1+jj*sdd+ps*1];
			d2 = pD[1+jj*sdd+ps*2];
			d3 = pD[1+jj*sdd+ps*3];
			pC[1+jj*sdc+ps*0] -= d0*pW[0+ldw*0] + d1*pW[0+ldw*1] + d2*pW[0+ldw*2] + d3*pW[0+ldw*3];
			pC[1+jj*sdc+ps*1] -= d0*pW[1+ldw*0] + d1*pW[1+ldw*1] + d2*pW[1+ldw*2] + d3*pW[1+ldw*3];
			//
			d0 = pD[2+jj*sdd+ps*0];
			d1 = pD[2+jj*sdd+ps*1];
			d2 = pD[2+jj*sdd+ps*2];
			d3 = pD[2+jj*sdd+ps*3];
			pC[2+jj*sdc+ps*0] -= d0*pW[0+ldw*0] + d1*pW[0+ldw*1] + d2*pW[0+ldw*2] + d3*pW[0+ldw*3];
			pC[2+jj*sdc+ps*1] -= d0*pW[1+ldw*0] + d1*pW[1+ldw*1] + d2*pW[1+ldw*2] + d3*pW[1+ldw*3];
			//
			d0 = pD[3+jj*sdd+ps*0];
			d1 = pD[3+jj*sdd+ps*1];
			d2 = pD[3+jj*sdd+ps*2];
			d3 = pD[3+jj*sdd+ps*3];
			pC[3+jj*sdc+ps*0] -= d0*pW[0+ldw*0] + d1*pW[0+ldw*1] + d2*pW[0+ldw*2] + d3*pW[0+ldw*3];
			pC[3+jj*sdc+ps*1] -= d0*pW[1+ldw*0] + d1*pW[1+ldw*1] + d2*pW[1+ldw*2] + d3*pW[1+ldw*3];
			}
		for(ll=0; ll<m-jj; ll++)
			{
			d0 = pD[ll+jj*sdd+ps*0];
			d1 = pD[ll+jj*sdd+ps*1];
			d2 = pD[ll+jj*sdd+ps*2];
			d3 = pD[ll+jj*sdd+ps*3];
			pC[ll+jj*sdc+ps*0] -= d0*pW[0+ldw*0] + d1*pW[0+ldw*1] + d2*pW[0+ldw*2] + d3*pW[0+ldw*3];
			pC[ll+jj*sdc+ps*1] -= d0*pW[1+ldw*0] + d1*pW[1+ldw*1] + d2*pW[1+ldw*2] + d3*pW[1+ldw*3];
			}
		}
	for( ; ii<n; ii++)
		{
		pC = pC0+ii*ps;
		// compute W^T = C^T * V
		// jj=0
		tmp = pC[0+ps*0];
		pW[0+ldw*0] = tmp;
		// jj=1
		tmp = pC[1+ps*0];
		pW[0+ldw*0] += tmp * pD[1+ps*0];
		pW[0+ldw*1] = tmp;
		// jj=2
		tmp = pC[2+ps*0];
		pW[0+ldw*0] += tmp * pD[2+ps*0];
		pW[0+ldw*1] += tmp * pD[2+ps*1];
		pW[0+ldw*2] = tmp;
		// jj=3
		tmp = pC[3+ps*0];
		pW[0+ldw*0] += tmp * pD[3+ps*0];
		pW[0+ldw*1] += tmp * pD[3+ps*1];
		pW[0+ldw*2] += tmp * pD[3+ps*2];
		pW[0+ldw*3] = tmp;
		for(jj=4; jj<m-3; jj+=4)
			{
			tmp = pC[0+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * pD[0+jj*sdd+ps*0];
			pW[0+ldw*1] += tmp * pD[0+jj*sdd+ps*1];
			pW[0+ldw*2] += tmp * pD[0+jj*sdd+ps*2];
			pW[0+ldw*3] += tmp * pD[0+jj*sdd+ps*3];
			tmp = pC[1+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * pD[1+jj*sdd+ps*0];
			pW[0+ldw*1] += tmp * pD[1+jj*sdd+ps*1];
			pW[0+ldw*2] += tmp * pD[1+jj*sdd+ps*2];
			pW[0+ldw*3] += tmp * pD[1+jj*sdd+ps*3];
			tmp = pC[2+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * pD[2+jj*sdd+ps*0];
			pW[0+ldw*1] += tmp * pD[2+jj*sdd+ps*1];
			pW[0+ldw*2] += tmp * pD[2+jj*sdd+ps*2];
			pW[0+ldw*3] += tmp * pD[2+jj*sdd+ps*3];
			tmp = pC[3+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * pD[3+jj*sdd+ps*0];
			pW[0+ldw*1] += tmp * pD[3+jj*sdd+ps*1];
			pW[0+ldw*2] += tmp * pD[3+jj*sdd+ps*2];
			pW[0+ldw*3] += tmp * pD[3+jj*sdd+ps*3];
			}
		for(ll=0; ll<m-jj; ll++)
			{
			tmp = pC[ll+jj*sdc+ps*0];
			pW[0+ldw*0] += tmp * pD[ll+jj*sdd+ps*0];
			pW[0+ldw*1] += tmp * pD[ll+jj*sdd+ps*1];
			pW[0+ldw*2] += tmp * pD[ll+jj*sdd+ps*2];
			pW[0+ldw*3] += tmp * pD[ll+jj*sdd+ps*3];
			}
		// compute W^T *= T
		pW[0+ldw*3] = pT[3+ldt*0]*pW[0+ldw*0] + pT[3+ldt*1]*pW[0+ldw*1] + pT[3+ldt*2]*pW[0+ldw*2] + pT[3+ldt*3]*pW[0+ldw*3];
		pW[0+ldw*2] = pT[2+ldt*0]*pW[0+ldw*0] + pT[2+ldt*1]*pW[0+ldw*1] + pT[2+ldt*2]*pW[0+ldw*2];
		pW[0+ldw*1] = pT[1+ldt*0]*pW[0+ldw*0] + pT[1+ldt*1]*pW[0+ldw*1];
		pW[0+ldw*0] = pT[0+ldt*0]*pW[0+ldw*0];
		// compute C -= V * W^T
		// jj=0
		pC[0+ps*0] -= pW[0+ldw*0];
		// jj=1
		pC[1+ps*0] -= pD[1+ps*0]*pW[0+ldw*0] + pW[0+ldw*1];
		// jj=2
		pC[2+ps*0] -= pD[2+ps*0]*pW[0+ldw*0] + pD[2+ps*1]*pW[0+ldw*1] + pW[0+ldw*2];
		// jj=3
		pC[3+ps*0] -= pD[3+ps*0]*pW[0+ldw*0] + pD[3+ps*1]*pW[0+ldw*1] + pD[3+ps*2]*pW[0+ldw*2] + pW[0+ldw*3];
		for(jj=4; jj<m-3; jj+=4)
			{
			pC[0+jj*sdc+ps*0] -= pD[0+jj*sdd+ps*0]*pW[0+ldw*0] + pD[0+jj*sdd+ps*1]*pW[0+ldw*1] + pD[0+jj*sdd+ps*2]*pW[0+ldw*2] + pD[0+jj*sdd+ps*3]*pW[0+ldw*3];
			pC[1+jj*sdc+ps*0] -= pD[1+jj*sdd+ps*0]*pW[0+ldw*0] + pD[1+jj*sdd+ps*1]*pW[0+ldw*1] + pD[1+jj*sdd+ps*2]*pW[0+ldw*2] + pD[1+jj*sdd+ps*3]*pW[0+ldw*3];
			pC[2+jj*sdc+ps*0] -= pD[2+jj*sdd+ps*0]*pW[0+ldw*0] + pD[2+jj*sdd+ps*1]*pW[0+ldw*1] + pD[2+jj*sdd+ps*2]*pW[0+ldw*2] + pD[2+jj*sdd+ps*3]*pW[0+ldw*3];
			pC[3+jj*sdc+ps*0] -= pD[3+jj*sdd+ps*0]*pW[0+ldw*0] + pD[3+jj*sdd+ps*1]*pW[0+ldw*1] + pD[3+jj*sdd+ps*2]*pW[0+ldw*2] + pD[3+jj*sdd+ps*3]*pW[0+ldw*3];
			}
		for(ll=0; ll<m-jj; ll++)
			{
			pC[ll+jj*sdc+ps*0] -= pD[ll+jj*sdd+ps*0]*pW[0+ldw*0] + pD[ll+jj*sdd+ps*1]*pW[0+ldw*1] + pD[ll+jj*sdd+ps*2]*pW[0+ldw*2] + pD[ll+jj*sdd+ps*3]*pW[0+ldw*3];
			}
		}

	return;
	}



void kernel_dlarf_t_4_lib4(int m, int n, double *pD, int sdd, double *pVt, double *dD, double *pC0, int sdc)
	{
	if(m<=0 | n<=0)
		return;
	int ii, jj, ll;
	const int ps = 4;
	double v10,
	       v20, v21,
		   v30, v31, v32;
	double c00, c01,
	       c10, c11,
	       c20, c21,
	       c30, c31;
	double a0, a1, a2, a3, b0, b1;
	double tmp, d0, d1, d2, d3;
	double *pC;
	double pT[16];// = {};
	int ldt = 4;
	double pW[8];// = {};
	int ldw = 4;
	// dot product of v
	// ii=0
	// ii=1
	v10 = 1.0 * pD[1+ps*0];
	// ii=2
	v10 += pD[2+ps*1] * pD[2+ps*0];
	v20 = 1.0 * pD[2+ps*0];
	v21 = 1.0 * pD[2+ps*1];
	// ii=3
	v10 += pD[3+ps*1] * pD[3+ps*0];
	v20 += pD[3+ps*2] * pD[3+ps*0];
	v21 += pD[3+ps*2] * pD[3+ps*1];
	v30 = 1.0 * pD[3+ps*0];
	v31 = 1.0 * pD[3+ps*1];
	v32 = 1.0 * pD[3+ps*2];
	for(ii=4; ii<m-3; ii+=4)
		{
		v10 += pD[0+ii*sdd+ps*1] * pD[0+ii*sdd+ps*0];
		v20 += pD[0+ii*sdd+ps*2] * pD[0+ii*sdd+ps*0];
		v21 += pD[0+ii*sdd+ps*2] * pD[0+ii*sdd+ps*1];
		v30 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*0];
		v31 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*1];
		v32 += pD[0+ii*sdd+ps*3] * pD[0+ii*sdd+ps*2];
		v10 += pD[1+ii*sdd+ps*1] * pD[1+ii*sdd+ps*0];
		v20 += pD[1+ii*sdd+ps*2] * pD[1+ii*sdd+ps*0];
		v21 += pD[1+ii*sdd+ps*2] * pD[1+ii*sdd+ps*1];
		v30 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*0];
		v31 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*1];
		v32 += pD[1+ii*sdd+ps*3] * pD[1+ii*sdd+ps*2];
		v10 += pD[2+ii*sdd+ps*1] * pD[2+ii*sdd+ps*0];
		v20 += pD[2+ii*sdd+ps*2] * pD[2+ii*sdd+ps*0];
		v21 += pD[2+ii*sdd+ps*2] * pD[2+ii*sdd+ps*1];
		v30 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*0];
		v31 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*1];
		v32 += pD[2+ii*sdd+ps*3] * pD[2+ii*sdd+ps*2];
		v10 += pD[3+ii*sdd+ps*1] * pD[3+ii*sdd+ps*0];
		v20 += pD[3+ii*sdd+ps*2] * pD[3+ii*sdd+ps*0];
		v21 += pD[3+ii*sdd+ps*2] * pD[3+ii*sdd+ps*1];
		v30 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*0];
		v31 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*1];
		v32 += pD[3+ii*sdd+ps*3] * pD[3+ii*sdd+ps*2];
		}
	for(ll=0; ll<m-ii; ll++)
		{
		v10 += pD[ll+ii*sdd+ps*1] * pD[ll+ii*sdd+ps*0];
		v20 += pD[ll+ii*sdd+ps*2] * pD[ll+ii*sdd+ps*0];
		v21 += pD[ll+ii*sdd+ps*2] * pD[ll+ii*sdd+ps*1];
		v30 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*0];
		v31 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*1];
		v32 += pD[ll+ii*sdd+ps*3] * pD[ll+ii*sdd+ps*2];
		}
	// compute lower triangular T containing tau for matrix update
	pT[0+ldt*0] = dD[0];
	pT[1+ldt*1] = dD[1];
	pT[2+ldt*2] = dD[2];
	pT[3+ldt*3] = dD[3];
	pT[1+ldt*0] = - dD[1] * (v10*pT[0+ldt*0]);
	pT[2+ldt*1] = - dD[2] * (v21*pT[1+ldt*1]);
	pT[3+ldt*2] = - dD[3] * (v32*pT[2+ldt*2]);
	pT[2+ldt*0] = - dD[2] * (v20*pT[0+ldt*0] + v21*pT[1+ldt*0]);
	pT[3+ldt*1] = - dD[3] * (v31*pT[1+ldt*1] + v32*pT[2+ldt*1]);
	pT[3+ldt*0] = - dD[3] * (v30*pT[0+ldt*0] + v31*pT[1+ldt*0] + v32*pT[2+ldt*0]);
	// downgrade matrix
	ii = 0;
	for( ; ii<n-1; ii+=2)
		{
		pC = pC0+ii*ps;
		// compute W^T = C^T * V
		// jj=0
		tmp = pC[0+ps*0];
		pW[0+ldw*0] = tmp;
		tmp = pC[0+ps*1];
		pW[0+ldw*1] = tmp;
		// jj=1
		d0 = pVt[0+ps*1];
		tmp = pC[1+ps*0];
		pW[0+ldw*0] += d0 * tmp;
		pW[1+ldw*0] = tmp;
		tmp = pC[1+ps*1];
		pW[0+ldw*1] += d0 * tmp;
		pW[1+ldw*1] = tmp;
		// jj=2
		d0 = pVt[0+ps*2];
		d1 = pVt[1+ps*2];
		tmp = pC[2+ps*0];
		pW[0+ldw*0] += d0 * tmp;
		pW[1+ldw*0] += d1 * tmp;
		pW[2+ldw*0] = tmp;
		tmp = pC[2+ps*1];
		pW[0+ldw*1] += d0 * tmp;
		pW[1+ldw*1] += d1 * tmp;
		pW[2+ldw*1] = tmp;
		// jj=3
		d0 = pVt[0+ps*3];
		d1 = pVt[1+ps*3];
		d2 = pVt[2+ps*3];
		tmp = pC[3+ps*0];
		pW[0+ldw*0] += d0 * tmp;
		pW[1+ldw*0] += d1 * tmp;
		pW[2+ldw*0] += d2 * tmp;
		pW[3+ldw*0] = tmp;
		tmp = pC[3+ps*1];
		pW[0+ldw*1] += d0 * tmp;
		pW[1+ldw*1] += d1 * tmp;
		pW[2+ldw*1] += d2 * tmp;
		pW[3+ldw*1] = tmp;
		for(jj=4; jj<m-3; jj+=4)
			{
			//
			d0 = pVt[0+ps*(0+jj)];
			d1 = pVt[1+ps*(0+jj)];
			d2 = pVt[2+ps*(0+jj)];
			d3 = pVt[3+ps*(0+jj)];
			tmp = pC[0+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			tmp = pC[0+jj*sdc+ps*1];
			pW[0+ldw*1] += d0 * tmp;
			pW[1+ldw*1] += d1 * tmp;
			pW[2+ldw*1] += d2 * tmp;
			pW[3+ldw*1] += d3 * tmp;
			//
			d0 = pVt[0+ps*(1+jj)];
			d1 = pVt[1+ps*(1+jj)];
			d2 = pVt[2+ps*(1+jj)];
			d3 = pVt[3+ps*(1+jj)];
			tmp = pC[1+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			tmp = pC[1+jj*sdc+ps*1];
			pW[0+ldw*1] += d0 * tmp;
			pW[1+ldw*1] += d1 * tmp;
			pW[2+ldw*1] += d2 * tmp;
			pW[3+ldw*1] += d3 * tmp;
			//
			d0 = pVt[0+ps*(2+jj)];
			d1 = pVt[1+ps*(2+jj)];
			d2 = pVt[2+ps*(2+jj)];
			d3 = pVt[3+ps*(2+jj)];
			tmp = pC[2+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			tmp = pC[2+jj*sdc+ps*1];
			pW[0+ldw*1] += d0 * tmp;
			pW[1+ldw*1] += d1 * tmp;
			pW[2+ldw*1] += d2 * tmp;
			pW[3+ldw*1] += d3 * tmp;
			//
			d0 = pVt[0+ps*(3+jj)];
			d1 = pVt[1+ps*(3+jj)];
			d2 = pVt[2+ps*(3+jj)];
			d3 = pVt[3+ps*(3+jj)];
			tmp = pC[3+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			tmp = pC[3+jj*sdc+ps*1];
			pW[0+ldw*1] += d0 * tmp;
			pW[1+ldw*1] += d1 * tmp;
			pW[2+ldw*1] += d2 * tmp;
			pW[3+ldw*1] += d3 * tmp;
			}
		for(ll=0; ll<m-jj; ll++)
			{
			d0 = pVt[0+ps*(ll+jj)];
			d1 = pVt[1+ps*(ll+jj)];
			d2 = pVt[2+ps*(ll+jj)];
			d3 = pVt[3+ps*(ll+jj)];
			tmp = pC[ll+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			tmp = pC[ll+jj*sdc+ps*1];
			pW[0+ldw*1] += d0 * tmp;
			pW[1+ldw*1] += d1 * tmp;
			pW[2+ldw*1] += d2 * tmp;
			pW[3+ldw*1] += d3 * tmp;
			}
		// compute W^T *= T
		pW[3+ldw*0] = pT[3+ldt*0]*pW[0+ldw*0] + pT[3+ldt*1]*pW[1+ldw*0] + pT[3+ldt*2]*pW[2+ldw*0] + pT[3+ldt*3]*pW[3+ldw*0];
		pW[3+ldw*1] = pT[3+ldt*0]*pW[0+ldw*1] + pT[3+ldt*1]*pW[1+ldw*1] + pT[3+ldt*2]*pW[2+ldw*1] + pT[3+ldt*3]*pW[3+ldw*1];
		pW[2+ldw*0] = pT[2+ldt*0]*pW[0+ldw*0] + pT[2+ldt*1]*pW[1+ldw*0] + pT[2+ldt*2]*pW[2+ldw*0];
		pW[2+ldw*1] = pT[2+ldt*0]*pW[0+ldw*1] + pT[2+ldt*1]*pW[1+ldw*1] + pT[2+ldt*2]*pW[2+ldw*1];
		pW[1+ldw*0] = pT[1+ldt*0]*pW[0+ldw*0] + pT[1+ldt*1]*pW[1+ldw*0];
		pW[1+ldw*1] = pT[1+ldt*0]*pW[0+ldw*1] + pT[1+ldt*1]*pW[1+ldw*1];
		pW[0+ldw*0] = pT[0+ldt*0]*pW[0+ldw*0];
		pW[0+ldw*1] = pT[0+ldt*0]*pW[0+ldw*1];
		// compute C -= V * W^T
		jj = 0;
		// load
		c00 = pC[0+jj*sdc+ps*0];
		c10 = pC[1+jj*sdc+ps*0];
		c20 = pC[2+jj*sdc+ps*0];
		c30 = pC[3+jj*sdc+ps*0];
		c01 = pC[0+jj*sdc+ps*1];
		c11 = pC[1+jj*sdc+ps*1];
		c21 = pC[2+jj*sdc+ps*1];
		c31 = pC[3+jj*sdc+ps*1];
		// rank1
		a1 = pD[1+jj*sdd+ps*0];
		a2 = pD[2+jj*sdd+ps*0];
		a3 = pD[3+jj*sdd+ps*0];
		b0 = pW[0+ldw*0];
		c00 -= b0;
		c10 -= a1*b0;
		c20 -= a2*b0;
		c30 -= a3*b0;
		b1 = pW[0+ldw*1];
		c01 -= b1;
		c11 -= a1*b1;
		c21 -= a2*b1;
		c31 -= a3*b1;
		// rank2
		a2 = pD[2+jj*sdd+ps*1];
		a3 = pD[3+jj*sdd+ps*1];
		b0 = pW[1+ldw*0];
		c10 -= b0;
		c20 -= a2*b0;
		c30 -= a3*b0;
		b1 = pW[1+ldw*1];
		c11 -= b1;
		c21 -= a2*b1;
		c31 -= a3*b1;
		// rank3
		a3 = pD[3+jj*sdd+ps*2];
		b0 = pW[2+ldw*0];
		c20 -= b0;
		c30 -= a3*b0;
		b1 = pW[2+ldw*1];
		c21 -= b1;
		c31 -= a3*b1;
		// rank4
		a3 = pD[3+jj*sdd+ps*3];
		b0 = pW[3+ldw*0];
		c30 -= b0;
		b1 = pW[3+ldw*1];
		c31 -= b1;
		// store
		pC[0+jj*sdc+ps*0] = c00;
		pC[1+jj*sdc+ps*0] = c10;
		pC[2+jj*sdc+ps*0] = c20;
		pC[3+jj*sdc+ps*0] = c30;
		pC[0+jj*sdc+ps*1] = c01;
		pC[1+jj*sdc+ps*1] = c11;
		pC[2+jj*sdc+ps*1] = c21;
		pC[3+jj*sdc+ps*1] = c31;
		for(jj=4; jj<m-3; jj+=4)
			{
			// load
			c00 = pC[0+jj*sdc+ps*0];
			c10 = pC[1+jj*sdc+ps*0];
			c20 = pC[2+jj*sdc+ps*0];
			c30 = pC[3+jj*sdc+ps*0];
			c01 = pC[0+jj*sdc+ps*1];
			c11 = pC[1+jj*sdc+ps*1];
			c21 = pC[2+jj*sdc+ps*1];
			c31 = pC[3+jj*sdc+ps*1];
			//
			a0 = pD[0+jj*sdd+ps*0];
			a1 = pD[1+jj*sdd+ps*0];
			a2 = pD[2+jj*sdd+ps*0];
			a3 = pD[3+jj*sdd+ps*0];
			b0 = pW[0+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			b1 = pW[0+ldw*1];
			c01 -= a0*b1;
			c11 -= a1*b1;
			c21 -= a2*b1;
			c31 -= a3*b1;
			//
			a0 = pD[0+jj*sdd+ps*1];
			a1 = pD[1+jj*sdd+ps*1];
			a2 = pD[2+jj*sdd+ps*1];
			a3 = pD[3+jj*sdd+ps*1];
			b0 = pW[1+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			b1 = pW[1+ldw*1];
			c01 -= a0*b1;
			c11 -= a1*b1;
			c21 -= a2*b1;
			c31 -= a3*b1;
			//
			a0 = pD[0+jj*sdd+ps*2];
			a1 = pD[1+jj*sdd+ps*2];
			a2 = pD[2+jj*sdd+ps*2];
			a3 = pD[3+jj*sdd+ps*2];
			b0 = pW[2+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			b1 = pW[2+ldw*1];
			c01 -= a0*b1;
			c11 -= a1*b1;
			c21 -= a2*b1;
			c31 -= a3*b1;
			//
			a0 = pD[0+jj*sdd+ps*3];
			a1 = pD[1+jj*sdd+ps*3];
			a2 = pD[2+jj*sdd+ps*3];
			a3 = pD[3+jj*sdd+ps*3];
			b0 = pW[3+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			b1 = pW[3+ldw*1];
			c01 -= a0*b1;
			c11 -= a1*b1;
			c21 -= a2*b1;
			c31 -= a3*b1;
			// store
			pC[0+jj*sdc+ps*0] = c00;
			pC[1+jj*sdc+ps*0] = c10;
			pC[2+jj*sdc+ps*0] = c20;
			pC[3+jj*sdc+ps*0] = c30;
			pC[0+jj*sdc+ps*1] = c01;
			pC[1+jj*sdc+ps*1] = c11;
			pC[2+jj*sdc+ps*1] = c21;
			pC[3+jj*sdc+ps*1] = c31;
			}
		for(ll=0; ll<m-jj; ll++)
			{
			// load
			c00 = pC[ll+jj*sdc+ps*0];
			c01 = pC[ll+jj*sdc+ps*1];
			//
			a0 = pD[ll+jj*sdd+ps*0];
			b0 = pW[0+ldw*0];
			c00 -= a0*b0;
			b1 = pW[0+ldw*1];
			c01 -= a0*b1;
			//
			a0 = pD[ll+jj*sdd+ps*1];
			b0 = pW[1+ldw*0];
			c00 -= a0*b0;
			b1 = pW[1+ldw*1];
			c01 -= a0*b1;
			//
			a0 = pD[ll+jj*sdd+ps*2];
			b0 = pW[2+ldw*0];
			c00 -= a0*b0;
			b1 = pW[2+ldw*1];
			c01 -= a0*b1;
			//
			a0 = pD[ll+jj*sdd+ps*3];
			b0 = pW[3+ldw*0];
			c00 -= a0*b0;
			b1 = pW[3+ldw*1];
			c01 -= a0*b1;
			// store
			pC[ll+jj*sdc+ps*0] = c00;
			pC[ll+jj*sdc+ps*1] = c01;
			}
		}
	for( ; ii<n; ii++)
		{
		pC = pC0+ii*ps;
		// compute W^T = C^T * V
		// jj=0
		tmp = pC[0+ps*0];
		pW[0+ldw*0] = tmp;
		// jj=1
		d0 = pVt[0+ps*1];
		tmp = pC[1+ps*0];
		pW[0+ldw*0] += d0 * tmp;
		pW[1+ldw*0] = tmp;
		// jj=2
		d0 = pVt[0+ps*2];
		d1 = pVt[1+ps*2];
		tmp = pC[2+ps*0];
		pW[0+ldw*0] += d0 * tmp;
		pW[1+ldw*0] += d1 * tmp;
		pW[2+ldw*0] = tmp;
		// jj=3
		d0 = pVt[0+ps*3];
		d1 = pVt[1+ps*3];
		d2 = pVt[2+ps*3];
		tmp = pC[3+ps*0];
		pW[0+ldw*0] += d0 * tmp;
		pW[1+ldw*0] += d1 * tmp;
		pW[2+ldw*0] += d2 * tmp;
		pW[3+ldw*0] = tmp;
		for(jj=4; jj<m-3; jj+=4)
			{
			//
			d0 = pVt[0+ps*(0+jj)];
			d1 = pVt[1+ps*(0+jj)];
			d2 = pVt[2+ps*(0+jj)];
			d3 = pVt[3+ps*(0+jj)];
			tmp = pC[0+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			//
			d0 = pVt[0+ps*(1+jj)];
			d1 = pVt[1+ps*(1+jj)];
			d2 = pVt[2+ps*(1+jj)];
			d3 = pVt[3+ps*(1+jj)];
			tmp = pC[1+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			//
			d0 = pVt[0+ps*(2+jj)];
			d1 = pVt[1+ps*(2+jj)];
			d2 = pVt[2+ps*(2+jj)];
			d3 = pVt[3+ps*(2+jj)];
			tmp = pC[2+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			//
			d0 = pVt[0+ps*(3+jj)];
			d1 = pVt[1+ps*(3+jj)];
			d2 = pVt[2+ps*(3+jj)];
			d3 = pVt[3+ps*(3+jj)];
			tmp = pC[3+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			}
		for(ll=0; ll<m-jj; ll++)
			{
			d0 = pVt[0+ps*(ll+jj)];
			d1 = pVt[1+ps*(ll+jj)];
			d2 = pVt[2+ps*(ll+jj)];
			d3 = pVt[3+ps*(ll+jj)];
			tmp = pC[ll+jj*sdc+ps*0];
			pW[0+ldw*0] += d0 * tmp;
			pW[1+ldw*0] += d1 * tmp;
			pW[2+ldw*0] += d2 * tmp;
			pW[3+ldw*0] += d3 * tmp;
			}
		// compute W^T *= T
		pW[3+ldw*0] = pT[3+ldt*0]*pW[0+ldw*0] + pT[3+ldt*1]*pW[1+ldw*0] + pT[3+ldt*2]*pW[2+ldw*0] + pT[3+ldt*3]*pW[3+ldw*0];
		pW[2+ldw*0] = pT[2+ldt*0]*pW[0+ldw*0] + pT[2+ldt*1]*pW[1+ldw*0] + pT[2+ldt*2]*pW[2+ldw*0];
		pW[1+ldw*0] = pT[1+ldt*0]*pW[0+ldw*0] + pT[1+ldt*1]*pW[1+ldw*0];
		pW[0+ldw*0] = pT[0+ldt*0]*pW[0+ldw*0];
		// compute C -= V * W^T
		jj = 0;
		// load
		c00 = pC[0+jj*sdc+ps*0];
		c10 = pC[1+jj*sdc+ps*0];
		c20 = pC[2+jj*sdc+ps*0];
		c30 = pC[3+jj*sdc+ps*0];
		// rank1
		a1 = pD[1+jj*sdd+ps*0];
		a2 = pD[2+jj*sdd+ps*0];
		a3 = pD[3+jj*sdd+ps*0];
		b0 = pW[0+ldw*0];
		c00 -= b0;
		c10 -= a1*b0;
		c20 -= a2*b0;
		c30 -= a3*b0;
		// rank2
		a2 = pD[2+jj*sdd+ps*1];
		a3 = pD[3+jj*sdd+ps*1];
		b0 = pW[1+ldw*0];
		c10 -= b0;
		c20 -= a2*b0;
		c30 -= a3*b0;
		// rank3
		a3 = pD[3+jj*sdd+ps*2];
		b0 = pW[2+ldw*0];
		c20 -= b0;
		c30 -= a3*b0;
		// rank4
		a3 = pD[3+jj*sdd+ps*3];
		b0 = pW[3+ldw*0];
		c30 -= b0;
		// store
		pC[0+jj*sdc+ps*0] = c00;
		pC[1+jj*sdc+ps*0] = c10;
		pC[2+jj*sdc+ps*0] = c20;
		pC[3+jj*sdc+ps*0] = c30;
		for(jj=4; jj<m-3; jj+=4)
			{
			// load
			c00 = pC[0+jj*sdc+ps*0];
			c10 = pC[1+jj*sdc+ps*0];
			c20 = pC[2+jj*sdc+ps*0];
			c30 = pC[3+jj*sdc+ps*0];
			//
			a0 = pD[0+jj*sdd+ps*0];
			a1 = pD[1+jj*sdd+ps*0];
			a2 = pD[2+jj*sdd+ps*0];
			a3 = pD[3+jj*sdd+ps*0];
			b0 = pW[0+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			//
			a0 = pD[0+jj*sdd+ps*1];
			a1 = pD[1+jj*sdd+ps*1];
			a2 = pD[2+jj*sdd+ps*1];
			a3 = pD[3+jj*sdd+ps*1];
			b0 = pW[1+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			//
			a0 = pD[0+jj*sdd+ps*2];
			a1 = pD[1+jj*sdd+ps*2];
			a2 = pD[2+jj*sdd+ps*2];
			a3 = pD[3+jj*sdd+ps*2];
			b0 = pW[2+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			//
			a0 = pD[0+jj*sdd+ps*3];
			a1 = pD[1+jj*sdd+ps*3];
			a2 = pD[2+jj*sdd+ps*3];
			a3 = pD[3+jj*sdd+ps*3];
			b0 = pW[3+ldw*0];
			c00 -= a0*b0;
			c10 -= a1*b0;
			c20 -= a2*b0;
			c30 -= a3*b0;
			// store
			pC[0+jj*sdc+ps*0] = c00;
			pC[1+jj*sdc+ps*0] = c10;
			pC[2+jj*sdc+ps*0] = c20;
			pC[3+jj*sdc+ps*0] = c30;
			}
		for(ll=0; ll<m-jj; ll++)
			{
			// load
			c00 = pC[ll+jj*sdc+ps*0];
			//
			a0 = pD[ll+jj*sdd+ps*0];
			b0 = pW[0+ldw*0];
			c00 -= a0*b0;
			//
			a0 = pD[ll+jj*sdd+ps*1];
			b0 = pW[1+ldw*0];
			c00 -= a0*b0;
			//
			a0 = pD[ll+jj*sdd+ps*2];
			b0 = pW[2+ldw*0];
			c00 -= a0*b0;
			//
			a0 = pD[ll+jj*sdd+ps*3];
			b0 = pW[3+ldw*0];
			c00 -= a0*b0;
			// store
			pC[ll+jj*sdc+ps*0] = c00;
			}
		}

	return;
	}
