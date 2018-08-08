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
#include <math.h>

#include "../include/blasfeo_common.h"



#if defined(LA_HIGH_PERFORMANCE)




void blasfeo_cvt_d2s_vec(int m, struct blasfeo_dvec *vd, int vdi, struct blasfeo_svec *vs, int vsi)
	{
	double *pd = vd->pa+vdi;
	float *ps = vs->pa+vsi;
	int ii;
	for(ii=0; ii<m; ii++)
		{
		ps[ii] = (float) pd[ii];
		}
	return;
	}



void blasfeo_cvt_s2d_vec(int m, struct blasfeo_svec *vs, int vsi, struct blasfeo_dvec *vd, int vdi)
	{
	double *pd = vd->pa+vdi;
	float *ps = vs->pa+vsi;
	int ii;
	for(ii=0; ii<m; ii++)
		{
		pd[ii] = (double) ps[ii];
		}
	return;
	}



void blasfeo_cvt_d2s_mat(int m, int n, struct blasfeo_dmat *Md, int mid, int nid, struct blasfeo_smat *Ms, int mis, int nis)
	{
	if(mid!=0 | mis!=0)
		{
		printf("\nblasfeo_cvt_d2s_mat: feature not implmeneted yet: mid=%d, mis=%d\n\n", mid, mis);
		exit(1);
		}
	const int psd = 4;
	const int pss = 8;
	const int sdd = Md->cn;
	double *D0 = Md->pA + nid*psd;
	double *D1;
	const int sds = Ms->cn;
	float *S = Ms->pA + nis*pss;
	int ii, jj, ll;
	for(ii=0; ii<m-7; ii+=8)
		{
		D1 = D0 + psd*sdd;
		for(jj=0; jj<n; jj++)
			{
			S[0+jj*pss] = (float) D0[0+jj*psd];
			S[1+jj*pss] = (float) D0[1+jj*psd];
			S[2+jj*pss] = (float) D0[2+jj*psd];
			S[3+jj*pss] = (float) D0[3+jj*psd];
			S[4+jj*pss] = (float) D1[0+jj*psd];
			S[5+jj*pss] = (float) D1[1+jj*psd];
			S[6+jj*pss] = (float) D1[2+jj*psd];
			S[7+jj*pss] = (float) D1[3+jj*psd];
			}
		D0 += 8*sdd;
		S  += 8*sds;
		}
	if(m-ii>0)
		{
		if(m-ii<4)
			{
			for(jj=0; jj<n; jj++)
				{
				for(ll=0; ll<m-ii; ll++)
					{
					S[ll+jj*pss] = (float) D0[ll+jj*psd];
					}
				}
			return;
			}
		else
			{
			D1 = D0 + psd*sdd;
			for(jj=0; jj<n; jj++)
				{
				S[0+jj*pss] = (float) D0[0+jj*psd];
				S[1+jj*pss] = (float) D0[1+jj*psd];
				S[2+jj*pss] = (float) D0[2+jj*psd];
				S[3+jj*pss] = (float) D0[3+jj*psd];
				for(ll=0; ll<m-ii-4; ll++)
					{
					S[4+ll+jj*pss] = (float) D1[ll+jj*psd];
					}
				}
			}
		}
	return;
	}



void blasfeo_cvt_s2d_mat(int m, int n, struct blasfeo_smat *Ms, int mis, int nis, struct blasfeo_dmat *Md, int mid, int nid)
	{
	printf("\nblasfeo_cvt_s2d_mat: feature not implmeneted yet\n\n");
	exit(1);
	return;
	}



#else

#error : wrong LA choice

#endif


