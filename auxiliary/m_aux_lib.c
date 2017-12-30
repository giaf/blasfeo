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



#if defined(LA_REFERENCE) | defined(LA_BLAS)




void m_cvt_d2blasfeo_svec(int m, struct blasfeo_dvec *vd, int vdi, struct blasfeo_svec *vs, int vsi)
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



void m_cvt_s2blasfeo_dvec(int m, struct blasfeo_svec *vs, int vsi, struct blasfeo_dvec *vd, int vdi)
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



void m_cvt_d2blasfeo_smat(int m, int n, struct blasfeo_dmat *Md, int mid, int nid, struct blasfeo_smat *Ms, int mis, int nis)
	{
	int lda = Md->m;
	int ldb = Ms->m;
	double *pA = Md->pA+mid+nid*lda;
	float *pB = Ms->pA+mis+nis*ldb;
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		for(ii=0; ii<m; ii++)
			{
			pB[ii+jj*ldb] = (float) pA[ii+jj*lda];
			}
		}
	return;
	}



void m_cvt_s2blasfeo_dmat(int m, int n, struct blasfeo_smat *Ms, int mis, int nis, struct blasfeo_dmat *Md, int mid, int nid)
	{
	int lda = Ms->m;
	int ldb = Md->m;
	float *pA = Ms->pA+mis+nis*lda;
	double *pB = Md->pA+mid+nid*ldb;
	int ii, jj;
	for(jj=0; jj<n; jj++)
		{
		for(ii=0; ii<m; ii++)
			{
			pB[ii+jj*ldb] = (double) pA[ii+jj*lda];
			}
		}
	return;
	}



#else

#error : wrong LA choice

#endif
