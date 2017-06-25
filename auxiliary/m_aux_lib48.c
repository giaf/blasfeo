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




void m_cvt_d2s_strvec(int m, struct d_strvec *vd, int vdi, struct s_strvec *vs, int vsi)
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



void m_cvt_s2d_strvec(int m, struct s_strvec *vs, int vsi, struct d_strvec *vd, int vdi)
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



void m_cvt_d2s_strmat(int m, int n, struct d_strmat *Md, int mid, int nid, struct s_strmat *Ms, int mis, int nis)
	{
	printf("\nm_cvt_d2s_strmat: feature not implmeneted yet\n\n");
	exit(1);
	return;
	}



void m_cvt_s2d_strmat(int m, int n, struct s_strmat *Ms, int mis, int nis, struct d_strmat *Md, int mid, int nid)
	{
	printf("\nm_cvt_s2d_strmat: feature not implmeneted yet\n\n");
	exit(1);
	return;
	}



#else

#error : wrong LA choice

#endif


