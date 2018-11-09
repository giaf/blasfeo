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
#include "../include/blasfeo_d_blas.h"



#if defined(FORTRAN_BLAS_API)
#define blasfeo_dtrsm dtrsm_
#define blasfeo_dpotrs dpotrs_
#endif



void blasfeo_dpotrs(char *uplo, int *pm, int *pn, double *A, int *plda, double *B, int *pldb, int *info)
	{

//	printf("\nblasfeo_dpotrs\n");

	int m = *pm;
	int n = *pn;

	char c_l = 'l';
	char c_n = 'n';
	char c_t = 't';
	char c_u = 'u';

	double d_1 = 1.0;

	*info = 0;

	if(m==0 | n==0)
		return;
	
	if(*uplo=='u' | *uplo=='U')
		{
		blasfeo_dtrsm(&c_l, &c_u, &c_t, &c_n, pm, pn, &d_1, A, plda, B, pldb);
		blasfeo_dtrsm(&c_l, &c_u, &c_n, &c_n, pm, pn, &d_1, A, plda, B, pldb);
		}
	else
		{
		blasfeo_dtrsm(&c_l, &c_l, &c_n, &c_n, pm, pn, &d_1, A, plda, B, pldb);
		blasfeo_dtrsm(&c_l, &c_l, &c_t, &c_n, pm, pn, &d_1, A, plda, B, pldb);
		}

	return;

	}


