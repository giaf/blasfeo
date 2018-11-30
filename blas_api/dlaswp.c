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



#if defined(FORTRAN_BLAS_API)
#define blasfeo_dlaswp dlaswp_
#endif



void blasfeo_dlaswp(int *pn, double *A, int *plda, int *pk1, int *pk2, int *ipiv, int *pincx)
	{

	int n = *pn;
	int lda = *plda;
	int k1 = *pk1;
	int k2 = *pk2;
	int incx = *pincx;

	int ix0, i1, i2, inc;

	int ii, jj, ix, ip;

	double tmp;

	if(incx>=0)
		{
		ix0 = k1;
		i1 = k1;
		i2 = k2;
		inc = 1;
		}
	else
		{
		ix0 = k1 + (k1-k2)*incx;
		i1 = k2;
		i2 = k1;
		inc = -1;
		}
	
	ix = ix0;
	for(ii=i1; ii<=i2; ii+=inc)
		{
		ip = ipiv[ix];
		if(ip!=ii)
			{
			for(jj=0; jj<n; jj++)
				{
				tmp = A[ii+jj*lda];
				A[ii+jj*lda] = A[ip+jj*lda];
				A[ip+jj*lda] = tmp;
				}
			}
		ix = ix + incx;
		}

	return;

	}

