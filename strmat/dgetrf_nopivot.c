/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison. All rights reserved.                                     *
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



#if defined(BLAS_LA)

void dgetf2_nopivot(int m, int n, double *A, int lda)
	{

	if(m<=0 | n<=0)
		return;
	
	int i, j, itmp0, itmp1;
	int jmax = m<n ? m : n;
	int i1 = 1;
	double dtmp;
	double dm1 = -1.0;

	for(j=0; j<jmax; j++)
		{
		itmp0 = m-j-1;
		dtmp = 1.0/A[j+lda*j];
		dscal_(&itmp0, &dtmp, &A[(j+1)+lda*j], &i1);
		itmp1 = n-j-1;
		dger_(&itmp0, &itmp1, &dm1, &A[(j+1)+lda*j], &i1, &A[j+lda*(j+1)], &lda, &A[(j+1)+lda*(j+1)], &lda);
		}
	
	return;

	}

#endif
