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

void GEMV_DIAG_LIBSTR(int m, REAL alpha, struct STRVEC *sA, int ai, struct STRVEC *sx, int xi, REAL beta, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	int ii;
	REAL *a = sA->pa + ai;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	if(alpha==1.0 & beta==1.0)
		{
		for(ii=0; ii<m; ii++)
			z[ii] = a[ii]*x[ii] + y[ii];
		}
	else
		{
		for(ii=0; ii<m; ii++)
			z[ii] = alpha*a[ii]*x[ii] + beta*y[ii];
		}

	return;

	}
