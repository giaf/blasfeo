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



void GEMM(char *ta, char *tb, int *pm, int *pn, int *pk, REAL *palpha, REAL *A, int *plda, REAL *B, int *pldb, REAL *pbeta, REAL *C, int *pldc)
	{

	struct MAT sA;
	sA.pA = A;
	sA.m = *plda;

	struct MAT sB;
	sB.pA = B;
	sB.m = *pldb;

	struct MAT sC;
	sC.pA = C;
	sC.m = *pldc;

	if(*ta=='n' | *ta=='N')
		{
		if(*tb=='n' | *tb=='N')
			{
			GEMM_NN(*pm, *pn, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		else if(*tb=='t' | *tb=='T' | *tb=='c' | *tb=='C')
			{
			GEMM_NT(*pm, *pn, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		else
			{
			printf("\nBLASFEO: gemm: wrong value for tb\n");
			return;
			}
		}
	else if(*ta=='t' | *ta=='T' | *ta=='c' | *ta=='C')
		{
		if(*tb=='n' | *tb=='N')
			{
			GEMM_TN(*pm, *pn, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		else if(*tb=='t' | *tb=='T' | *tb=='c'| *tb=='C')
			{
			GEMM_TT(*pm, *pn, *pk, *palpha, &sA, 0, 0, &sB, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
			}
		else
			{
			printf("\nBLASFEO: gemm: wrong value for tb\n");
			return;
			}
		}
	else
		{
		printf("\nBLASFEO: gemm: wrong value for ta\n");
		return;
		}

	return;

	}
