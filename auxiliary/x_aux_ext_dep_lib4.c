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

#include "../include/blasfeo_common.h"


#if defined(LA_HIGH_PERFORMANCE)


// create a matrix structure for a matrix of size m*n by dynamically allocating the memory
void ALLOCATE_STRMAT(int m, int n, struct STRMAT *sA)
	{
	const int ps = PS;
	int nc = D_NC;
	int al = ps*nc;
	sA->m = m;
	sA->n = n;
	int pm = (m+ps-1)/ps*ps;
	int cn = (n+nc-1)/nc*nc;
	sA->pm = pm;
	sA->cn = cn;
	ZEROS_ALIGN(&(sA->pA), sA->pm, sA->cn);
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	ZEROS_ALIGN(&(sA->dA), tmp, 1);
	sA->use_dA = 0;
	sA->memsize = (pm*cn+tmp)*sizeof(REAL);
	return;
	}



// free memory of a matrix structure
void FREE_STRMAT(struct STRMAT *sA)
	{
	FREE_ALIGN(sA->pA);
	FREE_ALIGN(sA->dA);
	return;
	}



// create a vector structure for a vector of size m by dynamically allocating the memory
void ALLOCATE_STRVEC(int m, struct STRVEC *sa)
	{
	const int ps = PS;
//	int nc = D_NC;
//	int al = ps*nc;
	sa->m = m;
	int pm = (m+ps-1)/ps*ps;
	sa->pm = pm;
	ZEROS_ALIGN(&(sa->pa), sa->pm, 1);
	sa->memsize = pm*sizeof(REAL);
	return;
	}



// free memory of a matrix structure
void FREE_STRVEC(struct STRVEC *sa)
	{
	FREE_ALIGN(sa->pa);
	return;
	}



// print a matrix structure
void PRINT_STRMAT(int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	const int ps = PS;
	int sda = sA->cn;
	REAL *pA = sA->pA + aj*ps + ai/ps*ps*sda + ai%ps;
	int ii, i, j, tmp;
	ii = 0;
	if(ai%ps>0)
		{
		tmp = ps-ai%ps;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%9.5f ", pA[i+ps*j]);
				}
			printf("\n");
			}
		pA += tmp + ps*(sda-1);
		m -= tmp;
		}
	for( ; ii<m-(ps-1); ii+=ps)
		{
		for(i=0; i<ps; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%9.5f ", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			}
		}
	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%9.5f ", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");
	return;
	}



// print a vector structure
void PRINT_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_MAT(m, 1, pa, m);
	return;
	}



// print the transposed of a vector structure
void PRINT_TRAN_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_MAT(1, m, pa, 1);
	return;
	}



// print a matrix structure
void PRINT_TO_FILE_STRMAT(FILE * file, int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	const int ps = PS;
	int sda = sA->cn;
	REAL *pA = sA->pA + aj*ps + ai/ps*ps*sda + ai%ps;
	int ii, i, j, tmp;
	ii = 0;
	if(ai%ps>0)
		{
		tmp = ps-ai%ps;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				fprintf(file, "%9.5f ", pA[i+ps*j]);
				}
			fprintf(file, "\n");
			}
		pA += tmp + ps*(sda-1);
		m -= tmp;
		}
	for( ; ii<m-(ps-1); ii+=ps)
		{
		for(i=0; i<ps; i++)
			{
			for(j=0; j<n; j++)
				{
				fprintf(file, "%9.5f ", pA[i+ps*j+sda*ii]);
				}
			fprintf(file, "\n");
			}
		}
	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				fprintf(file, "%9.5f ", pA[i+ps*j+sda*ii]);
				}
			fprintf(file, "\n");
			}
		}
	fprintf(file, "\n");
	return;
	}



// print a vector structure
void PRINT_TO_FILE_STRVEC(FILE * file, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_FILE_MAT(file, m, 1, pa, m);
	return;
	}



// print the transposed of a vector structure
void PRINT_TO_FILE_TRAN_STRVEC(FILE * file, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_FILE_MAT(file, 1, m, pa, 1);
	return;
	}



// print a matrix structure
void PRINT_E_STRMAT(int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	const int ps = PS;
	int sda = sA->cn;
	REAL *pA = sA->pA + aj*ps + ai/ps*ps*sda + ai%ps;
	int ii, i, j, tmp;
	ii = 0;
	if(ai%ps>0)
		{
		tmp = ps-ai%ps;
		tmp = m<tmp ? m : tmp;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%e\t", pA[i+ps*j]);
				}
			printf("\n");
			}
		pA += tmp + ps*(sda-1);
		m -= tmp;
		}
	for( ; ii<m-(ps-1); ii+=ps)
		{
		for(i=0; i<ps; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%e\t", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			}
		}
	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				printf("%e\t", pA[i+ps*j+sda*ii]);
				}
			printf("\n");
			}
		}
	printf("\n");
	return;
	}



// print a vector structure
void PRINT_E_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_E_MAT(m, 1, pa, m);
	return;
	}



// print the transposed of a vector structure
void PRINT_E_TRAN_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_E_MAT(1, m, pa, 1);
	return;
	}


#endif
