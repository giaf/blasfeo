/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

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
	// invalidate stored inverse diagonal
	sA->use_dA = 0;

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



// print the transposed of a matrix structure
void PRINT_TRAN_STRMAT(int m, int n, struct STRMAT *sA, int ai, int aj)
	{
	const int ps = PS;
	int sda = sA->cn;
	REAL *pA = sA->pA + aj*ps + ai/ps*ps*sda + ai%ps;
	int ii, i, j, tmp;
	REAL *pA_bkp = pA;
	int m_bkp = m;
	ii = 0;
	for(j=0; j<n; j++)
		{
		if(ai%ps>0)
			{
			pA = pA_bkp;
			m = m_bkp;
			tmp = ps-ai%ps;
			tmp = m<tmp ? m : tmp;
			for(i=0; i<tmp; i++)
				{
				printf("%9.5f ", pA[i+ps*j]);
				}
			pA += tmp + ps*(sda-1);
			m -= tmp;
			}
		for( ; ii<m-(ps-1); ii+=ps)
			{
			for(i=0; i<ps; i++)
				{
				printf("%9.5f ", pA[i+ps*j+sda*ii]);
				}
			}
		if(ii<m)
			{
			tmp = m-ii;
			for(i=0; i<tmp; i++)
				{
				printf("%9.5f ", pA[i+ps*j+sda*ii]);
				}
			}
		printf("\n");
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

// print a matrix structure
void PRINT_TO_FILE_EXP_STRMAT(FILE * file, int m, int n, struct STRMAT *sA, int ai, int aj)
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
				fprintf(file, "%9.5e ", pA[i+ps*j]);
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
				fprintf(file, "%9.5e ", pA[i+ps*j+sda*ii]);
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
				fprintf(file, "%9.5e ", pA[i+ps*j+sda*ii]);
				}
			fprintf(file, "\n");
			}
		}
	fprintf(file, "\n");
	return;
	}


// print a matrix structure
void PRINT_TO_STRING_STRMAT(char **buf_out, int m, int n, struct STRMAT *sA, int ai, int aj)
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
				*buf_out += sprintf(*buf_out, "%9.5f ", pA[i+ps*j]);
				}
			*buf_out += sprintf(*buf_out, "\n");
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
				*buf_out += sprintf(*buf_out, "%9.5f ", pA[i+ps*j+sda*ii]);
				}
			*buf_out += sprintf(*buf_out, "\n");
			}
		}
	if(ii<m)
		{
		tmp = m-ii;
		for(i=0; i<tmp; i++)
			{
			for(j=0; j<n; j++)
				{
				*buf_out += sprintf(*buf_out, "%9.5f ", pA[i+ps*j+sda*ii]);
				}
			*buf_out += sprintf(*buf_out, "\n");
			}
		}
	*buf_out += sprintf(*buf_out, "\n");
	return;
	}


// print a vector structure
void PRINT_TO_FILE_STRVEC(FILE * file, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_FILE_MAT(file, m, 1, pa, m);
	return;
	}



// print a vector structure
void PRINT_TO_STRING_STRVEC(char **buf_out, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_STRING_MAT(buf_out, m, 1, pa, m);
	return;
	}


// print the transposed of a vector structure
void PRINT_TO_FILE_TRAN_STRVEC(FILE * file, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_FILE_MAT(file, 1, m, pa, 1);
	return;
	}



// print the transposed of a vector structure
void PRINT_TO_STRING_TRAN_STRVEC(char **buf_out, int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_TO_STRING_MAT(buf_out, 1, m, pa, 1);
	return;
	}


// print a matrix structure
void PRINT_EXP_STRMAT(int m, int n, struct STRMAT *sA, int ai, int aj)
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
void PRINT_EXP_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_EXP_MAT(m, 1, pa, m);
	return;
	}



// print the transposed of a vector structure
void PRINT_EXP_TRAN_STRVEC(int m, struct STRVEC *sa, int ai)
	{
	REAL *pa = sa->pa + ai;
	PRINT_EXP_MAT(1, m, pa, 1);
	return;
	}


#endif
