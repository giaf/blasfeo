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
#include "../include/blasfeo_block_size.h"
#include "../include/blasfeo_s_kernel.h"



// copies a lower triangular packed matrix into a lower triangular packed matrix
void strcp_l_lib(int m, int offsetA, float *A, int sda, int offsetB, float *B, int sdb)
	{
	printf("\nstrcp_;l_lib: feature not implemented yet\n");
	exit(1);
	}



// scales and adds a strvec into a strvec
void svecad_libstr(int m, float alpha, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci)
	{
	float *pa = sa->pa + ai;
	float *pc = sc->pa + ci;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pc[ii+0] += alpha*pa[ii+0];
		pc[ii+1] += alpha*pa[ii+1];
		pc[ii+2] += alpha*pa[ii+2];
		pc[ii+3] += alpha*pa[ii+3];
		}
	for(; ii<m; ii++)
		{
		pc[ii+0] += alpha*pa[ii+0];
		}
	return;
	}



// transpose lower triangular matrix
void strtr_l_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc)
	{
	printf("\nstrtr_l_lib: feature not implemented yet\n");
	exit(1);
	}



// transpose an aligned upper triangular matrix into an aligned lower triangular matrix
void strtr_u_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc)
	{
	printf("\nstrtr_u_lib: feature not implemented yet\n");
	exit(1);
	}



// regularize diagonal
void sdiareg_lib(int kmax, float reg, int offset, float *pD, int sdd)
	{

	const int bs = 8;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	float *pD2;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] += reg;
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		kmax -= kna;
		}
	pD2 = pD;
	for(jj=0; jj<kmax-7; jj+=8)
		{
		pD2[0+0*bs] += reg;
		pD2[1+1*bs] += reg;
		pD2[2+2*bs] += reg;
		pD2[3+3*bs] += reg;
		pD2[4+4*bs] += reg;
		pD2[5+5*bs] += reg;
		pD2[6+6*bs] += reg;
		pD2[7+7*bs] += reg;
		pD2 += bs*sdd+bs*bs;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] += reg;
		}

	}



// insert vector to diagonal
void sdiain_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd)
	{

	const int bs = 8;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	float *pD2, *x2;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] = alpha*x[ll];
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	pD2 = pD;
	x2 = x;
	for(jj=0; jj<kmax-7; jj+=8)
		{
		pD2[0+bs*0] = alpha*x2[0];
		pD2[1+bs*1] = alpha*x2[1];
		pD2[2+bs*2] = alpha*x2[2];
		pD2[3+bs*3] = alpha*x2[3];
		pD2[4+bs*4] = alpha*x2[4];
		pD2[5+bs*5] = alpha*x2[5];
		pD2[6+bs*6] = alpha*x2[6];
		pD2[7+bs*7] = alpha*x2[7];
		pD2 += bs*sdd+bs*bs;
		x2 += bs;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] = alpha*x[jj+ll];
		}

	}



// add scalar to diagonal
void sdiare_libstr(int kmax, float alpha, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	int offsetA = ai%bs;

	int kna = (bs-offsetA%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pA[ll+bs*ll] += alpha;
			}
		pA += kna + bs*(sda-1) + kna*bs;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-7; jj+=8)
		{
		pA[jj*sda+(jj+0)*bs+0] += alpha;
		pA[jj*sda+(jj+1)*bs+1] += alpha;
		pA[jj*sda+(jj+2)*bs+2] += alpha;
		pA[jj*sda+(jj+3)*bs+3] += alpha;
		pA[jj*sda+(jj+4)*bs+4] += alpha;
		pA[jj*sda+(jj+5)*bs+5] += alpha;
		pA[jj*sda+(jj+6)*bs+6] += alpha;
		pA[jj*sda+(jj+7)*bs+7] += alpha;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pA[jj*sda+(jj+ll)*bs+ll] += alpha;
		}
	return;
	}




// insert sqrt of vector to diagonal
void sdiain_sqrt_lib(int kmax, float *x, int offset, float *pD, int sdd)
	{

	const int bs = 8;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	float *pD2, *x2;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] = sqrt(x[ll]);
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	pD2 = pD;
	x2 = x;
	for(jj=0; jj<kmax-7; jj+=8)
		{
		pD2[0+bs*0] = sqrt(x2[0]);
		pD2[1+bs*1] = sqrt(x2[1]);
		pD2[2+bs*2] = sqrt(x2[2]);
		pD2[3+bs*3] = sqrt(x2[3]);
		pD2[4+bs*4] = sqrt(x2[4]);
		pD2[5+bs*5] = sqrt(x2[5]);
		pD2[5+bs*6] = sqrt(x2[6]);
		pD2[7+bs*7] = sqrt(x2[7]);
		pD2 += bs*sdd+bs*bs;
		x2 += bs;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] = sqrt(x[jj+ll]);
		}

	}



// extract diagonal to vector
void sdiaex_lib(int kmax, float alpha, int offset, float *pD, int sdd, float *x)
	{

	const int bs = 8;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	float *pD2, *x2;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			x[ll] = alpha * pD[ll+bs*ll];
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	pD2 = pD;
	x2 = x;
	for(jj=0; jj<kmax-7; jj+=8)
		{
		x2[0] = alpha * pD2[0+bs*0];
		x2[1] = alpha * pD2[1+bs*1];
		x2[2] = alpha * pD2[2+bs*2];
		x2[3] = alpha * pD2[3+bs*3];
		x2[4] = alpha * pD2[4+bs*4];
		x2[5] = alpha * pD2[5+bs*5];
		x2[6] = alpha * pD2[6+bs*6];
		x2[7] = alpha * pD2[7+bs*7];
		pD2 += bs*sdd+bs*bs;
		x2 += bs;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		x[jj+ll] = alpha * pD[jj*sdd+(jj+ll)*bs+ll];
		}

	}



// add scaled vector to diagonal
void sdiaad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd)
	{

	const int bs = 8;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	float *pD2, *x2;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll+bs*ll] += alpha * x[ll];
			}
		pD += kna + bs*(sdd-1) + kna*bs;
		x  += kna;
		kmax -= kna;
		}
	pD2 = pD;
	x2 = x;
	for(jj=0; jj<kmax-7; jj+=8)
		{
		pD2[0+bs*0] += alpha * x2[0];
		pD2[1+bs*1] += alpha * x2[1];
		pD2[2+bs*2] += alpha * x2[2];
		pD2[3+bs*3] += alpha * x2[3];
		pD2[4+bs*4] += alpha * x2[4];
		pD2[5+bs*5] += alpha * x2[5];
		pD2[6+bs*6] += alpha * x2[6];
		pD2[7+bs*7] += alpha * x2[7];
		pD2 += bs*sdd+bs*bs;
		x2 += bs;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[jj*sdd+(jj+ll)*bs+ll] += alpha * x[jj+ll];
		}
	return;
	}



// insert vector to diagonal, sparse formulation
void sdiain_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs+ii*bs] = alpha * x[jj];
		}
	return;
	}



// extract diagonal to vector, sparse formulation
void sdiaex_libsp(int kmax, int *idx, float alpha, float *pD, int sdd, float *x)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		x[jj] = alpha * pD[ii/bs*bs*sdd+ii%bs+ii*bs];
		}
	return;
	}



// add scaled vector to diagonal, sparse formulation
void sdiaad_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs+ii*bs] += alpha * x[jj];
		}
	return;
	}



// add scaled vector to another vector and insert to diagonal, sparse formulation
void sdiaadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD, int sdd)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs+ii*bs] = y[jj] + alpha * x[jj];
		}
	return;
	}



// insert vector to row
void srowin_lib(int kmax, float alpha, float *x, float *pD)
	{

	const int bs = 8;

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[0*bs] = alpha * x[0];
		pD[1*bs] = alpha * x[1];
		pD[2*bs] = alpha * x[2];
		pD[3*bs] = alpha * x[3];
		pD += 4*bs;
		x += 4;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[ll*bs] = alpha*x[ll];
		}
	return;
	}



// extract row to vector
void srowex_lib(int kmax, float alpha, float *pD, float *x)
	{

	const int bs = 8;

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		x[0] = alpha * pD[0*bs];
		x[1] = alpha * pD[1*bs];
		x[2] = alpha * pD[2*bs];
		x[3] = alpha * pD[3*bs];
		pD += 4*bs;
		x += 4;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		x[ll] = alpha*pD[ll*bs];
		}
	return;
	}



// add scaled vector to row
void srowad_lib(int kmax, float alpha, float *x, float *pD)
	{

	const int bs = 8;

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pD[0*bs] += alpha * x[0];
		pD[1*bs] += alpha * x[1];
		pD[2*bs] += alpha * x[2];
		pD[3*bs] += alpha * x[3];
		pD += 4*bs;
		x += 4;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[ll*bs] += alpha * x[ll];
		}
	return;
	}



// insert vector to row, sparse formulation
void srowin_libsp(int kmax, float alpha, int *idx, float *x, float *pD)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*bs] = alpha*x[jj];
		}
	return;
	}



// add scaled vector to row, sparse formulation
void srowad_libsp(int kmax, int *idx, float alpha, float *x, float *pD)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*bs] += alpha * x[jj];
		}
	return;
	}



// add scaled vector to another vector and insert to row, sparse formulation
void srowadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii*bs] = y[jj] + alpha * x[jj];
		}
	return;
	}



// swap two rows
void srowsw_lib(int kmax, float *pA, float *pC)
	{

	const int bs = 8;

	int ii;
	float tmp;

	for(ii=0; ii<kmax-3; ii+=4)
		{
		tmp = pA[0+bs*0];
		pA[0+bs*0] = pC[0+bs*0];
		pC[0+bs*0] = tmp;
		tmp = pA[0+bs*1];
		pA[0+bs*1] = pC[0+bs*1];
		pC[0+bs*1] = tmp;
		tmp = pA[0+bs*2];
		pA[0+bs*2] = pC[0+bs*2];
		pC[0+bs*2] = tmp;
		tmp = pA[0+bs*3];
		pA[0+bs*3] = pC[0+bs*3];
		pC[0+bs*3] = tmp;
		pA += 4*bs;
		pC += 4*bs;
		}
	for( ; ii<kmax; ii++)
		{
		tmp = pA[0+bs*0];
		pA[0+bs*0] = pC[0+bs*0];
		pC[0+bs*0] = tmp;
		pA += 1*bs;
		pC += 1*bs;
		}
	return;
	}



// insert vector to column
void scolin_lib(int kmax, float *x, int offset, float *pD, int sdd)
	{

	const int bs = 8;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll] = x[ll];
			}
		pD += kna + bs*(sdd-1);
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-7; jj+=8)
		{
		pD[0] = x[0];
		pD[1] = x[1];
		pD[2] = x[2];
		pD[3] = x[3];
		pD[4] = x[4];
		pD[5] = x[5];
		pD[6] = x[6];
		pD[7] = x[7];
		pD += bs*sdd;
		x += bs;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[ll] = x[ll];
		}

	}



// add scaled vector to column
void scolad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd)
	{

	const int bs = 8;

	int kna = (bs-offset%bs)%bs;
	kna = kmax<kna ? kmax : kna;

	int jj, ll;

	if(kna>0)
		{
		for(ll=0; ll<kna; ll++)
			{
			pD[ll] += alpha * x[ll];
			}
		pD += kna + bs*(sdd-1);
		x  += kna;
		kmax -= kna;
		}
	for(jj=0; jj<kmax-7; jj+=8)
		{
		pD[0] += alpha * x[0];
		pD[1] += alpha * x[1];
		pD[2] += alpha * x[2];
		pD[3] += alpha * x[3];
		pD[4] += alpha * x[4];
		pD[5] += alpha * x[5];
		pD[6] += alpha * x[6];
		pD[7] += alpha * x[7];
		pD += bs*sdd;
		x += bs;
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pD[ll] += alpha * x[ll];
		}

	}



// insert vector to diagonal, sparse formulation
void scolin_libsp(int kmax, int *idx, float *x, float *pD, int sdd)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs] = x[jj];
		}

	}



// add scaled vector to diagonal, sparse formulation
void scolad_libsp(int kmax, float alpha, int *idx, float *x, float *pD, int sdd)
	{

	const int bs = 8;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[ii/bs*bs*sdd+ii%bs] += alpha * x[jj];
		}

	}



// swaps two cols
void scolsw_lib(int kmax, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc)
	{

	const int bs = 8;

	int ii;

	float tmp;

	if(offsetA==offsetC)
		{
		if(offsetA>0)
			{
			ii = 0;
			for(; ii<bs-offsetA; ii++)
				{
				tmp = pA[0+bs*0];
				pA[0+bs*0] = pC[0+bs*0];
				pC[0+bs*0] = tmp;
				pA += 1;
				pC += 1;
				}
			pA += bs*(sda-1);
			pC += bs*(sdc-1);
			kmax -= bs-offsetA;
			}
		ii = 0;
		for(; ii<kmax-7; ii+=8)
			{
			tmp = pA[0+bs*0];
			pA[0+bs*0] = pC[0+bs*0];
			pC[0+bs*0] = tmp;
			tmp = pA[1+bs*0];
			pA[1+bs*0] = pC[1+bs*0];
			pC[1+bs*0] = tmp;
			tmp = pA[2+bs*0];
			pA[2+bs*0] = pC[2+bs*0];
			pC[2+bs*0] = tmp;
			tmp = pA[3+bs*0];
			pA[3+bs*0] = pC[3+bs*0];
			pC[3+bs*0] = tmp;
			tmp = pA[4+bs*0];
			pA[4+bs*0] = pC[4+bs*0];
			pC[4+bs*0] = tmp;
			tmp = pA[5+bs*0];
			pA[5+bs*0] = pC[5+bs*0];
			pC[5+bs*0] = tmp;
			tmp = pA[6+bs*0];
			pA[6+bs*0] = pC[6+bs*0];
			pC[6+bs*0] = tmp;
			tmp = pA[7+bs*0];
			pA[7+bs*0] = pC[7+bs*0];
			pC[7+bs*0] = tmp;
			pA += bs*sda;
			pC += bs*sdc;
			}
		for(; ii<kmax; ii++)
			{
			tmp = pA[0+bs*0];
			pA[0+bs*0] = pC[0+bs*0];
			pC[0+bs*0] = tmp;
			pA += 1;
			pC += 1;
			}
		}
	else
		{
		printf("\nscolsw: feature not implemented yet: offsetA!=offsetC\n\n");
		exit(1);
		}

	return;

	}



// insert vector to vector, sparse formulation
void svecin_libsp(int kmax, int *idx, float *x, float *y)
	{

	int jj;

	for(jj=0; jj<kmax; jj++)
		{
		y[idx[jj]] = x[jj];
		}

	}



// adds vector to vector, sparse formulation
void svecad_libsp(int kmax, int *idx, float alpha, float *x, float *y)
	{

	int jj;

	for(jj=0; jj<kmax; jj++)
		{
		y[idx[jj]] += alpha * x[jj];
		}

	}



/****************************
* new interface
****************************/



#if defined(LA_HIGH_PERFORMANCE)



// return the memory size (in bytes) needed for a strmat
int s_size_strmat(int m, int n)
	{
	const int bs = 8;
	int nc = S_NC;
	int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	int memory_size = (pm*cn+tmp)*sizeof(float);
	return memory_size;
	}



// return the memory size (in bytes) needed for the digonal of a strmat
int s_size_diag_strmat(int m, int n)
	{
	const int bs = 8;
	int nc = S_NC;
	int al = bs*nc;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	int memory_size = tmp*sizeof(float);
	return memory_size;
	}



// create a matrix structure for a matrix of size m*n by using memory passed by a pointer
void s_create_strmat(int m, int n, struct s_strmat *sA, void *memory)
	{
	const int bs = 8;
	int nc = S_NC;
	int al = bs*nc;
	sA->m = m;
	sA->n = n;
	int pm = (m+bs-1)/bs*bs;
	int cn = (n+nc-1)/nc*nc;
	sA->pm = pm;
	sA->cn = cn;
	float *ptr = (float *) memory;
	sA->pA = ptr;
	ptr += pm*cn;
	int tmp = m<n ? (m+al-1)/al*al : (n+al-1)/al*al; // al(min(m,n)) // XXX max ???
	sA->dA = ptr;
	ptr += tmp;
	sA->use_dA = 0;
	sA->memory_size = (pm*cn+tmp)*sizeof(float);
	return;
	}



// return memory size (in bytes) needed for a strvec
int s_size_strvec(int m)
	{
	const int bs = 8;
//	int nc = S_NC;
//	int al = bs*nc;
	int pm = (m+bs-1)/bs*bs;
	int memory_size = pm*sizeof(float);
	return memory_size;
	}



// create a vector structure for a vector of size m by using memory passed by a pointer
void s_create_strvec(int m, struct s_strvec *sa, void *memory)
	{
	const int bs = 8;
//	int nc = S_NC;
//	int al = bs*nc;
	sa->m = m;
	int pm = (m+bs-1)/bs*bs;
	sa->pm = pm;
	float *ptr = (float *) memory;
	sa->pa = ptr;
//	ptr += pm;
	sa->memory_size = pm*sizeof(float);
	return;
	}



// convert a matrix into a matrix structure
void s_cvt_mat2strmat(int m, int n, float *A, int lda, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	int i, ii, j, jj, m0, m1, m2;
	float *B, *pB;
	m0 = (bs-ai%bs)%bs;
	if(m0>m)
		m0 = m;
	m1 = m - m0;
	jj = 0;
	for( ; jj<n-3; jj+=4)
		{
		B  =  A + jj*lda;
		pB = pA + jj*bs;
		ii = 0;
		if(m0>0)
			{
			for( ; ii<m0; ii++)
				{
				pB[ii+bs*0] = B[ii+lda*0];
				pB[ii+bs*1] = B[ii+lda*1];
				pB[ii+bs*2] = B[ii+lda*2];
				pB[ii+bs*3] = B[ii+lda*3];
				}
			B  += m0;
			pB += m0 + bs*(sda-1);
			}
		for( ; ii<m-7; ii+=8)
			{
			// unroll 0
			pB[0+bs*0] = B[0+lda*0];
			pB[1+bs*0] = B[1+lda*0];
			pB[2+bs*0] = B[2+lda*0];
			pB[3+bs*0] = B[3+lda*0];
			pB[4+bs*0] = B[4+lda*0];
			pB[5+bs*0] = B[5+lda*0];
			pB[6+bs*0] = B[6+lda*0];
			pB[7+bs*0] = B[7+lda*0];
			// unroll 1
			pB[0+bs*1] = B[0+lda*1];
			pB[1+bs*1] = B[1+lda*1];
			pB[2+bs*1] = B[2+lda*1];
			pB[3+bs*1] = B[3+lda*1];
			pB[4+bs*1] = B[4+lda*1];
			pB[5+bs*1] = B[5+lda*1];
			pB[6+bs*1] = B[6+lda*1];
			pB[7+bs*1] = B[7+lda*1];
			// unroll 2
			pB[0+bs*2] = B[0+lda*2];
			pB[1+bs*2] = B[1+lda*2];
			pB[2+bs*2] = B[2+lda*2];
			pB[3+bs*2] = B[3+lda*2];
			pB[4+bs*2] = B[4+lda*2];
			pB[5+bs*2] = B[5+lda*2];
			pB[6+bs*2] = B[6+lda*2];
			pB[7+bs*2] = B[7+lda*2];
			// unroll 3
			pB[0+bs*3] = B[0+lda*3];
			pB[1+bs*3] = B[1+lda*3];
			pB[2+bs*3] = B[2+lda*3];
			pB[3+bs*3] = B[3+lda*3];
			pB[4+bs*3] = B[4+lda*3];
			pB[5+bs*3] = B[5+lda*3];
			pB[6+bs*3] = B[6+lda*3];
			pB[7+bs*3] = B[7+lda*3];
			// update
			B  += 8;
			pB += bs*sda;
			}
		for( ; ii<m; ii++)
			{
			// col 0
			pB[0+bs*0] = B[0+lda*0];
			// col 1
			pB[0+bs*1] = B[0+lda*1];
			// col 2
			pB[0+bs*2] = B[0+lda*2];
			// col 3
			pB[0+bs*3] = B[0+lda*3];
			// update
			B  += 1;
			pB += 1;
			}
		}
	for( ; jj<n; jj++)
		{

		B  =  A + jj*lda;
		pB = pA + jj*bs;

		ii = 0;
		if(m0>0)
			{
			for( ; ii<m0; ii++)
				{
				pB[ii+bs*0] = B[ii+lda*0];
				}
			B  += m0;
			pB += m0 + bs*(sda-1);
			}
		for( ; ii<m-7; ii+=8)
			{
			// col 0
			pB[0+bs*0] = B[0+lda*0];
			pB[1+bs*0] = B[1+lda*0];
			pB[2+bs*0] = B[2+lda*0];
			pB[3+bs*0] = B[3+lda*0];
			pB[4+bs*0] = B[4+lda*0];
			pB[5+bs*0] = B[5+lda*0];
			pB[6+bs*0] = B[6+lda*0];
			pB[7+bs*0] = B[7+lda*0];
			// update
			B  += 8;
			pB += bs*sda;
			}
		for( ; ii<m; ii++)
			{
			// col 0
			pB[0+bs*0] = B[0+lda*0];
			// update
			B  += 1;
			pB += 1;
			}
		}
	return;
	}



// convert and transpose a matrix into a matrix structure
void s_cvt_tran_mat2strmat(int m, int n, float *A, int lda, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	int i, ii, j, m0, m1, m2;
	float 	*B, *pB;
	m0 = (bs-ai%bs)%bs;
	if(m0>n)
		m0 = n;
	m1 = n - m0;
	ii = 0;
	if(m0>0)
		{
		for(j=0; j<m; j++)
			{
			for(i=0; i<m0; i++)
				{
				pA[i+j*bs+ii*sda] = A[j+(i+ii)*lda];
				}
			}
		A  += m0*lda;
		pA += m0 + bs*(sda-1);
		}
	ii = 0;
	for(; ii<m1-7; ii+=bs)
		{
		j=0;
		B  = A + ii*lda;
		pB = pA + ii*sda;
		for(; j<m-3; j+=4)
			{
			// unroll 0
			pB[0+0*bs] = B[0+0*lda];
			pB[1+0*bs] = B[0+1*lda];
			pB[2+0*bs] = B[0+2*lda];
			pB[3+0*bs] = B[0+3*lda];
			pB[4+0*bs] = B[0+4*lda];
			pB[5+0*bs] = B[0+5*lda];
			pB[6+0*bs] = B[0+6*lda];
			pB[7+0*bs] = B[0+7*lda];
			// unroll 1
			pB[0+1*bs] = B[1+0*lda];
			pB[1+1*bs] = B[1+1*lda];
			pB[2+1*bs] = B[1+2*lda];
			pB[3+1*bs] = B[1+3*lda];
			pB[4+1*bs] = B[1+4*lda];
			pB[5+1*bs] = B[1+5*lda];
			pB[6+1*bs] = B[1+6*lda];
			pB[7+1*bs] = B[1+7*lda];
			// unroll 2
			pB[0+2*bs] = B[2+0*lda];
			pB[1+2*bs] = B[2+1*lda];
			pB[2+2*bs] = B[2+2*lda];
			pB[3+2*bs] = B[2+3*lda];
			pB[4+2*bs] = B[2+4*lda];
			pB[5+2*bs] = B[2+5*lda];
			pB[6+2*bs] = B[2+6*lda];
			pB[7+2*bs] = B[2+7*lda];
			// unroll 3
			pB[0+3*bs] = B[3+0*lda];
			pB[1+3*bs] = B[3+1*lda];
			pB[2+3*bs] = B[3+2*lda];
			pB[3+3*bs] = B[3+3*lda];
			pB[4+3*bs] = B[3+4*lda];
			pB[5+3*bs] = B[3+5*lda];
			pB[6+3*bs] = B[3+6*lda];
			pB[7+3*bs] = B[3+7*lda];
			B  += 4;
			pB += 4*bs;
			}
		for(; j<m; j++)
			{
			// unroll 0
			pB[0+0*bs] = B[0+0*lda];
			pB[1+0*bs] = B[0+1*lda];
			pB[2+0*bs] = B[0+2*lda];
			pB[3+0*bs] = B[0+3*lda];
			pB[4+0*bs] = B[0+4*lda];
			pB[5+0*bs] = B[0+5*lda];
			pB[6+0*bs] = B[0+6*lda];
			pB[7+0*bs] = B[0+7*lda];
			B  += 1;
			pB += 1*bs;
			}
		}
	if(ii<m1)
		{
		m2 = m1-ii;
		if(bs<m2) m2 = bs;
		for(j=0; j<m; j++)
			{
			for(i=0; i<m2; i++)
				{
				pA[i+j*bs+ii*sda] = A[j+(i+ii)*lda];
				}
			}
		}
	return;
	}



// convert a vector into a vector structure
void s_cvt_vec2strvec(int m, float *a, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	int ii;
	for(ii=0; ii<m; ii++)
		pa[ii] = a[ii];
	return;
	}



// convert a matrix structure into a matrix
void s_cvt_strmat2mat(int m, int n, struct s_strmat *sA, int ai, int aj, float *A, int lda)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	int i, ii, jj;
	int m0 = (bs-ai%bs)%bs;
	float *ptr_pA;
	jj=0;
	for(; jj<n-3; jj+=4)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(m0>0)
			{
			for(; ii<m0; ii++)
				{
				// unroll 0
				A[ii+lda*(jj+0)] = ptr_pA[0+bs*0];
				// unroll 1
				A[ii+lda*(jj+1)] = ptr_pA[0+bs*1];
				// unroll 2
				A[ii+lda*(jj+2)] = ptr_pA[0+bs*2];
				// unroll 3
				A[ii+lda*(jj+3)] = ptr_pA[0+bs*3];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		// TODO update A !!!!!
		for(; ii<m-bs+1; ii+=bs)
			{
			// unroll 0
			A[0+ii+lda*(jj+0)] = ptr_pA[0+bs*0];
			A[1+ii+lda*(jj+0)] = ptr_pA[1+bs*0];
			A[2+ii+lda*(jj+0)] = ptr_pA[2+bs*0];
			A[3+ii+lda*(jj+0)] = ptr_pA[3+bs*0];
			A[4+ii+lda*(jj+0)] = ptr_pA[4+bs*0];
			A[5+ii+lda*(jj+0)] = ptr_pA[5+bs*0];
			A[6+ii+lda*(jj+0)] = ptr_pA[6+bs*0];
			A[7+ii+lda*(jj+0)] = ptr_pA[7+bs*0];
			// unroll 0
			A[0+ii+lda*(jj+1)] = ptr_pA[0+bs*1];
			A[1+ii+lda*(jj+1)] = ptr_pA[1+bs*1];
			A[2+ii+lda*(jj+1)] = ptr_pA[2+bs*1];
			A[3+ii+lda*(jj+1)] = ptr_pA[3+bs*1];
			A[4+ii+lda*(jj+1)] = ptr_pA[4+bs*1];
			A[5+ii+lda*(jj+1)] = ptr_pA[5+bs*1];
			A[6+ii+lda*(jj+1)] = ptr_pA[6+bs*1];
			A[7+ii+lda*(jj+1)] = ptr_pA[7+bs*1];
			// unroll 0
			A[0+ii+lda*(jj+2)] = ptr_pA[0+bs*2];
			A[1+ii+lda*(jj+2)] = ptr_pA[1+bs*2];
			A[2+ii+lda*(jj+2)] = ptr_pA[2+bs*2];
			A[3+ii+lda*(jj+2)] = ptr_pA[3+bs*2];
			A[4+ii+lda*(jj+2)] = ptr_pA[4+bs*2];
			A[5+ii+lda*(jj+2)] = ptr_pA[5+bs*2];
			A[6+ii+lda*(jj+2)] = ptr_pA[6+bs*2];
			A[7+ii+lda*(jj+2)] = ptr_pA[7+bs*2];
			// unroll 0
			A[0+ii+lda*(jj+3)] = ptr_pA[0+bs*3];
			A[1+ii+lda*(jj+3)] = ptr_pA[1+bs*3];
			A[2+ii+lda*(jj+3)] = ptr_pA[2+bs*3];
			A[3+ii+lda*(jj+3)] = ptr_pA[3+bs*3];
			A[4+ii+lda*(jj+3)] = ptr_pA[4+bs*3];
			A[5+ii+lda*(jj+3)] = ptr_pA[5+bs*3];
			A[6+ii+lda*(jj+3)] = ptr_pA[6+bs*3];
			A[7+ii+lda*(jj+3)] = ptr_pA[7+bs*3];
			ptr_pA += sda*bs;
			}
		for(; ii<m; ii++)
			{
			// unroll 0
			A[ii+lda*(jj+0)] = ptr_pA[0+bs*0];
			// unroll 1
			A[ii+lda*(jj+1)] = ptr_pA[0+bs*1];
			// unroll 2
			A[ii+lda*(jj+2)] = ptr_pA[0+bs*2];
			// unroll 3
			A[ii+lda*(jj+3)] = ptr_pA[0+bs*3];
			ptr_pA++;
			}
		}
	for(; jj<n; jj++)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(m0>0)
			{
			for(; ii<m0; ii++)
				{
				A[ii+lda*jj] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<m-bs+1; ii+=bs)
			{
			A[0+ii+lda*(jj+0)] = ptr_pA[0+bs*0];
			A[1+ii+lda*(jj+0)] = ptr_pA[1+bs*0];
			A[2+ii+lda*(jj+0)] = ptr_pA[2+bs*0];
			A[3+ii+lda*(jj+0)] = ptr_pA[3+bs*0];
			A[4+ii+lda*(jj+0)] = ptr_pA[4+bs*0];
			A[5+ii+lda*(jj+0)] = ptr_pA[5+bs*0];
			A[6+ii+lda*(jj+0)] = ptr_pA[6+bs*0];
			A[7+ii+lda*(jj+0)] = ptr_pA[7+bs*0];
			ptr_pA += sda*bs;
			}
		for(; ii<m; ii++)
			{
			A[ii+lda*jj] = ptr_pA[0];
			ptr_pA++;
			}
		}
	return;
	}



// convert and transpose a matrix structure into a matrix
void s_cvt_tran_strmat2mat(int m, int n, struct s_strmat *sA, int ai, int aj, float *A, int lda)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + aj*bs + ai/bs*bs*sda + ai%bs;
	int i, ii, jj;
	int m0 = (bs-ai%bs)%bs;
	float *ptr_pA;
	jj=0;
	for(; jj<n-3; jj+=4)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(m0>0)
			{
			for(; ii<m0; ii++)
				{
				// unroll 0
				A[jj+0+lda*ii] = ptr_pA[0+bs*0];
				// unroll 1
				A[jj+1+lda*ii] = ptr_pA[0+bs*1];
				// unroll 2
				A[jj+2+lda*ii] = ptr_pA[0+bs*2];
				// unroll 3
				A[jj+3+lda*ii] = ptr_pA[0+bs*3];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		// TODO update A !!!!!
		for(; ii<m-bs+1; ii+=bs)
			{
			// unroll 0
			A[jj+0+lda*(ii+0)] = ptr_pA[0+bs*0];
			A[jj+0+lda*(ii+1)] = ptr_pA[1+bs*0];
			A[jj+0+lda*(ii+2)] = ptr_pA[2+bs*0];
			A[jj+0+lda*(ii+3)] = ptr_pA[3+bs*0];
			A[jj+0+lda*(ii+4)] = ptr_pA[4+bs*0];
			A[jj+0+lda*(ii+5)] = ptr_pA[5+bs*0];
			A[jj+0+lda*(ii+6)] = ptr_pA[6+bs*0];
			A[jj+0+lda*(ii+7)] = ptr_pA[7+bs*0];
			// unroll 1
			A[jj+1+lda*(ii+0)] = ptr_pA[0+bs*1];
			A[jj+1+lda*(ii+1)] = ptr_pA[1+bs*1];
			A[jj+1+lda*(ii+2)] = ptr_pA[2+bs*1];
			A[jj+1+lda*(ii+3)] = ptr_pA[3+bs*1];
			A[jj+1+lda*(ii+4)] = ptr_pA[4+bs*1];
			A[jj+1+lda*(ii+5)] = ptr_pA[5+bs*1];
			A[jj+1+lda*(ii+6)] = ptr_pA[6+bs*1];
			A[jj+1+lda*(ii+7)] = ptr_pA[7+bs*1];
			// unroll 2
			A[jj+2+lda*(ii+0)] = ptr_pA[0+bs*2];
			A[jj+2+lda*(ii+1)] = ptr_pA[1+bs*2];
			A[jj+2+lda*(ii+2)] = ptr_pA[2+bs*2];
			A[jj+2+lda*(ii+3)] = ptr_pA[3+bs*2];
			A[jj+2+lda*(ii+4)] = ptr_pA[4+bs*2];
			A[jj+2+lda*(ii+5)] = ptr_pA[5+bs*2];
			A[jj+2+lda*(ii+6)] = ptr_pA[6+bs*2];
			A[jj+2+lda*(ii+7)] = ptr_pA[7+bs*2];
			// unroll 3
			A[jj+3+lda*(ii+0)] = ptr_pA[0+bs*3];
			A[jj+3+lda*(ii+1)] = ptr_pA[1+bs*3];
			A[jj+3+lda*(ii+2)] = ptr_pA[2+bs*3];
			A[jj+3+lda*(ii+3)] = ptr_pA[3+bs*3];
			A[jj+3+lda*(ii+4)] = ptr_pA[4+bs*3];
			A[jj+3+lda*(ii+5)] = ptr_pA[5+bs*3];
			A[jj+3+lda*(ii+6)] = ptr_pA[6+bs*3];
			A[jj+3+lda*(ii+7)] = ptr_pA[7+bs*3];
			ptr_pA += sda*bs;
			}
		for(; ii<m; ii++)
			{
			// unroll 0
			A[jj+0+lda*ii] = ptr_pA[0+bs*0];
			// unroll 1
			A[jj+1+lda*ii] = ptr_pA[0+bs*1];
			// unroll 2
			A[jj+2+lda*ii] = ptr_pA[0+bs*2];
			// unroll 3
			A[jj+3+lda*ii] = ptr_pA[0+bs*3];
			ptr_pA++;
			}
		}
	for(; jj<n; jj++)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(m0>0)
			{
			for(; ii<m0; ii++)
				{
				A[jj+lda*ii] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<m-bs+1; ii+=bs)
			{
			i=0;
			// TODO update A !!!!!
			// TODO unroll !!!!!!
			for(; i<bs; i++)
				{
				A[jj+lda*(i+ii)] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<m; ii++)
			{
			A[jj+lda*ii] = ptr_pA[0];
			ptr_pA++;
			}
		}
	return;
	}



// convert a vector structure into a vector
void s_cvt_strvec2vec(int m, struct s_strvec *sa, int ai, float *a)
	{
	float *pa = sa->pa + ai;
	int ii;
	for(ii=0; ii<m; ii++)
		a[ii] = pa[ii];
	return;
	}



// cast a matrix into a matrix structure
void s_cast_mat2strmat(float *A, struct s_strmat *sA)
	{
	sA->pA = A;
	return;
	}



// cast a matrix into the diagonal of a matrix structure
void s_cast_diag_mat2strmat(float *dA, struct s_strmat *sA)
	{
	sA->dA = dA;
	return;
	}



// cast a vector into a vector structure
void s_cast_vec2vecmat(float *a, struct s_strvec *sa)
	{
	sa->pa = a;
	return;
	}



// insert element into strmat
void sgein1_libstr(float a, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	pA[0] = a;
	return;
	}



// extract element from strmat
float sgeex1_libstr(struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	return pA[0];
	}



// insert element into strvec
void svecin1_libstr(float a, struct s_strvec *sx, int xi)
	{
	const int bs = 8;
	float *x = sx->pa + xi;
	x[0] = a;
	return;
	}



// extract element from strvec
float svecex1_libstr(struct s_strvec *sx, int xi)
	{
	const int bs = 8;
	float *x = sx->pa + xi;
	return x[0];
	}



// set all elements of a strmat to a value
void sgese_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai%bs + ai/bs*bs*sda + aj*bs;
	int m0 = m<(bs-ai%bs)%bs ? m : (bs-ai%bs)%bs;
	int ii, jj;
	if(m0>0)
		{
		for(ii=0; ii<m0; ii++)
			{
			for(jj=0; jj<n; jj++)
				{
				pA[jj*bs] = alpha;
				}
			pA += 1;
			}
		pA += bs*(sda-1);
		m -= m0;
		}
	for(ii=0; ii<m-7; ii+=8)
		{
		for(jj=0; jj<n; jj++)
			{
			pA[0+jj*bs] = alpha;
			pA[1+jj*bs] = alpha;
			pA[2+jj*bs] = alpha;
			pA[3+jj*bs] = alpha;
			pA[4+jj*bs] = alpha;
			pA[5+jj*bs] = alpha;
			pA[6+jj*bs] = alpha;
			pA[7+jj*bs] = alpha;
			}
		pA += bs*sda;
		}
	for( ; ii<m; ii++)
		{
		for(jj=0; jj<n; jj++)
			{
			pA[jj*bs] = alpha;
			}
		pA += 1;
		}
	return;
	}



// set all elements of a strvec to a value
void svecse_libstr(int m, float alpha, struct s_strvec *sx, int xi)
	{
	float *x = sx->pa + xi;
	int ii;
	for(ii=0; ii<m; ii++)
		x[ii] = alpha;
	return;
	}



// extract diagonal to vector
void sdiaex_libstr(int kmax, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	float *x = sx->pa + xi;
	sdiaex_lib(kmax, alpha, ai%bs, pA, sda, x);
	return;
	}



// insert a vector into diagonal
void sdiain_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	float *x = sx->pa + xi;
	sdiain_lib(kmax, alpha, x, ai%bs, pA, sda);
	return;
	}



// swap two rows of a matrix struct
void srowsw_libstr(int kmax, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	int sdc = sC->cn;
	float *pC = sC->pA + ci/bs*bs*sdc + ci%bs + cj*bs;
	srowsw_lib(kmax, pA, pC);
	return;
	}



// permute the rows of a matrix struct
void srowpe_libstr(int kmax, int *ipiv, struct s_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			srowsw_libstr(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}


// inverse permute the rows of a matrix struct
void srowpei_libstr(int kmax, int *ipiv, struct s_strmat *sA)
	{
	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			srowsw_libstr(sA->n, sA, ii, 0, sA, ipiv[ii], 0);
		}
	return;
	}


// extract a row int a vector
void srowex_libstr(int kmax, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strvec *sx, int xi)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	float *x = sx->pa + xi;
	srowex_lib(kmax, alpha, pA, x);
	return;
	}



// insert a vector into a row
void srowin_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	float *x = sx->pa + xi;
	srowin_lib(kmax, alpha, x, pA);
	return;
	}



// add a vector to a row
void srowad_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strmat *sA, int ai, int aj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	float *x = sx->pa + xi;
	srowad_lib(kmax, alpha, x, pA);
	return;
	}



// swap two cols of a matrix struct
void scolsw_libstr(int kmax, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	int sdc = sC->cn;
	float *pC = sC->pA + ci/bs*bs*sdc + ci%bs + cj*bs;
	scolsw_lib(kmax, ai%bs, pA, sda, ci%bs, pC, sdc);
	return;
	}



// permute the cols of a matrix struct
void scolpe_libstr(int kmax, int *ipiv, struct s_strmat *sA)
	{
	int ii;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			scolsw_libstr(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// inverse permute the cols of a matrix struct
void scolpei_libstr(int kmax, int *ipiv, struct s_strmat *sA)
	{
	int ii;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			scolsw_libstr(sA->m, sA, 0, ii, sA, 0, ipiv[ii]);
		}
	return;
	}



// scale a generic strmat
void sgesc_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj)
	{

	// early return
	if(m==0 | n==0)
		return;

#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** sgesc_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** sgesc_libstr : n<0 : %d<0 *****\n", n);
	// non-negative offset
	if(ai<0) printf("\n****** sgesc_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** sgesc_libstr : aj<0 : %d<0 *****\n", aj);
	// inside matrix
	// A: m x n
	if(ai+m > sA->m) printf("\n***** sgesc_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+n > sA->n) printf("\n***** sgesc_libstr : aj+n > col(A) : %d+%d > %d *****\n", aj, n, sA->n);
#endif

	const int bs = 8;

	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + aj*bs;
	int offsetA = ai%bs;

	int ii, mna;

	// alignment
	if(offsetA>0)
		{
		mna = bs-offsetA;
		mna = m<mna ? m : mna;
		kernel_sgesc_8_0_gen_u_lib8(n, &alpha, &pA[offsetA], mna);
		m -= mna;
		pA += 8*sda;
		}

	ii = 0;
	// main loop
	for( ; ii<m-7; ii+=8)
		{
		kernel_sgesc_8_0_lib8(n, &alpha, &pA[0]);
		pA += 8*sda;
		}
	// clean up
	if(ii<m)
		{
		kernel_sgesc_8_0_gen_lib8(n, &alpha, &pA[0], m-ii);
		}

	return;

	}



// copy a generic strmat into a generic strmat
void sgecp_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj)
	{

	// early return
	if(m==0 | n==0)
		return;

#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** sgecp_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** sgecp_libstr : n<0 : %d<0 *****\n", n);
	// non-negative offset
	if(ai<0) printf("\n****** sgecp_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** sgecp_libstr : aj<0 : %d<0 *****\n", aj);
	if(bi<0) printf("\n****** sgecp_libstr : bi<0 : %d<0 *****\n", bi);
	if(bj<0) printf("\n****** sgecp_libstr : bj<0 : %d<0 *****\n", bj);
	// inside matrix
	// A: m x n
	if(ai+m > sA->m) printf("\n***** sgecp_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+n > sA->n) printf("\n***** sgecp_libstr : aj+n > col(A) : %d+%d > %d *****\n", aj, n, sA->n);
	// B: m x n
	if(bi+m > sB->m) printf("\n***** sgecp_libstr : bi+m > row(B) : %d+%d > %d *****\n", bi, m, sB->m);
	if(bj+n > sB->n) printf("\n***** sgecp_libstr : bj+n > col(B) : %d+%d > %d *****\n", bj, n, sB->n);
#endif

	const int bs = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	float *pA = sA->pA + ai/bs*bs*sda + aj*bs;
	float *pB = sB->pA + bi/bs*bs*sdb + bj*bs;
	int offsetA = ai%bs;
	int offsetB = bi%bs;

	int ii, mna;

	if(offsetB>0)
		{
		if(offsetB>offsetA)
			{
			mna = bs-offsetB;
			mna = m<mna ? m : mna;
			kernel_sgecp_8_0_gen_u_lib8(n, &pA[offsetA], &pB[offsetB], mna);
			m -= mna;
			pB += 8*sdb;
			}
		else
			{
			if(offsetA==0)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_0_gen_lib8(n, &pA[0], &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==1)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_1_gen_lib8(n, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==2)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_2_gen_lib8(n, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==3)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_3_gen_lib8(n, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==4)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_4_gen_lib8(n, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==5)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_5_gen_lib8(n, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==6)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_6_gen_lib8(n, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==7)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecp_8_7_gen_lib8(n, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			}
		}

	// same alignment
	if(offsetA==offsetB)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_0_lib8(n, pA, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_0_gen_lib8(n, pA, pB, m-ii);
			}
		return;
		}
	// XXX different alignment: search tree ???
	// skip one element of A
	else if(offsetA==(offsetB+1)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_1_lib8(n, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_1_gen_lib8(n, pA, sda, pB, m-ii);
			}
		}
	// skip two elements of A
	else if(offsetA==(offsetB+2)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_2_lib8(n, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_2_gen_lib8(n, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip three elements of A
	else if(offsetA==(offsetB+3)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_3_lib8(n, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_3_gen_lib8(n, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip four elements of A
	else if(offsetA==(offsetB+4)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_4_lib8(n, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_4_gen_lib8(n, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip five elements of A
	else if(offsetA==(offsetB+5)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_5_lib8(n, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_5_gen_lib8(n, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip six elements of A
	else if(offsetA==(offsetB+6)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_6_lib8(n, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_6_gen_lib8(n, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip seven elements of A
	else //if(offsetA==(offsetB+7)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecp_8_7_lib8(n, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecp_8_7_gen_lib8(n, pA, sda, pB, m-ii);
			}
		return;
		}

	return;

	}



// copy and scale a generic strmat into a generic strmat
void sgecpsc_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj)
	{

	// early return
	if(m==0 | n==0)
		return;

#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** sgecpsc_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** sgecpsc_libstr : n<0 : %d<0 *****\n", n);
	// non-negative offset
	if(ai<0) printf("\n****** sgecpsc_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** sgecpsc_libstr : aj<0 : %d<0 *****\n", aj);
	if(bi<0) printf("\n****** sgecpsc_libstr : bi<0 : %d<0 *****\n", bi);
	if(bj<0) printf("\n****** sgecpsc_libstr : bj<0 : %d<0 *****\n", bj);
	// inside matrix
	// A: m x n
	if(ai+m > sA->m) printf("\n***** sgecpsc_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+n > sA->n) printf("\n***** sgecpsc_libstr : aj+n > col(A) : %d+%d > %d *****\n", aj, n, sA->n);
	// B: m x n
	if(bi+m > sB->m) printf("\n***** sgecpsc_libstr : bi+m > row(B) : %d+%d > %d *****\n", bi, m, sB->m);
	if(bj+n > sB->n) printf("\n***** sgecpsc_libstr : bj+n > col(B) : %d+%d > %d *****\n", bj, n, sB->n);
#endif

	const int bs = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	float *pA = sA->pA + ai/bs*bs*sda + aj*bs;
	float *pB = sB->pA + bi/bs*bs*sdb + bj*bs;
	int offsetA = ai%bs;
	int offsetB = bi%bs;

	int ii, mna;

	if(offsetB>0)
		{
		if(offsetB>offsetA)
			{
			mna = bs-offsetB;
			mna = m<mna ? m : mna;
			kernel_sgecpsc_8_0_gen_u_lib8(n, &alpha, &pA[offsetA], &pB[offsetB], mna);
			m -= mna;
			pB += 8*sdb;
			}
		else
			{
			if(offsetA==0)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_0_gen_lib8(n, &alpha, &pA[0], &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==1)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_1_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==2)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_2_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==3)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_3_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==4)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_4_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==5)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_5_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==6)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_6_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==7)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgecpsc_8_7_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			}
		}

	// same alignment
	if(offsetA==offsetB)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_0_lib8(n, &alpha, pA, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_0_gen_lib8(n, &alpha, pA, pB, m-ii);
			}
		return;
		}
	// XXX different alignment: search tree ???
	// skip one element of A
	else if(offsetA==(offsetB+1)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_1_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_1_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		}
	// skip two elements of A
	else if(offsetA==(offsetB+2)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_2_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_2_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip three elements of A
	else if(offsetA==(offsetB+3)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_3_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_3_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip four elements of A
	else if(offsetA==(offsetB+4)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_4_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_4_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip five elements of A
	else if(offsetA==(offsetB+5)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_5_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_5_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip six elements of A
	else if(offsetA==(offsetB+6)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_6_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_6_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip seven elements of A
	else //if(offsetA==(offsetB+7)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgecpsc_8_7_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgecpsc_8_7_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}

	return;

	}


// scale a strvec
void svecsc_libstr(int m, float alpha, struct s_strvec *sa, int ai)
	{
	float *pa = sa->pa + ai;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pa[ii+0] *= alpha;
		pa[ii+1] *= alpha;
		pa[ii+2] *= alpha;
		pa[ii+3] *= alpha;
		}
	for(; ii<m; ii++)
		{
		pa[ii+0] *= alpha;
		}
	return;
	}



// copy a strvec into a strvec
void sveccp_libstr(int m, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci)
	{
	float *pa = sa->pa + ai;
	float *pc = sc->pa + ci;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pc[ii+0] = pa[ii+0];
		pc[ii+1] = pa[ii+1];
		pc[ii+2] = pa[ii+2];
		pc[ii+3] = pa[ii+3];
		}
	for(; ii<m; ii++)
		{
		pc[ii+0] = pa[ii+0];
		}
	return;
	}



// copy and scale a strvec into a strvec
void sveccpsc_libstr(int m, float alpha, struct s_strvec *sa, int ai, struct s_strvec *sc, int ci)
	{
	float *pa = sa->pa + ai;
	float *pc = sc->pa + ci;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		pc[ii+0] = alpha*pa[ii+0];
		pc[ii+1] = alpha*pa[ii+1];
		pc[ii+2] = alpha*pa[ii+2];
		pc[ii+3] = alpha*pa[ii+3];
		}
	for(; ii<m; ii++)
		{
		pc[ii+0] = alpha*pa[ii+0];
		}
	return;
	}



// copy a lower triangular strmat into a lower triangular strmat
void strcp_l_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	int sdc = sC->cn;
	float *pC = sC->pA + ci/bs*bs*sdc + ci%bs + cj*bs;
	strcp_l_lib(m, ai%bs, pA, sda, ci%bs, pC, sdc);
	// XXX uses full matrix copy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//	sgecp_libstr(m, m, sA, ai, aj, sC, ci, cj);
	return;
	}



// scale and add a generic strmat into a generic strmat
void sgead_libstr(int m, int n, float alpha, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj)
	{

	// early return
	if(m==0 | n==0)
		return;

#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** sgead_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** sgead_libstr : n<0 : %d<0 *****\n", n);
	// non-negative offset
	if(ai<0) printf("\n****** sgead_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** sgead_libstr : aj<0 : %d<0 *****\n", aj);
	if(bi<0) printf("\n****** sgead_libstr : bi<0 : %d<0 *****\n", bi);
	if(bj<0) printf("\n****** sgead_libstr : bj<0 : %d<0 *****\n", bj);
	// inside matrix
	// A: m x n
	if(ai+m > sA->m) printf("\n***** sgead_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+n > sA->n) printf("\n***** sgead_libstr : aj+n > col(A) : %d+%d > %d *****\n", aj, n, sA->n);
	// B: m x n
	if(bi+m > sB->m) printf("\n***** sgead_libstr : bi+m > row(B) : %d+%d > %d *****\n", bi, m, sB->m);
	if(bj+n > sB->n) printf("\n***** sgead_libstr : bj+n > col(B) : %d+%d > %d *****\n", bj, n, sB->n);
#endif

	const int bs = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	float *pA = sA->pA + ai/bs*bs*sda + aj*bs;
	float *pB = sB->pA + bi/bs*bs*sdb + bj*bs;
	int offsetA = ai%bs;
	int offsetB = bi%bs;

	int ii, mna;

#if 1
	if(offsetB>0)
		{
		if(offsetB>offsetA)
			{
			mna = bs-offsetB;
			mna = m<mna ? m : mna;
			kernel_sgead_8_0_gen_lib8(n, &alpha, &pA[offsetA], &pB[offsetB], mna);
			m -= mna;
			//pA += 8*sda;
			pB += 8*sdb;
			}
		else
			{
			if(offsetA==0)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_0_gen_lib8(n, &alpha, &pA[0], &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==1)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_1_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==2)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_2_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==3)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_3_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==4)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_4_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==5)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_5_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==6)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_6_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			else if(offsetA==7)
				{
				mna = bs-offsetB;
				mna = m<mna ? m : mna;
				kernel_sgead_8_7_gen_lib8(n, &alpha, &pA[0], sda, &pB[offsetB], mna);
				m -= mna;
				pA += 8*sda;
				pB += 8*sdb;
				}
			}
		}
#endif

	// same alignment
	if(offsetA==offsetB)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_0_lib8(n, &alpha, pA, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_0_gen_lib8(n, &alpha, pA, pB, m-ii);
			}
		return;
		}
	// XXX different alignment: search tree ???
	// skip one element of A
	else if(offsetA==(offsetB+1)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_1_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_1_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		}
	// skip two elements of A
	else if(offsetA==(offsetB+2)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_2_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_2_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip three elements of A
	else if(offsetA==(offsetB+3)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_3_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_3_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip four elements of A
	else if(offsetA==(offsetB+4)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_4_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_4_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip five elements of A
	else if(offsetA==(offsetB+5)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_5_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_5_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip six elements of A
	else if(offsetA==(offsetB+6)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_6_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_6_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}
	// skip seven elements of A
	else //if(offsetA==(offsetB+7)%bs)
		{
		ii = 0;
		for( ; ii<m-7; ii+=8)
			{
			kernel_sgead_8_7_lib8(n, &alpha, pA, sda, pB);
			pA += 8*sda;
			pB += 8*sdb;
			}
		if(ii<m)
			{
			kernel_sgead_8_7_gen_lib8(n, &alpha, pA, sda, pB, m-ii);
			}
		return;
		}

	return;

	}



// copy and transpose a generic strmat into a generic strmat
void sgetr_libstr(int m, int n, struct s_strmat *sA, int ai, int aj, struct s_strmat *sB, int bi, int bj)
	{

	// early return
	if(m==0 | n==0)
		return;

#if defined(DIM_CHECK)
	// non-negative size
	if(m<0) printf("\n****** sgetr_libstr : m<0 : %d<0 *****\n", m);
	if(n<0) printf("\n****** sgetr_libstr : n<0 : %d<0 *****\n", n);
	// non-negative offset
	if(ai<0) printf("\n****** sgetr_libstr : ai<0 : %d<0 *****\n", ai);
	if(aj<0) printf("\n****** sgetr_libstr : aj<0 : %d<0 *****\n", aj);
	if(bi<0) printf("\n****** sgetr_libstr : bi<0 : %d<0 *****\n", bi);
	if(bj<0) printf("\n****** sgetr_libstr : bj<0 : %d<0 *****\n", bj);
	// inside matrix
	// A: m x n
	if(ai+m > sA->m) printf("\n***** sgetr_libstr : ai+m > row(A) : %d+%d > %d *****\n", ai, m, sA->m);
	if(aj+n > sA->n) printf("\n***** sgetr_libstr : aj+n > col(A) : %d+%d > %d *****\n", aj, n, sA->n);
	// B: n x m
	if(bi+n > sB->m) printf("\n***** sgetr_libstr : bi+n > row(B) : %d+%d > %d *****\n", bi, n, sB->m);
	if(bj+m > sB->n) printf("\n***** sgetr_libstr : bj+m > col(B) : %d+%d > %d *****\n", bj, m, sB->n);
#endif

	const int bs = 8;

	int sda = sA->cn;
	int sdb = sB->cn;
	float *pA = sA->pA + ai/bs*bs*sda + aj*bs;
	float *pB = sB->pA + bi/bs*bs*sdb + bj*bs;
	int offsetA = ai%bs;
	int offsetB = bi%bs;

	int ii, nna;

	if(offsetA==0)
		{
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_0_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for(ii=0; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_0_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_0_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}
	// TODO log serach for offsetA>0 ???
	else if(offsetA==1)
		{
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_1_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for(ii=0; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_1_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_1_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}
	else if(offsetA==2)
		{
		ii = 0;
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_2_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for( ; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_2_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_2_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}
	else if(offsetA==3)
		{
		ii = 0;
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_3_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for( ; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_3_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_3_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}
	else if(offsetA==4)
		{
		ii = 0;
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_4_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for( ; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_4_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_4_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}
	else if(offsetA==5)
		{
		ii = 0;
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_5_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for( ; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_5_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_5_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}
	else if(offsetA==6)
		{
		ii = 0;
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_6_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for( ; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_6_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_6_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}
	else if(offsetA==7)
		{
		ii = 0;
		if(offsetB>0)
			{
			nna = bs-offsetB;
			nna = n<nna ? n : nna;
			kernel_sgetr_8_7_gen_lib8(m, &pA[0], sda, &pB[offsetB], nna);
			n -= nna;
			pA += nna*bs;
			pB += 8*sdb;
			}
		for( ; ii<n-7; ii+=8)
			{
			kernel_sgetr_8_7_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb]);
			}
		if(ii<n)
			{
			kernel_sgetr_8_7_gen_lib8(m, &pA[ii*bs], sda, &pB[ii*sdb], n-ii);
			}
		}

	return;

	}



// copy and transpose a lower triangular strmat into an upper triangular strmat
void strtr_l_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	int sdc = sC->cn;
	float *pC = sC->pA + ci/bs*bs*sdc + ci%bs + cj*bs;
	strtr_l_lib(m, 1.0, ai%bs, pA, sda, ci%bs, pC, sdc); // TODO remove alpha !!!
	return;
	}



// copy and transpose an upper triangular strmat into a lower triangular strmat
void strtr_u_libstr(int m, struct s_strmat *sA, int ai, int aj, struct s_strmat *sC, int ci, int cj)
	{
	const int bs = 8;
	int sda = sA->cn;
	float *pA = sA->pA + ai/bs*bs*sda + ai%bs + aj*bs;
	int sdc = sC->cn;
	float *pC = sC->pA + ci/bs*bs*sdc + ci%bs + cj*bs;
	strtr_u_lib(m, 1.0, ai%bs, pA, sda, ci%bs, pC, sdc); // TODO remove alpha !!!
	return;
	}



// insert a strvec to diagonal of strmat, sparse formulation
void sdiain_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj)
	{
	const int bs = 8;
	float *x = sx->pa + xi;
	int sdd = sD->cn;
	float *pD = sD->pA;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[(ii+di)/bs*bs*sdd+(ii+di)%bs+(ii+dj)*bs] = alpha * x[jj];
		}
	return;
	}



// extract the diagonal of a strmat to a strvec, sparse formulation
void sdiaex_sp_libstr(int kmax, float alpha, int *idx, struct s_strmat *sD, int di, int dj, struct s_strvec *sx, int xi)
	{
	const int bs = 8;
	float *x = sx->pa + xi;
	int sdd = sD->cn;
	float *pD = sD->pA;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		x[jj] = alpha * pD[(ii+di)/bs*bs*sdd+(ii+di)%bs+(ii+dj)*bs];
		}
	return;
	}



// add scaled strvec to diagonal of strmat, sparse formulation
void sdiaad_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj)
	{
	const int bs = 8;
	float *x = sx->pa + xi;
	int sdd = sD->cn;
	float *pD = sD->pA;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[(ii+di)/bs*bs*sdd+(ii+di)%bs+(ii+dj)*bs] += alpha * x[jj];
		}
	return;
	}



// add scaled strvec to another strvec and insert to diagonal of strmat, sparse formulation
void sdiaadin_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, struct s_strvec *sy, int yi, int *idx, struct s_strmat *sD, int di, int dj)
	{
	const int bs = 8;
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	int sdd = sD->cn;
	float *pD = sD->pA;
	int ii, jj;
	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pD[(ii+di)/bs*bs*sdd+(ii+di)%bs+(ii+dj)*bs] = y[jj] + alpha * x[jj];
		}
	return;
	}



// add scaled strvec to row of strmat, sparse formulation
void srowad_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strmat *sD, int di, int dj)
	{
	const int bs = 8;
	float *x = sx->pa + xi;
	int sdd = sD->cn;
	float *pD = sD->pA + di/bs*bs*sdd + di%bs + dj*bs;
	srowad_libsp(kmax, idx, alpha, x, pD);
	return;
	}



// adds strvec to strvec, sparse formulation
void svecad_sp_libstr(int kmax, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strvec *sy, int yi)
	{
	float *x = sx->pa + xi;
	float *y = sy->pa + yi;
	svecad_libsp(kmax, idx, alpha, x, y);
	return;
	}



void svecin_sp_libstr(int m, float alpha, struct s_strvec *sx, int xi, int *idx, struct s_strvec *sz, int zi)
	{
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[idx[ii]] = alpha * x[ii];
	return;
	}



void svecex_sp_libstr(int m, float alpha, int *idx, struct s_strvec *sx, int xi, struct s_strvec *sz, int zi)
	{
	float *x = sx->pa + xi;
	float *z = sz->pa + zi;
	int ii;
	for(ii=0; ii<m; ii++)
		z[ii] = alpha * x[idx[ii]];
	return;
	}



// permute elements of a vector struct
void svecpe_libstr(int kmax, int *ipiv, struct s_strvec *sx, int xi)
	{
	int ii;
	float tmp;
	float *x = sx->pa + xi;
	for(ii=0; ii<kmax; ii++)
		{
		if(ipiv[ii]!=ii)
			{
			tmp = x[ipiv[ii]];
			x[ipiv[ii]] = x[ii];
			x[ii] = tmp;
			}
		}
	return;
	}



// inverse permute elements of a vector struct
void svecpei_libstr(int kmax, int *ipiv, struct s_strvec *sx, int xi)
	{
	int ii;
	float tmp;
	float *x = sx->pa + xi;
	for(ii=kmax-1; ii>=0; ii--)
		{
		if(ipiv[ii]!=ii)
			{
			tmp = x[ipiv[ii]];
			x[ipiv[ii]] = x[ii];
			x[ii] = tmp;
			}
		}
	return;
	}



#else

#error : wrong LA choice

#endif




