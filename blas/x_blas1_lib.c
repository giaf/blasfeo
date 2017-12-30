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



#if defined(LA_REFERENCE)



void AXPY_LIBSTR(int m, REAL alpha, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	int ii;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
		z[ii+1] = y[ii+1] + alpha*x[ii+1];
		z[ii+2] = y[ii+2] + alpha*x[ii+2];
		z[ii+3] = y[ii+3] + alpha*x[ii+3];
		}
	for(; ii<m; ii++)
		z[ii+0] = y[ii+0] + alpha*x[ii+0];
	return;
	}

void AXPBY_LIBSTR(int m, REAL alpha, struct STRVEC *sx, int xi, REAL beta, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	int ii;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z[ii+0] = beta*y[ii+0] + alpha*x[ii+0];
		z[ii+1] = beta*y[ii+1] + alpha*x[ii+1];
		z[ii+2] = beta*y[ii+2] + alpha*x[ii+2];
		z[ii+3] = beta*y[ii+3] + alpha*x[ii+3];
		}
	for(; ii<m; ii++)
		z[ii+0] = beta*y[ii+0] + alpha*x[ii+0];
	return;
	}



// multiply two vectors
void VECMULACC_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	int ii;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z[ii+0] += x[ii+0] * y[ii+0];
		z[ii+1] += x[ii+1] * y[ii+1];
		z[ii+2] += x[ii+2] * y[ii+2];
		z[ii+3] += x[ii+3] * y[ii+3];
		}
	for(; ii<m; ii++)
		{
		z[ii+0] += x[ii+0] * y[ii+0];
		}
	return;
	}



// multiply two vectors and compute dot product
REAL VECMULDOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		z[ii+1] = x[ii+1] * y[ii+1];
		z[ii+2] = x[ii+2] * y[ii+2];
		z[ii+3] = x[ii+3] * y[ii+3];
		dot += z[ii+0] + z[ii+1] + z[ii+2] + z[ii+3];
		}
	for(; ii<m; ii++)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		dot += z[ii+0];
		}
	return dot;
	}



// compute dot product of two vectors
REAL DOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		dot += x[ii+0] * y[ii+0];
		dot += x[ii+1] * y[ii+1];
		dot += x[ii+2] * y[ii+2];
		dot += x[ii+3] * y[ii+3];
		}
	for(; ii<m; ii++)
		{
		dot += x[ii+0] * y[ii+0];
		}
	return dot;
	}



// construct givens plane rotation
void ROTG_LIBSTR(REAL a, REAL b, REAL *c, REAL *s)
	{
	REAL aa = FABS(a);
	REAL bb = FABS(b);
	REAL roe = (aa >= bb) ? a : b;
	REAL scale = aa + bb;
	REAL r;
	if (scale == 0)
		{
		*c = 1.0;
		*s = 0.0;
		}
	else
		{
		aa = a/scale;
		bb = b/scale;
		r = scale * SQRT(aa*aa + bb*bb);
		r = r * (roe >= 0 ? 1 : -1);
		*c = a / r;
		*s = b / r;	
		}
	return;
	}



// apply plane rotation to the aj0 and aj1 columns of A at row index ai
void COLROT_LIBSTR(int m, struct STRMAT *sA, int ai, int aj0, int aj1, REAL c, REAL s)
	{
	int lda = sA->m;
	REAL *px = sA->pA + ai + aj0*lda;
	REAL *py = sA->pA + ai + aj1*lda;
	int ii;
	REAL d_tmp;
	for(ii=0; ii<m; ii++)
		{
		d_tmp  = c*px[ii] + s*py[ii];
		py[ii] = c*py[ii] - s*px[ii];
		px[ii] = d_tmp;
		}
	return;
	}



// apply plane rotation to the ai0 and ai1 rows of A at column index aj
void ROWROT_LIBSTR(int m, struct STRMAT *sA, int ai0, int ai1, int aj, REAL c, REAL s)
	{
	int lda = sA->m;
	REAL *px = sA->pA + ai0 + aj*lda;
	REAL *py = sA->pA + ai1 + aj*lda;
	int ii;
	REAL d_tmp;
	for(ii=0; ii<m; ii++)
		{
		d_tmp  = c*px[ii*lda] + s*py[ii*lda];
		py[ii*lda] = c*py[ii*lda] - s*px[ii*lda];
		px[ii*lda] = d_tmp;
		}
	return;
	}



#elif defined(LA_BLAS)



void AXPY_LIBSTR(int m, REAL alpha, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	int i1 = 1;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	if(y!=z)
		COPY(&m, y, &i1, z, &i1);
	AXPY(&m, &alpha, x, &i1, z, &i1);
	return;
	}


void AXPBY_LIBSTR(int m, REAL alpha, struct STRVEC *sx, REAL beta, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return;
	int i1 = 1;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	if(y!=z)
		COPY(&m, y, &i1, z, &i1);
	SCAL(&m, &beta, z, &i1);
	AXPY(&m, &alpha, x, &i1, z, &i1);
	return;
	}


// multiply two vectors and compute dot product
REAL VECMULDOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi, struct STRVEC *sz, int zi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	REAL *z = sz->pa + zi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m; ii++)
		{
		z[ii+0] = x[ii+0] * y[ii+0];
		dot += z[ii+0];
		}
	return dot;
	}



// compute dot product of two vectors
REAL DOT_LIBSTR(int m, struct STRVEC *sx, int xi, struct STRVEC *sy, int yi)
	{
	if(m<=0)
		return 0.0;
	REAL *x = sx->pa + xi;
	REAL *y = sy->pa + yi;
	int ii;
	REAL dot = 0.0;
	ii = 0;
	for(; ii<m-3; ii+=4)
		{
		dot += x[ii+0] * y[ii+0];
		dot += x[ii+1] * y[ii+1];
		dot += x[ii+2] * y[ii+2];
		dot += x[ii+3] * y[ii+3];
		}
	for(; ii<m; ii++)
		{
		dot += x[ii+0] * y[ii+0];
		}
	return dot;
	}



// construct givens plane rotation
void ROTG_LIBSTR(REAL a, REAL b, REAL *c, REAL *s)
	{
	ROTG(&a, &b, c, s);
	return;
	}



// apply plane rotation to the aj0 and aj1 columns of A at row index ai
void COLROT_LIBSTR(int m, struct STRMAT *sA, int ai, int aj0, int aj1, REAL c, REAL s)
	{
	int lda = sA->m;
	REAL *px = sA->pA + ai + aj0*lda;
	REAL *py = sA->pA + ai + aj1*lda;
	int i1 = 1;
	ROT(&m, px, &i1, py, &i1, &c, &s);
	return;
	}



// apply plane rotation to the ai0 and ai1 rows of A at column index aj
void ROWROT_LIBSTR(int m, struct STRMAT *sA, int ai0, int ai1, int aj, REAL c, REAL s)
	{
	int lda = sA->m;
	REAL *px = sA->pA + ai0 + aj*lda;
	REAL *py = sA->pA + ai1 + aj*lda;
	ROT(&m, px, &lda, py, &lda, &c, &s);
	return;
	}



#else

#error : wrong LA choice

#endif


