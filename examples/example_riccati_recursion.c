/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl and at        *
* DTU Compute (Technical University of Denmark) under the supervision of John Bagterp Jorgensen.  *
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
#include <sys/time.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_i_aux.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"
#include "../include/blasfeo_d_blas.h"



void d_back_ric_libstr(int N, int *nx, int *nu, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hsux, struct d_strvec *hspi, struct d_strmat *hswork_mat, struct d_strvec *hswork_vec)
	{

	int nn;

	// factorization and backward substitution
//	for(nn=0; nn<N; nn++)
//		{
//		printf("\n%d %d\n", N, nn);
//		d_print_strmat(nu[nn]+nx[nn]+1, nx[nn+1], &hsBAbt[nn], 0, 0);
//		}
//	for(nn=0; nn<N; nn++)
//		{
//		printf("\n%d %d\n", N, N-nn-1);
//		d_print_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], &hsBAbt[N-nn-1], 0, 0);
//		}
//	return;
	// last stage
	dpotrf_l_libstr(nx[N]+1, nx[N], &hsRSQrq[N], 0, 0, &hsL[N], 0, 0);
//	d_print_strmat(nx[N]+1, nx[N], &hsL[N], 0, 0);
	dtrtr_l_libstr(nx[N], &hsL[N], 0, 0, &hsLxt[N], 0, 0);
//	d_print_strmat(nx[N], nx[N], &hsLxt[N], 0, 0);
	for(nn=0; nn<N; nn++)
		{
//		d_print_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], &hsBAbt[N-nn-1], 0, 0);
//		d_print_strmat(nx[N-nn-1], nx[N-nn-1], &hsLxt[N-nn], 0, 0);
		dtrmm_rutn_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], 1.0, &hsBAbt[N-nn-1], 0, 0, &hsLxt[N-nn], 0, 0, 0.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0);
		dgead_libstr(1, nx[N-nn], 1.0, &hsL[N-nn], nu[N-nn]+nx[N-nn], nu[N-nn], &hswork_mat[0], nu[N-nn-1]+nx[N-nn-1], 0);
//		d_print_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], &hswork_mat[0], 0, 0);
#if 0
		dsyrk_ln_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], nx[N-nn], 1.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0, 1.0, &hsRSQrq[N-nn-1], 0, 0, &hsL[N-nn-1], 0, 0);
//		d_print_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], &hsL[N-nn-1], 0, 0);
		dpotrf_l_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], &hsL[N-nn-1], 0, 0, &hsL[N-nn-1], 0, 0);
#else
		dsyrk_dpotrf_ln_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], nx[N-nn], &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0, &hsRSQrq[N-nn-1], 0, 0, &hsL[N-nn-1], 0, 0);
#endif
//		d_print_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], &hsL[N-nn-1], 0, 0);
		dtrtr_l_libstr(nx[N-nn-1], &hsL[N-nn-1], nu[N-nn-1], nu[N-nn-1], &hsLxt[N-nn-1], 0, 0);
//		d_print_strmat(nx[N-nn-1], nx[N-nn-1], &hsLxt[N-nn-1], 0, 0);
		}
	
	// forward substitution
	nn = 0;
//	d_print_strmat(nu[nn]+nx[nn]+1, nu[nn]+nx[nn], &hsL[nn], 0, 0);
	drowex_libstr(nu[nn]+nx[nn], -1.0, &hsL[nn], nu[nn]+nx[nn], 0, &hsux[nn], 0);
//	d_print_tran_strvec(nu[nn]+nx[nn], &hsux[nn], 0);
	dtrsv_ltn_libstr(nu[nn]+nx[nn], nu[nn]+nx[nn], &hsL[nn], 0, 0, &hsux[nn], 0, &hsux[nn], 0);
//	d_print_tran_strvec(nu[nn]+nx[nn], &hsux[nn], 0);
	drowex_libstr(nx[nn+1], 1.0, &hsBAbt[nn], nu[nn]+nx[nn], 0, &hsux[nn+1], nu[nn+1]);
//	d_print_tran_strvec(nu[nn+1]+nx[nn+1], &hsux[nn+1], 0);
//	d_print_strmat(nu[nn]+nx[nn]+1, nx[nn+1], &hsBAbt[nn], 0, 0);
	dgemv_t_libstr(nu[nn]+nx[nn], nx[nn+1], 1.0, &hsBAbt[nn], 0, 0, &hsux[nn], 0, 1.0, &hsux[nn+1], nu[nn+1], &hsux[nn+1], nu[nn+1]);
//	d_print_tran_strvec(nu[nn+1]+nx[nn+1], &hsux[nn+1], 0);
	dveccp_libstr(nx[nn+1], &hsux[nn+1], nu[nn+1], &hswork_vec[0], 0); // align
//	d_print_tran_strvec(nx[nn+1], &hswork_vec[0], 0);
//	d_print_strmat(nu[nn+1]+nx[nn+1]+1, nu[nn+1]+nx[nn+1], &hsL[nn+1], 0, 0);
//	d_print_strmat(nx[nn+1], nx[nn+1], &hsLxt[nn+1], 0, 0);
	drowex_libstr(nx[nn+1], 1.0, &hsL[nn+1], nu[nn+1]+nx[nn+1], nu[nn+1], &hswork_vec[1], 0);
//	d_print_tran_strvec(nx[nn+1], &hswork_vec[1], 0);
	dtrmv_unn_libstr(nx[nn+1], nx[nn+1], 1.0, &hsLxt[nn+1], 0, 0, &hswork_vec[0], 0, 1.0, &hswork_vec[1], 0, &hswork_vec[1], 0);
//	d_print_tran_strvec(nx[nn+1], &hswork_vec[1], 0);
	dtrmv_utn_libstr(nx[nn+1], nx[nn+1], 1.0, &hsLxt[nn+1], 0, 0, &hswork_vec[1], 0, 0.0, &hspi[nn], 0, &hspi[nn], 0);
//	d_print_tran_strvec(nx[nn+1], &hspi[nn], 0);

	for(nn=1; nn<N; nn++)
		{

	//	d_print_strmat(nu[nn]+nx[nn]+1, nu[nn]+nx[nn], &hsL[nn], 0, 0);
		drowex_libstr(nu[nn], -1.0, &hsL[nn], nu[nn]+nx[nn], 0, &hsux[nn], 0);
	//	d_print_tran_strvec(nu[nn]+nx[nn], &hsux[nn], 0);
		dtrsv_ltn_libstr(nu[nn]+nx[nn], nu[nn], &hsL[nn], 0, 0, &hsux[nn], 0, &hsux[nn], 0);
	//	d_print_tran_strvec(nu[nn]+nx[nn], &hsux[nn], 0);
		drowex_libstr(nx[nn+1], 1.0, &hsBAbt[nn], nu[nn]+nx[nn], 0, &hsux[nn+1], nu[nn+1]);
	//	d_print_tran_strvec(nu[nn+1]+nx[nn+1], &hsux[nn+1], 0);
	//	d_print_strmat(nu[nn]+nx[nn]+1, nx[nn+1], &hsBAbt[nn], 0, 0);
		dgemv_t_libstr(nu[nn]+nx[nn], nx[nn+1], 1.0, &hsBAbt[nn], 0, 0, &hsux[nn], 0, 1.0, &hsux[nn+1], nu[nn+1], &hsux[nn+1], nu[nn+1]);
	//	d_print_tran_strvec(nu[nn+1]+nx[nn+1], &hsux[nn+1], 0);
		dveccp_libstr(nx[nn+1], &hsux[nn+1], nu[nn+1], &hswork_vec[0], 0); // align
	//	d_print_tran_strvec(nx[nn+1], &hswork_vec[0], 0);
	//	d_print_strmat(nu[nn+1]+nx[nn+1]+1, nu[nn+1]+nx[nn+1], &hsL[nn+1], 0, 0);
		drowex_libstr(nx[nn+1], 1.0, &hsL[nn+1], nu[nn+1]+nx[nn+1], nu[nn+1], &hswork_vec[1], 0);
	//	d_print_tran_strvec(nx[nn+1], &hswork_vec[1], 0);
		dtrmv_unn_libstr(nx[nn+1], nx[nn+1], 1.0, &hsLxt[nn+1], 0, 0, &hswork_vec[0], 0, 1.0, &hswork_vec[1], 0, &hswork_vec[1], 0);
	//	d_print_tran_strvec(nx[nn+1], &hswork_vec[1], 0);
		dtrmv_utn_libstr(nx[nn+1], nx[nn+1], 1.0, &hsLxt[nn+1], 0, 0, &hswork_vec[1], 0, 0.0, &hspi[nn], 0, &hspi[nn], 0);
	//	d_print_tran_strvec(nx[nn+1], &hspi[nn], 0);

		}
	return;
	}




/************************************************ 
Mass-spring system: nx/2 masses connected each other with springs (in a row), and the first and the last one to walls. nu (<=nx) controls act on the first nu masses. The system is sampled with sampling time Ts. 
************************************************/
void mass_spring_system(double Ts, int nx, int nu, int N, double *A, double *B, double *b, double *x0)
	{

	int nx2 = nx*nx;

	int info = 0;

	int pp = nx/2; // number of masses
	
/************************************************
* build the continuous time system 
************************************************/
	
	double *T; d_zeros(&T, pp, pp);
	int ii;
	for(ii=0; ii<pp; ii++) T[ii*(pp+1)] = -2;
	for(ii=0; ii<pp-1; ii++) T[ii*(pp+1)+1] = 1;
	for(ii=1; ii<pp; ii++) T[ii*(pp+1)-1] = 1;

	double *Z; d_zeros(&Z, pp, pp);
	double *I; d_zeros(&I, pp, pp); for(ii=0; ii<pp; ii++) I[ii*(pp+1)]=1.0; // = eye(pp);
	double *Ac; d_zeros(&Ac, nx, nx);
	dmcopy(pp, pp, Z, pp, Ac, nx);
	dmcopy(pp, pp, T, pp, Ac+pp, nx);
	dmcopy(pp, pp, I, pp, Ac+pp*nx, nx);
	dmcopy(pp, pp, Z, pp, Ac+pp*(nx+1), nx); 
	free(T);
	free(Z);
	free(I);
	
	d_zeros(&I, nu, nu); for(ii=0; ii<nu; ii++) I[ii*(nu+1)]=1.0; //I = eye(nu);
	double *Bc; d_zeros(&Bc, nx, nu);
	dmcopy(nu, nu, I, nu, Bc+pp, nx);
	free(I);
	
/************************************************
* compute the discrete time system 
************************************************/

	double *bb; d_zeros(&bb, nx, 1);
	dmcopy(nx, 1, bb, nx, b, nx);
		
	dmcopy(nx, nx, Ac, nx, A, nx);
	dscal_3l(nx2, Ts, A);
	expm(nx, A);
	
	d_zeros(&T, nx, nx);
	d_zeros(&I, nx, nx); for(ii=0; ii<nx; ii++) I[ii*(nx+1)]=1.0; //I = eye(nx);
	dmcopy(nx, nx, A, nx, T, nx);
	daxpy_3l(nx2, -1.0, I, T);
	dgemm_nn_3l(nx, nu, nx, T, nx, Bc, nx, B, nx);
	free(T);
	free(I);
	
	int *ipiv = (int *) malloc(nx*sizeof(int));
	dgesv_3l(nx, nu, Ac, nx, ipiv, B, nx, &info);
	free(ipiv);

	free(Ac);
	free(Bc);
	free(bb);
	
			
/************************************************
* initial state 
************************************************/
	
	if(nx==4)
		{
		x0[0] = 5;
		x0[1] = 10;
		x0[2] = 15;
		x0[3] = 20;
		}
	else
		{
		int jj;
		for(jj=0; jj<nx; jj++)
			x0[jj] = 1;
		}

	}



int main()
	{

	printf("\nExample of LU factorization and backsolve\n\n");

#if defined(LA_BLASFEO)

	printf("\nLA provided by BLASFEO\n\n");

#elif defined(LA_BLAS)

	printf("\nLA provided by BLAS\n\n");

#else

	printf("\nLA provided by ???\n\n");
	exit(2);

#endif

	// loop index
	int ii;

/************************************************
* problem size
************************************************/	

	// problem size
	int N = 10;
	int nx_ = 30;
	int nu_ = 5;

	// stage-wise variant size
	int nx[N+1];
	nx[0] = 0;
	for(ii=1; ii<=N; ii++)
		nx[ii] = nx_;
	nx[N] = nx_;

	int nu[N+1];
	for(ii=0; ii<N; ii++)
		nu[ii] = nu_;
	nu[N] = 0;

/************************************************
* dynamical system
************************************************/	

	double *A; d_zeros(&A, nx_, nx_); // states update matrix

	double *B; d_zeros(&B, nx_, nu_); // inputs matrix

	double *b; d_zeros(&b, nx_, 1); // states offset
	double *x0; d_zeros_align(&x0, nx_, 1); // initial state

	double Ts = 0.5; // sampling time
	mass_spring_system(Ts, nx_, nu_, N, A, B, b, x0);
	
	for(ii=0; ii<nx_; ii++)
		b[ii] = 0.1;
	
	for(ii=0; ii<nx_; ii++)
		x0[ii] = 0;
	x0[0] = 2.5;
	x0[1] = 2.5;

	d_print_mat(nx_, nx_, A, nx_);
	d_print_mat(nx_, nu_, B, nx_);
	d_print_mat(1, nx_, b, 1);
	d_print_mat(1, nx_, x0, 1);

/************************************************
* cost function
************************************************/	

	double *R; d_zeros(&R, nu_, nu_);
	for(ii=0; ii<nu_; ii++) R[ii*(nu_+1)] = 2.0;

	double *S; d_zeros(&S, nu_, nx_);

	double *Q; d_zeros(&Q, nx_, nx_);
	for(ii=0; ii<nx_; ii++) Q[ii*(nx_+1)] = 1.0;

	double *r; d_zeros(&r, nu_, 1);
	for(ii=0; ii<nu_; ii++) r[ii] = 0.2;

	double *q; d_zeros(&q, nx_, 1);
	for(ii=0; ii<nx_; ii++) q[ii] = 0.1;

	d_print_mat(nu_, nu_, R, nu_);
	d_print_mat(nu_, nx_, S, nu_);
	d_print_mat(nx_, nx_, Q, nx_);
	d_print_mat(1, nu_, r, 1);
	d_print_mat(1, nx_, q, 1);

/************************************************
* matrices as strmat
************************************************/	

	struct d_strmat sA;
	d_allocate_strmat(nx_, nx_, &sA);
	d_cvt_mat2strmat(nx_, nx_, A, nx_, &sA, 0, 0);
	struct d_strvec sb;
	d_allocate_strvec(nx_, &sb);
	d_cvt_vec2strvec(nx_, b, &sb, 0);
	struct d_strvec sx0;
	d_allocate_strvec(nx_, &sx0);
	d_cvt_vec2strvec(nx_, x0, &sx0, 0);
	struct d_strvec sb0;
	d_allocate_strvec(nx_, &sb0);
	double *b0; d_zeros(&b0, nx_, 1); // states offset
	dgemv_n_libstr(nx_, nx_, 1.0, &sA, 0, 0, &sx0, 0, 1.0, &sb, 0, &sb0, 0);
	d_print_tran_strvec(nx_, &sb0, 0);

	struct d_strmat sBbt0;
	d_allocate_strmat(nu_+nx_+1, nx_, &sBbt0);
	d_cvt_tran_mat2strmat(nx_, nx_, B, nx_, &sBbt0, 0, 0);
	drowin_libstr(nx_, 1.0, &sb0, 0, &sBbt0, nu_, 0);
	d_print_strmat(nu_+1, nx_, &sBbt0, 0, 0);

	struct d_strmat sBAbt1;
	d_allocate_strmat(nu_+nx_+1, nx_, &sBAbt1);
	d_cvt_tran_mat2strmat(nx_, nu_, B, nx_, &sBAbt1, 0, 0);
	d_cvt_tran_mat2strmat(nx_, nx_, A, nx_, &sBAbt1, nu_, 0);
	d_cvt_tran_mat2strmat(nx_, 1, b, nx_, &sBAbt1, nu_+nx_, 0);
	d_print_strmat(nu_+nx_+1, nx_, &sBAbt1, 0, 0);

	struct d_strmat sRr0; // XXX no need to update r0 since S=0
	d_allocate_strmat(nu_+1, nu_, &sRr0);
	d_cvt_mat2strmat(nu_, nu_, R, nu_, &sRr0, 0, 0);
	d_cvt_tran_mat2strmat(nu_, 1, r, nu_, &sRr0, nu_, 0);
	d_print_strmat(nu_+1, nu_, &sRr0, 0, 0);

	struct d_strmat sRSQrq1;
	d_allocate_strmat(nu_+nx_+1, nu_+nx_, &sRSQrq1);
	d_cvt_mat2strmat(nu_, nu_, R, nu_, &sRSQrq1, 0, 0);
	d_cvt_tran_mat2strmat(nu_, nx_, S, nu_, &sRSQrq1, nu_, 0);
	d_cvt_mat2strmat(nx_, nx_, Q, nx_, &sRSQrq1, nu_, nu_);
	d_cvt_tran_mat2strmat(nu_, 1, r, nu_, &sRSQrq1, nu_+nx_, 0);
	d_cvt_tran_mat2strmat(nx_, 1, q, nx_, &sRSQrq1, nu_+nx_, nu_);
	d_print_strmat(nu_+nx_+1, nu_+nx_, &sRSQrq1, 0, 0);

	struct d_strmat sQqN;
	d_allocate_strmat(nx_+1, nx_, &sQqN);
	d_cvt_mat2strmat(nx_, nx_, Q, nx_, &sQqN, 0, 0);
	d_cvt_tran_mat2strmat(nx_, 1, q, nx_, &sQqN, nx_, 0);
	d_print_strmat(nx_+1, nx_, &sQqN, 0, 0);

/************************************************
* array of matrices
************************************************/	
	
	struct d_strmat hsBAbt[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strmat hsL[N+1];
	struct d_strmat hsLxt[N+1];
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N];
	struct d_strmat hswork_mat[1];
	struct d_strvec hswork_vec[2];

	hsRSQrq[0] = sRr0;
	hsBAbt[0] = sBbt0;
	d_allocate_strmat(nu_+1, nu_, &hsL[0]);
//	d_allocate_strmat(nu_+1, nu_, &hsLxt[0]);
	d_allocate_strvec(nx_+nu_+1, &hsux[0]);
	d_allocate_strvec(nx_, &hspi[0]);
	for(ii=1; ii<N; ii++)
		{
		hsRSQrq[ii] = sRSQrq1;
		hsBAbt[ii] = sBAbt1;
		d_allocate_strmat(nu_+nx_+1, nu_+nx_, &hsL[ii]);
		d_allocate_strmat(nx_, nu_+nx_, &hsLxt[ii]);
		d_allocate_strvec(nx_+nu_+1, &hsux[ii]);
		d_allocate_strvec(nx_, &hspi[ii]);
		}
	hsRSQrq[N] = sQqN;
	d_allocate_strmat(nx_+1, nx_, &hsL[N]);
	d_allocate_strmat(nx_, nx_, &hsLxt[N]);
	d_allocate_strvec(nx_+nu_+1, &hsux[N]);
	d_allocate_strmat(nu_+nx_+1, nx_, &hswork_mat[0]);
	d_allocate_strvec(nx_, &hswork_vec[0]);
	d_allocate_strvec(nx_, &hswork_vec[1]);

//	for(ii=0; ii<N; ii++)
//		d_print_strmat(nu[ii]+nx[ii]+1, nx[ii+1], &hsBAbt[ii], 0, 0);
//	return 0;

/************************************************
* call Riccati solver
************************************************/	
	
	// timing 
	struct timeval tv0, tv1, tv2, tv3;
	int nrep = 1000;
	int rep;

	gettimeofday(&tv0, NULL); // start

	for(rep=0; rep<nrep; rep++)
		{
		d_back_ric_libstr(N, nx, nu, hsBAbt, hsRSQrq, hsL, hsLxt, hsux, hspi, hswork_mat, hswork_vec);
		}

	gettimeofday(&tv1, NULL); // start

	float time_sv = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	// print sol
	printf("\nux = \n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);

	printf("\npi = \n\n");
	for(ii=0; ii<N; ii++)
		d_print_tran_strvec(nx[ii+1], &hspi[ii], 0);

	printf("\ntime\n");
	printf("\n%e\n", time_sv);

/************************************************
* free memory
************************************************/	

	d_free(A);
	d_free(B);
	d_free(b);
	d_free_align(x0);
	d_free(R);
	d_free(S);
	d_free(Q);
	d_free(r);
	d_free(q);
	d_free(b0);
	d_free_strmat(&sA);
	d_free_strmat(&sBbt0);
	d_free_strmat(&sBAbt1);
	d_free_strmat(&sRr0);
	d_free_strmat(&sRSQrq1);
	d_free_strmat(&sQqN);
	d_free_strmat(&hsL[0]);
//	d_free_strmat(&hsLxt[0]);
	d_free_strvec(&hsux[0]);
	d_free_strvec(&hspi[0]);
	for(ii=1; ii<N; ii++)
		{
		d_free_strmat(&hsL[ii]);
		d_free_strmat(&hsLxt[ii]);
		d_free_strvec(&hsux[ii]);
		d_free_strvec(&hspi[ii]);
		}
	d_free_strmat(&hsL[N]);
	d_free_strmat(&hsLxt[N]);
	d_free_strvec(&hsux[N]);
	d_free_strmat(&hswork_mat[0]);
	d_free_strvec(&hswork_vec[0]);
	d_free_strvec(&hswork_vec[1]);


/************************************************
* return
************************************************/	

	return 0;

	}



