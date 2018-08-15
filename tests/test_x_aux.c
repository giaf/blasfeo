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
	int n = 21;
	int p_n = 15;
	int N = 10;

	// initialized matrices in column-major format
	//
	REAL *A;
	// standard column major allocation (malloc)
	s_zeros(&A, n, n);
	for(ii=0; ii<n*n; ii++) A[ii] = ii;

	REAL *B;
	// standard column major allocation (malloc)
	s_zeros(&B, n, n);
	for(ii=0; ii<n*n; ii++) B[ii] = 2*ii;


	/* -------- instantiate blasfeo_smat */

	// compute memory size
	int size_strmat = N*blasfeo_memsize_smat(n, n);
	// inizilize void pointer
	void *memory_strmat;

	// initialize pointer
	// memory allocation
	v_zeros_align(&memory_strmat, size_strmat);

	// get point to strmat
	char *ptr_memory_strmat = (char *) memory_strmat;

	// -------- instantiate blasfeo_smat
	printf("\nInstantiate matrices\n\n");

	// instantiate blasfeo_smat depend on compilation flag LA_BLAS || LA_REFERENCE
	struct blasfeo_smat sA;
	blasfeo_create_smat(n, n, &sA, ptr_memory_strmat);
	ptr_memory_strmat += sA.memsize;
	blasfeo_pack_smat(n, n, A, n, &sA, 0, 0);

	struct blasfeo_smat sB;
	blasfeo_create_smat(n, n, &sB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memsize;
	blasfeo_pack_smat(n, n, B, n, &sB, 0, 0);

	// Testing comparison
	// reference matrices, column major

	struct blasfeo_smat rA;
	test_blasfeo_create_smat(n, n, &rA, ptr_memory_strmat);
	ptr_memory_strmat += rA.memsize;
	test_blasfeo_pack_smat(n, n, A, n, &rA, 0, 0);

	struct blasfeo_smat rB;
	test_blasfeo_create_smat(n, n, &rB, ptr_memory_strmat);
	ptr_memory_strmat += sB.memsize;
	test_blasfeo_pack_smat(n, n, B, n, &rB, 0, 0);


	// -------- instantiate blasfeo_smat

	// test operations
	//
	/* blasfeo_sgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 1.0, &sB, 0, 0, &sC, 0, 0); */

	/* printf("\nPrint mat B:\n\n"); */
	/* s_print_mat(p_n, p_n, B, n); */

	printf("\nPrint strmat HP A:\n\n");
	blasfeo_print_smat(p_n, p_n, &sA, 0, 0);

	printf("\nPrint strmat REF A:\n\n");
	test_blasfeo_print_smat(p_n, p_n, &rA, 0, 0);

	printf("\nPrint strmat HP B:\n\n");
	blasfeo_print_smat(p_n, p_n, &sB, 0, 0);

	printf("\nPrint strmat REF B:\n\n");
	test_blasfeo_print_smat(p_n, p_n, &rB, 0, 0);


	/* printf("\nPrint stored strmat A:\n\n"); */
	/* blasfeo_print_smat((&sA)->pm, (&sA)->cn, &sA, 0, 0); */

	/* printf("\nPrint strmat B:\n\n"); */
	/* blasfeo_print_smat(p_n, p_n, &sB, 0, 0); */

	/* AUX */

	/* ----------- memory */
	/* printf("----------- STRMAT memory\n\n"); */
	/* for (int i=0; i<12; i++) */
	/* { */
		/* printf("%d: %f, %f\n", i, sA.pA[i], A[i]); */
	/* } */
	/* printf("...\n\n"); */


	/* ---------- extraction */
	/* printf("----------- Extraction\n\n"); */

	/* int ai = 8; */
	/* int aj = 1; */

	// ---- strmat
	/* REAL ex_val = blasfeo_sgeex1(&sA, ai, aj); */
	/* printf("Extract %d,%d for A: %f\n\n", ai, aj, ex_val); */

	/* ---- column major */
	/* struct blasfeo_smat* ssA = &sA; */
	/* int lda = (&sA)->m; */
	/* REAL pointer + n_rows + n_col*leading_dimension; */
	/* REAL *pA = (&sA)->pA + ai + aj*lda; */
	/* REAL val = pA[0]; */

	/* ----------- copy and scale */
	printf("\n\n----------- TEST Copy&Scale\n\n");

	REAL alpha;
	alpha = 1.5;
	int ret, ni, mi;
	ni = 12;
	mi = 10;

	printf("Compute different combinations of submatrix offsets\n\n");

	// loop over A offset
	for (ii = 0; ii < 8; ii++)
		{

		// ---- Scale
		//
		printf("Scale A[%d:%d,%d:%d] by %f\n",
						ii,ni, 0,mi,    alpha);

		blasfeo_sgesc(     ni, mi, alpha, &sA, ii, 0);
		test_blasfeo_sgesc(ni, mi, alpha, &rA, ii, 0);

		/* printf("value 0,1: %f", MATEL_LIBSTR(&sA, 0,1)); */
		/* printf("PS:%d", PS); */

		assert(sgecmp_libstr(n, n, &sA, &rA));

		// loop over B offset
		for (jj = 0; jj < 8; jj++)
			{

			// ---- Copy&Scale
			//

			printf("Copy-Scale A[%d:%d,%d:%d] by %f in B[%d:%d,%d:%d]\n",
							     ii,ni, 0,mi,    alpha,  jj,ni, 0,mi);

			// HP submatrix copy&scale
			blasfeo_sgecpsc(ni, mi, alpha, &sA, ii, 0, &sB, jj, 0);
			// REF submatrix copy&scale
			test_blasfeo_sgecpsc(ni, mi, alpha, &rA, ii, 0, &rB, jj, 0);
			// check against blas with blasfeo REF
			assert(sgecmp_libstr(n, n, &sB, &rB));

			// ---- Copy
			//
			printf("Copy A[%d:%d,%d:%d] in B[%d:%d,%d:%d]\n",
							ii,ni, 0,mi,     jj,ni, 0,mi);

			blasfeo_sgecp(     ni, mi, &sA, ii, 0, &sB, jj, 0);
			test_blasfeo_sgecp(ni, mi, &rA, ii, 0, &rB, jj, 0);
			assert(sgecmp_libstr(n, n, &sB, &rB));

			printf("\n");
			}

		printf("\n");
		}

	printf("\n\n----------- END TEST Copy&Scale\n\n");

#if defined(LA)
SHOW_DEFINE(LA)
#endif
#if defined(TARGET)
SHOW_DEFINE(TARGET)
#endif

	}
