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

/*
 * auxiliary algebra operations header
 *
 * include/blasfeo_aux_lib*.h
 *
 */

#ifdef __cplusplus
extern "C" {
#endif


// expose reference BLASFEO for testing

void blasfeo_d_create_strmat_ref(int m, int n, struct blasfeo_dmat_ref *sA, void *memory);
void blasfeo_d_cvt_mat2strmat_ref(int m, int n, double *A, int lda, struct blasfeo_dmat_ref *sA, int ai, int aj);
int blasfeo_d_memsize_strmat_ref(int m, int n);

void blasfeo_dgecp_ref(int m, int n,\
					struct blasfeo_dmat_ref *sA, int ai, int aj,\
					struct blasfeo_dmat_ref *sB, int bi, int bj);
void blasfeo_dgesc_ref(int m, int n,\
					double alpha,\
					struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_dgecpsc_ref(int m, int n,
					double alpha,\
					struct blasfeo_dmat_ref *sA, int ai, int aj,\
					struct blasfeo_dmat_ref *sB, int bi, int bj);


#ifdef __cplusplus
}
#endif
