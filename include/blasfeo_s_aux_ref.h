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

#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif



// expose reference BLASFEO for testing

void blasfeo_create_smat_ref(int m, int n, struct blasfeo_smat_ref *sA, void *memory);
void blasfeo_pack_smat_ref(int m, int n, float *A, int lda, struct blasfeo_smat_ref *sA, int ai, int aj);
int blasfeo_memsize_smat_ref(int m, int n);

void blasfeo_sgecp_ref(int m, int n,\
					struct blasfeo_smat_ref *sA, int ai, int aj,\
					struct blasfeo_smat_ref *sB, int bi, int bj);
void blasfeo_sgesc_ref(int m, int n,\
					float alpha,\
					struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_sgecpsc_ref(int m, int n,
					float alpha,\
					struct blasfeo_smat_ref *sA, int ai, int aj,\
					struct blasfeo_smat_ref *sB, int bi, int bj);


#ifdef __cplusplus
}
#endif

