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
 * auxiliary algebra operation external dependancies header
 *
 * include/blasfeo_d_aux_ext_dep.h
 *
 * - dynamic memory allocation
 * - print
 *
 */

#ifndef BLASFEO_D_AUX_EXT_DEP_REF_H_
#define BLASFEO_D_AUX_EXT_DEP_REF_H_


#include <stdio.h>

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// expose reference BLASFEO for testing
// see blasfeo_d_aux_exp_dep.h for help

void blasfeo_print_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_allocate_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA);
void blasfeo_allocate_dvec_ref(int m, struct blasfeo_dvec_ref *sa);
void blasfeo_free_dmat_ref(struct blasfeo_dmat_ref *sA);
void blasfeo_free_dvec_ref(struct blasfeo_dvec_ref *sa);
void blasfeo_print_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_e_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_dmat_ref(FILE *file, int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_e_dmat_ref(FILE *file, int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_to_string_dmat_ref(char **buf_out, int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_e_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_file_dvec(FILE *file, int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_string_dvec(char **buf_out, int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_tran_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_e_tran_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_file_tran_dvec(FILE *file, int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_string_tran_dvec(char **buf_out, int m, struct blasfeo_dvec *sa, int ai);

#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_AUX_EXT_DEP_REF_H_
