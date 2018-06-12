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

#ifndef BLASFEO_S_AUX_EXT_DEP_REF_H_
#define BLASFEO_S_AUX_EXT_DEP_REF_H_

#if defined(EXT_DEP)



#include <stdio.h>

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// expose reference BLASFEO for testing
// see blasfeo_s_aux_exp_dep.h for help

void blasfeo_print_smat_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_allocate_smat_ref(int m, int n, struct blasfeo_smat_ref *sA);
void blasfeo_allocate_svec_ref(int m, struct blasfeo_svec_ref *sa);
void blasfeo_free_smat_ref(struct blasfeo_smat_ref *sA);
void blasfeo_free_svec_ref(struct blasfeo_svec_ref *sa);
void blasfeo_print_smat_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_e_smat_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_smat_ref(FILE *file, int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_e_smat_ref(FILE *file, int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_to_string_smat_ref(char **buf_out, int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_e_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_file_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_string_svec(char **buf_out, int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_tran_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_e_tran_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_file_tran_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_string_tran_svec(char **buf_out, int m, struct blasfeo_svec *sa, int ai);


#ifdef __cplusplus
}
#endif



#endif  // EXT_DEP

#endif  // BLASFEO_S_AUX_EXT_DEP_REF_H_
