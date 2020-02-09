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

#include <stdlib.h>
#include <stdio.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux_ext_dep.h"


#define ZEROS d_zeros
#define ZEROS_ALIGN d_zeros_align

#define FREE d_free
#define FREE_ALIGN d_free_align

#define PRINT_MAT d_print_mat
#define PRINT_TO_FILE_MAT d_print_to_file_mat
#define PRINT_TO_FILE_EXP_MAT d_print_to_file_exp_mat
#define PRINT_TO_STRING_MAT d_print_to_string_mat

#define PRINT_TRAN_MAT d_print_tran_mat
#define PRINT_TO_FILE_TRAN_MAT d_print_to_file_tran_mat

#define PRINT_EXP_MAT d_print_exp_mat
#define PRINT_EXP_TRAN_MAT d_print_exp_tran_mat



#define REAL double
#define STRMAT blasfeo_dmat_ref
#define STRVEC blasfeo_dvec_ref


#define ALLOCATE_STRMAT blasfeo_allocate_dmat_ref
#define ALLOCATE_STRVEC blasfeo_allocate_dvec_ref

#define FREE_STRMAT blasfeo_free_dmat_ref
#define FREE_STRVEC blasfeo_free_dvec_ref

#define PRINT_STRMAT blasfeo_print_dmat_ref
#define PRINT_TRAN_STRMAT blasfeo_print_tran_dmat_ref
#define PRINT_STRVEC blasfeo_print_dvec_ref
#define PRINT_TRAN_STRVEC blasfeo_print_tran_dvec_ref

#define PRINT_TO_FILE_STRMAT blasfeo_print_to_file_dmat_ref
#define PRINT_TO_FILE_EXP_STRMAT blasfeo_print_to_file_exp_dmat_ref
#define PRINT_TO_FILE_STRVEC blasfeo_print_to_file_dvec_ref
#define PRINT_TO_FILE_TRAN_STRVEC blasfeo_d_print_to_file_tran_strvec_ref
#define PRINT_TO_STRING_STRMAT blasfeo_print_to_string_dmat_ref
#define PRINT_TO_STRING_STRVEC blasfeo_print_to_string_dvec_ref
#define PRINT_TO_STRING_TRAN_STRVEC blasfeo_print_to_string_tran_dvec_ref

#define PRINT_EXP_STRMAT blasfeo_print_exp_dmat_ref
#define PRINT_EXP_STRVEC blasfeo_print_exp_dvec_ref
#define PRINT_EXP_TRAN_STRVEC blasfeo_print_exp_tran_dvec_ref

#include "x_aux_ext_dep_lib0.c"

