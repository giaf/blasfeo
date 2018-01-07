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

#include "../include/blasfeo_block_size.h"


#define REAL float
#define STRMAT blasfeo_smat
#define STRVEC blasfeo_svec
#define PS S_PS


#define ZEROS s_zeros
#define ZEROS_ALIGN s_zeros_align

#define FREE s_free
#define FREE_ALIGN s_free_align

#define PRINT_MAT s_print_mat
#define PRINT_TO_FILE_MAT s_print_to_file_mat

#define PRINT_TRAN_MAT s_print_tran_mat
#define PRINT_TO_FILE_TRAN_MAT s_print_to_file_tran_mat

#define PRINT_E_MAT s_print_e_mat
#define PRINT_E_TRAN_MAT s_print_e_tran_mat

#include "x_aux_ext_dep_lib.c"


#if defined(LA_BLAS) | defined(LA_REFERENCE)


#define ALLOCATE_STRMAT blasfeo_allocate_smat
#define ALLOCATE_STRVEC blasfeo_allocate_svec

#define FREE_STRMAT blasfeo_free_smat
#define FREE_STRVEC blasfeo_free_svec

#define PRINT_STRMAT blasfeo_print_smat
#define PRINT_STRVEC blasfeo_print_svec
#define PRINT_TRAN_STRVEC blasfeo_print_tran_svec

#define PRINT_TO_FILE_STRMAT blasfeo_print_to_file_smat
#define PRINT_TO_FILE_STRVEC blasfeo_print_to_file_svec
#define PRINT_TO_FILE_TRAN_STRVEC s_print_to_file_tran_strvec

#define PRINT_E_STRMAT blasfeo_print_exp_smat
#define PRINT_E_STRVEC blasfeo_print_exp_svec
#define PRINT_E_TRAN_STRVEC blasfeo_print_exp_tran_svec


#include "x_aux_ext_dep_lib0.c"

#endif

