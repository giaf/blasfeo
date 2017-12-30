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

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_d_aux_ext_dep.h"


#define ZEROS d_zeros
#define ZEROS_ALIGN d_zeros_align

#define FREE d_free
#define FREE_ALIGN d_free_align

#define PRINT_MAT d_print_mat
#define PRINT_TO_FILE_MAT d_print_to_file_mat

#define PRINT_TRAN_MAT d_print_tran_mat
#define PRINT_TO_FILE_TRAN_MAT d_print_to_file_tran_mat

#define PRINT_E_MAT d_print_e_mat
#define PRINT_E_TRAN_MAT d_print_e_tran_mat



#define REAL double
#define STRMAT blasfeo_dmat_ref
#define STRVEC blasfeo_dvec_ref


#define ALLOCATE_STRMAT blasfeo_blasfeo_allocate_dmat_ref
#define ALLOCATE_STRVEC blasfeo_blasfeo_allocate_dvec_ref

#define FREE_STRMAT blasfeo_blasfeo_free_dmat_ref
#define FREE_STRVEC blasfeo_blasfeo_free_dvec_ref

#define PRINT_STRMAT blasfeo_d_print_strmat_ref
#define PRINT_STRVEC blasfeo_d_print_strvec_ref
#define PRINT_TRAN_STRVEC blasfeo_d_print_tran_strvec_ref

#define PRINT_TO_FILE_STRMAT blasfeo_d_print_to_file_strmat_ref
#define PRINT_TO_FILE_STRVEC blasfeo_d_print_to_file_strvec_ref
#define PRINT_TO_FILE_TRAN_STRVEC blasfeo_d_print_to_file_tran_strvec_ref

#define PRINT_E_STRMAT blasfeo_d_print_e_strmat_ref
#define PRINT_E_STRVEC blasfeo_d_print_e_strvec_ref
#define PRINT_E_TRAN_STRVEC blasfeo_d_print_e_tran_strvec_ref

#include "x_aux_ext_dep_lib0.c"

