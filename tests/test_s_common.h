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

// BLASFEO routines
#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_blas.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_kernel.h"

#include "../include/blasfeo_s_blas_api.h"

// BLASFEO External dependencies
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_timing.h"

// BLASFEO LA:REFERENCE routines
#include "../include/blasfeo_s_blasfeo_api_ref.h"
#include "../include/blasfeo_s_aux_ref.h"
#include "../include/blasfeo_s_aux_ext_dep_ref.h"

#include "../include/blasfeo_s_aux_test.h"
#include "../include/s_blas.h"

#define PRECISION Single
#define GECMP_LIBSTR sgecmp_libstr
#define GECMP_BLASAPI sgecmp_blasapi
#define REAL float

#define ZEROS s_zeros
#define FREE s_free

#define STRMAT blasfeo_smat
#define STRVEC blasfeo_svec

#define ALLOCATE_STRMAT blasfeo_allocate_smat
#define FREE_STRMAT blasfeo_free_smat
#define GESE_LIBSTR blasfeo_sgese
#define PACK_STRMAT blasfeo_pack_smat
#define PRINT_STRMAT blasfeo_print_smat

#define PS S_PS

#define STRMAT_REF blasfeo_smat_ref
#define STRVEC_REF blasfeo_svec_ref

#define ALLOCATE_STRMAT_REF blasfeo_allocate_smat_ref
#define FREE_STRMAT_REF blasfeo_free_smat_ref
#define GESE_REF blasfeo_sgese_ref
#define GECP_REF blasfeo_sgecp_ref
#define PACK_STRMAT_REF blasfeo_pack_smat_ref
#define PRINT_STRMAT_REF blasfeo_print_smat_ref
