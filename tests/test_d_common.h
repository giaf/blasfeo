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
#include "../include/blasfeo_d_blas.h"
#include "../include/blasfeo_d_aux.h"
#include "../include/blasfeo_d_kernel.h"

// BLASFEO External dependencies
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_d_aux_ext_dep.h"
#include "../include/blasfeo_timing.h"

// BLASFEO LA:REFERENCE routines
#include "../include/blasfeo_d_aux_ref.h"
#include "../include/blasfeo_d_aux_ext_dep_ref.h"
#include "../include/blasfeo_d_blas3_ref.h"

#include "../include/blasfeo_d_aux_test.h"


#define PRECISION Double
#define GECMP_LIBSTR dgecmp_libstr
#define REAL double

#define ZEROS d_zeros
#define FREE d_free

#define STRMAT blasfeo_dmat
#define STRVEC blasfeo_dvec

#define ALLOCATE_STRMAT blasfeo_allocate_dmat
#define PACK_STRMAT blasfeo_pack_dmat
#define PRINT_STRMAT blasfeo_print_dmat
#define FREE_STRMAT blasfeo_free_dmat

#define PS D_PS


#define STRMAT_REF blasfeo_dmat_ref
#define STRVEC_REF blasfeo_dvec_ref

#define ALLOCATE_STRMAT_REF blasfeo_allocate_dmat_ref
#define PACK_STRMAT_REF blasfeo_pack_dmat_ref
#define PRINT_STRMAT_REF blasfeo_print_dmat_ref
#define FREE_STRMAT_REF blasfeo_free_dmat_ref


#define PRINT_STRMAT_REF blasfeo_print_dmat_ref

