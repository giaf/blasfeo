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

#if defined(TESTING_MODE)

// libc
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#ifndef ROUTINE
	#define ROUTINE blasfeo_dgemm_nn
	#define ROUTINE_CLASS_GEMM 1
#endif

// tests helpers
#include "test_d_common.h"
#include "test_x_common.h"
#include "test_x_common.c"

// template base on routine class
#ifdef ROUTINE_CLASS_GETRF
#include "test_class_getrf.c"
#endif
#ifdef ROUTINE_CLASS_GEMM
#include "test_class_gemm.c"
#endif
#ifdef ROUTINE_CLASS_SYRK
#include "test_class_syrk.c"
#endif
#ifdef ROUTINE_CLASS_TRM
#include "test_class_trm.c"
#endif

#include "test_x.c"

#else

#include <stdio.h>

int main()
	{
	printf("\n\n Recompile BLASFEO with TESTING_MODE=1 to run this test.\n");
	printf("On CMake use -DBLASFEO_TESTING=ON .\n\n");
	return 0;
	}

#endif
