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

/*
 * auxiliary functions for LA:REFERENCE (column major)
 *
 * auxiliary/d_aux_lib*.c
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <blasfeo_common.h>


#define MF_COLMAJ



#if defined(MF_COLMAJ)
	#define XMATEL_A(X, Y) pA[(X)+lda*(Y)]
	#define XMATEL_B(X, Y) pB[(X)+ldb*(Y)]
#else
	#define XMATEL_A(X, Y) XMATEL(sA, X, Y)
	#define XMATEL_B(X, Y) XMATEL(sB, X, Y)
#endif



#define REAL float
#define MAT blasfeo_smat_ref
#define MATEL BLASFEO_SMATEL
#define VEC blasfeo_svec_ref
#define VECEL BLASFEO_SVECEL



#define MEMSIZE_MAT blasfeo_memsize_smat_ref
#define MEMSIZE_DIAG_MAT blasfeo_memsize_diag_smat_ref
#define MEMSIZE_VEC blasfeo_memsize_svec_ref
#define CREATE_MAT blasfeo_create_smat_ref
#define CREATE_VEC blasfeo_create_svec_ref
#define PACK_MAT blasfeo_pack_smat_ref
#define PACK_TRAN_MAT blasfeo_pack_tran_smat_ref
#define PACK_VEC blasfeo_pack_svec_ref
#define UNPACK_MAT blasfeo_unpack_smat_ref
#define UNPACK_TRAN_MAT blasfeo_unpack_tran_smat_ref
#define UNPACK_VEC blasfeo_unpack_svec_ref
#define CAST_MAT2STRMAT blasfeo_s_cast_mat2strmat_ref
#define CAST_DIAG_MAT2STRMAT blasfeo_s_cast_diag_mat2strmat_ref
#define CAST_VEC2VECMAT blasfeo_s_cast_vec2vecmat_ref
#define GEAD blasfeo_sgead_ref
#define GECP blasfeo_sgecp_ref
#define GECPSC blasfeo_sgecpsc_ref
#define GESC blasfeo_sgesc_ref
#define GESE blasfeo_sgese_ref



// TESTING_MODE
#include "x_aux_lib.c"
