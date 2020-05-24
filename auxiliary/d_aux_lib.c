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



#define MF_COLMAJ



#include <blasfeo_common.h>



#if defined(MF_COLMAJ)
	#define XMATEL_A(X, Y) pA[(X)+lda*(Y)]
	#define XMATEL_B(X, Y) pB[(X)+ldb*(Y)]
#else
	#define XMATEL_A(X, Y) XMATEL(sA, X, Y)
	#define XMATEL_B(X, Y) XMATEL(sB, X, Y)
#endif



#define REAL double
#define MAT blasfeo_dmat
#define MATEL BLASFEO_DMATEL
#define VEC blasfeo_dvec
#define VECEL BLASFEO_DVECEL



#define MEMSIZE_MAT blasfeo_memsize_dmat
#define MEMSIZE_DIAG_MAT blasfeo_memsize_diag_dmat
#define MEMSIZE_VEC blasfeo_memsize_dvec
#define CREATE_MAT blasfeo_create_dmat
#define CREATE_VEC blasfeo_create_dvec
#define PACK_MAT blasfeo_pack_dmat
#define PACK_TRAN_MAT blasfeo_pack_tran_dmat
#define PACK_VEC blasfeo_pack_dvec
#define UNPACK_MAT blasfeo_unpack_dmat
#define UNPACK_TRAN_MAT blasfeo_unpack_tran_dmat
#define UNPACK_VEC blasfeo_unpack_dvec
#define CAST_MAT2STRMAT d_cast_mat2strmat
#define CAST_DIAG_MAT2STRMAT d_cast_diag_mat2strmat
#define CAST_VEC2VECMAT d_cast_vec2vecmat
#define GEAD blasfeo_dgead
#define GECP blasfeo_dgecp
#define GECPSC blasfeo_dgecpsc
#define GESC blasfeo_dgesc
#define GESE blasfeo_dgese

#define GETR blasfeo_dgetr
#define GEIN1 blasfeo_dgein1
#define GEEX1 blasfeo_dgeex1
#define TRCP_L blasfeo_dtrcp_l
#define TRTR_L blasfeo_dtrtr_l
#define TRTR_U blasfeo_dtrtr_u
#define VECSE blasfeo_dvecse
#define VECCP blasfeo_dveccp
#define VECSC blasfeo_dvecsc
#define VECCPSC blasfeo_dveccpsc
#define VECAD blasfeo_dvecad
#define VECAD_SP blasfeo_dvecad_sp
#define VECIN_SP blasfeo_dvecin_sp
#define VECEX_SP blasfeo_dvecex_sp
#define VECIN1 blasfeo_dvecin1
#define VECEX1 blasfeo_dvecex1
#define VECPE blasfeo_dvecpe
#define VECPEI blasfeo_dvecpei
#define VECCL blasfeo_dveccl
#define VECCL_MASK blasfeo_dveccl_mask
#define VECZE blasfeo_dvecze
#define VECNRM_INF blasfeo_dvecnrm_inf
#define DIAIN blasfeo_ddiain
#define DIAIN_SP blasfeo_ddiain_sp
#define DIAEX blasfeo_ddiaex
#define DIAEX_SP blasfeo_ddiaex_sp
#define DIAAD blasfeo_ddiaad
#define DIAAD_SP blasfeo_ddiaad_sp
#define DIAADIN_SP blasfeo_ddiaadin_sp
#define DIARE blasfeo_ddiare
#define ROWEX blasfeo_drowex
#define ROWIN blasfeo_drowin
#define ROWAD blasfeo_drowad
#define ROWAD_SP blasfeo_drowad_sp
#define ROWSW blasfeo_drowsw
#define ROWPE blasfeo_drowpe
#define ROWPEI blasfeo_drowpei
#define COLEX blasfeo_dcolex
#define COLIN blasfeo_dcolin
#define COLAD blasfeo_dcolad
#define COLSC blasfeo_dcolsc
#define COLSW blasfeo_dcolsw
#define COLPE blasfeo_dcolpe
#define COLPEI blasfeo_dcolpei



// LA_REFERENCE | LA_EXTERNAL_BLAS_WRAPPER
#include "x_aux_lib.c"
