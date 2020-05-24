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



#define REAL float
#define MAT blasfeo_smat
#define MATEL BLASFEO_SMATEL
#define VEC blasfeo_svec
#define VECEL BLASFEO_SVECEL



#define MEMSIZE_MAT blasfeo_memsize_smat
#define MEMSIZE_DIAG_MAT blasfeo_memsize_diag_smat
#define MEMSIZE_VEC blasfeo_memsize_svec
#define CREATE_MAT blasfeo_create_smat
#define CREATE_VEC blasfeo_create_svec
#define PACK_MAT blasfeo_pack_smat
#define PACK_TRAN_MAT blasfeo_pack_tran_smat
#define PACK_VEC blasfeo_pack_svec
#define UNPACK_MAT blasfeo_unpack_smat
#define UNPACK_TRAN_MAT blasfeo_unpack_tran_smat
#define UNPACK_VEC blasfeo_unpack_svec
#define CAST_MAT2STRMAT s_cast_mat2strmat
#define CAST_DIAG_MAT2STRMAT s_cast_diag_mat2strmat
#define CAST_VEC2VECMAT s_cast_vec2vecmat
#define GEAD blasfeo_sgead
#define GECP blasfeo_sgecp
#define GECPSC blasfeo_sgecpsc
#define GESC blasfeo_sgesc
#define GESE blasfeo_sgese

#define GETR blasfeo_sgetr
#define GEIN1 blasfeo_sgein1
#define GEEX1 blasfeo_sgeex1
#define TRCP_L blasfeo_strcp_l
#define TRTR_L blasfeo_strtr_l
#define TRTR_U blasfeo_strtr_u
#define VECSE blasfeo_svecse
#define VECCP blasfeo_sveccp
#define VECSC blasfeo_svecsc
#define VECCPSC blasfeo_sveccpsc
#define VECAD blasfeo_svecad
#define VECAD_SP blasfeo_svecad_sp
#define VECIN_SP blasfeo_svecin_sp
#define VECEX_SP blasfeo_svecex_sp
#define VECIN1 blasfeo_svecin1
#define VECEX1 blasfeo_svecex1
#define VECPE blasfeo_svecpe
#define VECPEI blasfeo_svecpei
#define VECCL blasfeo_sveccl
#define VECCL_MASK blasfeo_sveccl_mask
#define VECZE blasfeo_svecze
#define VECNRM_INF blasfeo_svecnrm_inf
#define DIAIN blasfeo_sdiain
#define DIAIN_SP blasfeo_sdiain_sp
#define DIAEX blasfeo_sdiaex
#define DIAEX_SP blasfeo_sdiaex_sp
#define DIAAD blasfeo_sdiaad
#define DIAAD_SP blasfeo_sdiaad_sp
#define DIAADIN_SP blasfeo_sdiaadin_sp
#define DIARE blasfeo_sdiare
#define ROWEX blasfeo_srowex
#define ROWIN blasfeo_srowin
#define ROWAD blasfeo_srowad
#define ROWAD_SP blasfeo_srowad_sp
#define ROWSW blasfeo_srowsw
#define ROWPE blasfeo_srowpe
#define ROWPEI blasfeo_srowpei
#define COLEX blasfeo_scolex
#define COLIN blasfeo_scolin
#define COLAD blasfeo_scolad
#define COLSC blasfeo_scolsc
#define COLSW blasfeo_scolsw
#define COLPE blasfeo_scolpe
#define COLPEI blasfeo_scolpei



// LA_REFERENCE | LA_EXTERNAL_BLAS_WRAPPER
#include "x_aux_lib.c"
