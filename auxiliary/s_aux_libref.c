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
#define MATEL MATEL_REF
#define VEC blasfeo_svec_ref
#define VECEL VECEL_REF



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

#define GETR blasfeo_sgetr_ref
#define GEIN1 blasfeo_sgein1_ref
#define GEEX1 blasfeo_sgeex1_ref
#define TRCP_L blasfeo_strcp_l_ref
#define TRTR_L blasfeo_strtr_l_ref
#define TRTR_U blasfeo_strtr_u_ref
#define VECSE blasfeo_svecse_ref
#define VECCP blasfeo_sveccp_ref
#define VECSC blasfeo_svecsc_ref
#define VECCPSC blasfeo_sveccpsc_ref
#define VECAD blasfeo_svecad_ref
#define VECAD_SP blasfeo_svecad_sp_ref
#define VECIN_SP blasfeo_svecin_sp_ref
#define VECEX_SP blasfeo_svecex_sp_ref
#define VECIN1 blasfeo_svecin1_ref
#define VECEX1 blasfeo_svecex1_ref
#define VECPE blasfeo_svecpe_ref
#define VECPEI blasfeo_svecpei_ref
#define VECCL blasfeo_sveccl_ref
#define VECCL_MASK blasfeo_sveccl_mask_ref
#define VECZE blasfeo_svecze_ref
#define VECNRM_INF blasfeo_svecnrm_inf_ref
#define DIAIN blasfeo_sdiain_ref
#define DIAIN_SP blasfeo_sdiain_sp_ref
#define DIAEX blasfeo_sdiaex_ref
#define DIAEX_SP blasfeo_sdiaex_sp_ref
#define DIAAD blasfeo_sdiaad_ref
#define DIAAD_SP blasfeo_sdiaad_sp_ref
#define DIAADIN_SP blasfeo_sdiaadin_sp_ref
#define DIARE blasfeo_sdiare_ref
#define ROWEX blasfeo_srowex_ref
#define ROWIN blasfeo_srowin_ref
#define ROWAD blasfeo_srowad_ref
#define ROWAD_SP blasfeo_srowad_sp_ref
#define ROWSW blasfeo_srowsw_ref
#define ROWPE blasfeo_srowpe_ref
#define ROWPEI blasfeo_srowpei_ref
#define COLEX blasfeo_scolex_ref
#define COLIN blasfeo_scolin_ref
#define COLAD blasfeo_scolad_ref
#define COLSC blasfeo_scolsc_ref
#define COLSW blasfeo_scolsw_ref
#define COLPE blasfeo_scolpe_ref
#define COLPEI blasfeo_scolpei_ref



// TESTING_MODE
#include "x_aux_ref.c"
