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



#define REAL double
#define MAT blasfeo_dmat_ref
#define MATEL MATEL_REF
#define VEC blasfeo_dvec_ref
#define VECEL VECEL_REF



#define MEMSIZE_MAT blasfeo_memsize_dmat_ref
#define MEMSIZE_DIAG_MAT blasfeo_memsize_diag_dmat_ref
#define MEMSIZE_VEC blasfeo_memsize_dvec_ref
#define CREATE_MAT blasfeo_create_dmat_ref
#define CREATE_VEC blasfeo_create_dvec_ref
#define PACK_MAT blasfeo_pack_dmat_ref
#define PACK_TRAN_MAT blasfeo_pack_tran_dmat_ref
#define PACK_VEC blasfeo_pack_dvec_ref
#define UNPACK_MAT blasfeo_unpack_dmat_ref
#define UNPACK_TRAN_MAT blasfeo_unpack_tran_dmat_ref
#define UNPACK_VEC blasfeo_unpack_dvec_ref
#define CAST_MAT2STRMAT blasfeo_d_cast_mat2strmat_ref
#define CAST_DIAG_MAT2STRMAT blasfeo_d_cast_diag_mat2strmat_ref
#define CAST_VEC2VECMAT blasfeo_d_cast_vec2vecmat_ref
#define GEAD blasfeo_dgead_ref
#define GECP blasfeo_dgecp_ref
#define GECPSC blasfeo_dgecpsc_ref
#define GESC blasfeo_dgesc_ref
#define GESE blasfeo_dgese_ref

#define GETR blasfeo_dgetr_ref
#define GEIN1 blasfeo_dgein1_ref
#define GEEX1 blasfeo_dgeex1_ref
#define TRCP_L blasfeo_dtrcp_l_ref
#define TRTR_L blasfeo_dtrtr_l_ref
#define TRTR_U blasfeo_dtrtr_u_ref
#define VECSE blasfeo_dvecse_ref
#define VECCP blasfeo_dveccp_ref
#define VECSC blasfeo_dvecsc_ref
#define VECCPSC blasfeo_dveccpsc_ref
#define VECAD blasfeo_dvecad_ref
#define VECAD_SP blasfeo_dvecad_sp_ref
#define VECIN_SP blasfeo_dvecin_sp_ref
#define VECEX_SP blasfeo_dvecex_sp_ref
#define VECIN1 blasfeo_dvecin1_ref
#define VECEX1 blasfeo_dvecex1_ref
#define VECPE blasfeo_dvecpe_ref
#define VECPEI blasfeo_dvecpei_ref
#define VECCL blasfeo_dveccl_ref
#define VECCL_MASK blasfeo_dveccl_mask_ref
#define VECZE blasfeo_dvecze_ref
#define VECNRM_INF blasfeo_dvecnrm_inf_ref
#define DIAIN blasfeo_ddiain_ref
#define DIAIN_SP blasfeo_ddiain_sp_ref
#define DIAEX blasfeo_ddiaex_ref
#define DIAEX_SP blasfeo_ddiaex_sp_ref
#define DIAAD blasfeo_ddiaad_ref
#define DIAAD_SP blasfeo_ddiaad_sp_ref
#define DIAADIN_SP blasfeo_ddiaadin_sp_ref
#define DIARE blasfeo_ddiare_ref
#define ROWEX blasfeo_drowex_ref
#define ROWIN blasfeo_drowin_ref
#define ROWAD blasfeo_drowad_ref
#define ROWAD_SP blasfeo_drowad_sp_ref
#define ROWSW blasfeo_drowsw_ref
#define ROWPE blasfeo_drowpe_ref
#define ROWPEI blasfeo_drowpei_ref
#define COLEX blasfeo_dcolex_ref
#define COLIN blasfeo_dcolin_ref
#define COLAD blasfeo_dcolad_ref
#define COLSC blasfeo_dcolsc_ref
#define COLSW blasfeo_dcolsw_ref
#define COLPE blasfeo_dcolpe_ref
#define COLPEI blasfeo_dcolpei_ref



// TESTING_MODE
#include "x_aux_ref.c"
