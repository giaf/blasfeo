/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS for embedded optimization.                                                      *
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

#define blasfeo_hp_dgemm_nt_m2 blasfeo_hp_zen5_dgemm_nt_m2
#define blasfeo_hp_dgemm_nt_n2 blasfeo_hp_zen5_dgemm_nt_n2

#if ( defined(BLAS_API) & defined(MF_PANELMAJ) )

#define blasfeo_hp_cm_dgemm_nn blasfeo_hp_cm_zen5_dgemm_nn
#define blasfeo_hp_cm_dgemm_nt blasfeo_hp_cm_zen5_dgemm_nt
#define blasfeo_hp_cm_dgemm_tn blasfeo_hp_cm_zen5_dgemm_tn
#define blasfeo_hp_cm_dgemm_tt blasfeo_hp_cm_zen5_dgemm_tt
#define blasfeo_cm_dgemm_nn blasfeo_cm_zen5_dgemm_nn
#define blasfeo_cm_dgemm_nt blasfeo_cm_zen5_dgemm_nt
#define blasfeo_cm_dgemm_tn blasfeo_cm_zen5_dgemm_tn
#define blasfeo_cm_dgemm_tt blasfeo_cm_zen5_dgemm_tt

#else

#define blasfeo_hp_dgemm_nn blasfeo_hp_zen5_dgemm_nn
#define blasfeo_hp_dgemm_nt blasfeo_hp_zen5_dgemm_nt
#define blasfeo_hp_dgemm_tn blasfeo_hp_zen5_dgemm_tn
#define blasfeo_hp_dgemm_tt blasfeo_hp_zen5_dgemm_tt
#define blasfeo_dgemm_nn blasfeo_zen5_dgemm_nn
#define blasfeo_dgemm_nt blasfeo_zen5_dgemm_nt
#define blasfeo_dgemm_tn blasfeo_zen5_dgemm_tn
#define blasfeo_dgemm_tt blasfeo_zen5_dgemm_tt

#endif

#define TARGET_X64_AMD_ZEN5
#define TARGET_X64_INTEL_SKYLAKE_X

#include "dgemm.c"
