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

#if ( defined(BLAS_API) & defined(MF_PANELMAJ) )

#define blasfeo_hp_cm_dsyrk3_ln blasfeo_hp_cm_zen5_dsyrk3_ln
#define blasfeo_hp_cm_dsyrk3_lt blasfeo_hp_cm_zen5_dsyrk3_lt
#define blasfeo_hp_cm_dsyrk3_un blasfeo_hp_cm_zen5_dsyrk3_un
#define blasfeo_hp_cm_dsyrk3_ut blasfeo_hp_cm_zen5_dsyrk3_ut
#define blasfeo_hp_cm_dsyrk_ln blasfeo_hp_cm_zen5_dsyrk_ln
#define blasfeo_hp_cm_dsyrk_ln_mn blasfeo_hp_cm_zen5_dsyrk_ln_mn
#define blasfeo_hp_cm_dsyrk_lt blasfeo_hp_cm_zen5_dsyrk_lt
#define blasfeo_hp_cm_dsyrk_un blasfeo_hp_cm_zen5_dsyrk_un
#define blasfeo_hp_cm_dsyrk_ut blasfeo_hp_cm_zen5_dsyrk_ut
#define blasfeo_cm_dsyrk3_ln blasfeo_cm_zen5_dsyrk3_ln
#define blasfeo_cm_dsyrk3_lt blasfeo_cm_zen5_dsyrk3_lt
#define blasfeo_cm_dsyrk3_un blasfeo_cm_zen5_dsyrk3_un
#define blasfeo_cm_dsyrk3_ut blasfeo_cm_zen5_dsyrk3_ut
#define blasfeo_cm_dsyrk_ln blasfeo_cm_zen5_dsyrk_ln
#define blasfeo_cm_dsyrk_ln_mn blasfeo_cm_zen5_dsyrk_ln_mn
#define blasfeo_cm_dsyrk_lt blasfeo_cm_zen5_dsyrk_lt
#define blasfeo_cm_dsyrk_un blasfeo_cm_zen5_dsyrk_un
#define blasfeo_cm_dsyrk_ut blasfeo_cm_zen5_dsyrk_ut

#else

#define blasfeo_hp_dsyrk3_ln blasfeo_hp_zen5_dsyrk3_ln
#define blasfeo_hp_dsyrk3_lt blasfeo_hp_zen5_dsyrk3_lt
#define blasfeo_hp_dsyrk3_un blasfeo_hp_zen5_dsyrk3_un
#define blasfeo_hp_dsyrk3_ut blasfeo_hp_zen5_dsyrk3_ut
#define blasfeo_hp_dsyrk_ln blasfeo_hp_zen5_dsyrk_ln
#define blasfeo_hp_dsyrk_ln_mn blasfeo_hp_zen5_dsyrk_ln_mn
#define blasfeo_hp_dsyrk_lt blasfeo_hp_zen5_dsyrk_lt
#define blasfeo_hp_dsyrk_un blasfeo_hp_zen5_dsyrk_un
#define blasfeo_hp_dsyrk_ut blasfeo_hp_zen5_dsyrk_ut
#define blasfeo_dsyrk3_ln blasfeo_zen5_dsyrk3_ln
#define blasfeo_dsyrk3_lt blasfeo_zen5_dsyrk3_lt
#define blasfeo_dsyrk3_un blasfeo_zen5_dsyrk3_un
#define blasfeo_dsyrk3_ut blasfeo_zen5_dsyrk3_ut
#define blasfeo_dsyrk_ln blasfeo_zen5_dsyrk_ln
#define blasfeo_dsyrk_ln_mn blasfeo_zen5_dsyrk_ln_mn
#define blasfeo_dsyrk_lt blasfeo_zen5_dsyrk_lt
#define blasfeo_dsyrk_un blasfeo_zen5_dsyrk_un
#define blasfeo_dsyrk_ut blasfeo_zen5_dsyrk_ut

#endif

#define TARGET_X64_AMD_ZEN5
#define TARGET_X64_INTEL_SKYLAKE_X

#include "dsyrk.c"

