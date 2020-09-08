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



void SYRK(char *uplo, char *trans, int *pm, int *pk, REAL *palpha, REAL *A, int *plda, REAL *pbeta, REAL *C, int *pldc)
	{

#if defined(DIM_CHECK)
	if( !(*uplo=='l' | *uplo=='L' | *uplo=='u' | *uplo=='U') )
		{
		printf("\nBLASFEO: syrk: wrong value for uplo\n");
		return;
		}
	if( !(*trans=='c' | *trans=='C' | *trans=='n' | *trans=='N' | *trans=='t' | *trans=='T') )
		{
		printf("\nBLASFEO: syrk: wrong value for trans\n");
		return;
		}
#endif

#if defined(FALLBACK_TO_EXTERNAL_BLAS)
	// TODO
#endif

	struct MAT sA;
	sA.pA = A;
	sA.m = *plda;

	struct MAT sC;
	sC.pA = C;
	sC.m = *pldc;

	if(*uplo=='l' | *uplo=='L')
		{
		if(*trans=='n' | *trans=='N')
			{
//#if ( defined(LA_HIGH_PERFORMANCE) & defined(MF_COLMAJ) & defined(DOUBLE_PRECISION) )
#if ( defined(LA_HIGH_PERFORMANCE) & defined(DOUBLE_PRECISION) )
			SYRK3_LN(*pm, *pk, *palpha, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#else
			SYRK_LN(*pm, *pk, *palpha, &sA, 0, 0, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#endif
			}
		else
			{
//#if ( defined(LA_HIGH_PERFORMANCE) & defined(MF_COLMAJ) & defined(DOUBLE_PRECISION) )
#if ( defined(LA_HIGH_PERFORMANCE) & defined(DOUBLE_PRECISION) )
			SYRK3_LT(*pm, *pk, *palpha, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#else
			SYRK_LT(*pm, *pk, *palpha, &sA, 0, 0, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#endif
			}
		}
	else
		{
		if(*trans=='n' | *trans=='N')
			{
//#if ( defined(LA_HIGH_PERFORMANCE) & defined(MF_COLMAJ) & defined(DOUBLE_PRECISION) )
#if ( defined(LA_HIGH_PERFORMANCE) & defined(DOUBLE_PRECISION) )
			SYRK3_UN(*pm, *pk, *palpha, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#else
			SYRK_UN(*pm, *pk, *palpha, &sA, 0, 0, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#endif
			}
		else
			{
//#if ( defined(LA_HIGH_PERFORMANCE) & defined(MF_COLMAJ) & defined(DOUBLE_PRECISION) )
#if ( defined(LA_HIGH_PERFORMANCE) & defined(DOUBLE_PRECISION) )
			SYRK3_UT(*pm, *pk, *palpha, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#else
			SYRK_UT(*pm, *pk, *palpha, &sA, 0, 0, &sA, 0, 0, *pbeta, &sC, 0, 0, &sC, 0, 0);
#endif
			}
		}

	return;

	}

