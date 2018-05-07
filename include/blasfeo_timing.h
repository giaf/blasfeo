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

#ifndef BLASFEO_TIMING_H_
#define BLASFEO_TIMING_H_

#include <stdbool.h>

#if (defined _WIN32 || defined _WIN64) && !(defined __MINGW32__ || defined __MINGW64__)

	/* Use Windows QueryPerformanceCounter for timing. */
	#include <Windows.h>

	/** A structure for keeping internal timer data. */
	typedef struct blasfeo_timer_ {
		LARGE_INTEGER tic;
		LARGE_INTEGER toc;
		LARGE_INTEGER freq;
	} blasfeo_timer;

#elif(defined __APPLE__)

	#include <mach/mach_time.h>

	/** A structure for keeping internal timer data. */
	typedef struct blasfeo_timer_ {
		uint64_t tic;
		uint64_t toc;
		mach_timebase_info_data_t tinfo;
	} blasfeo_timer;

#elif(defined __DSPACE__)

	#include <brtenv.h>

	typedef struct blasfeo_timer_ {
		double time;
	} blasfeo_timer;

#else

	/* Use POSIX clock_gettime() for timing on non-Windows machines. */
	#include <time.h>

	#if __STDC_VERSION__ >= 199901L  // C99 Mode

		#include <sys/stat.h>
		#include <sys/time.h>

		typedef struct blasfeo_timer_ {
			struct timeval tic;
			struct timeval toc;
		} blasfeo_timer;

	#else  // ANSI C Mode

		/** A structure for keeping internal timer data. */
		typedef struct blasfeo_timer_ {
			struct timespec tic;
			struct timespec toc;
		} blasfeo_timer;

	#endif  // __STDC_VERSION__ >= 199901L

#endif  // (defined _WIN32 || defined _WIN64)

/** A function for measurement of the current time. */
void blasfeo_tic(blasfeo_timer* t);

/** A function which returns the elapsed time. */
double blasfeo_toc(blasfeo_timer* t);

#endif  // BLASFEO_TIMING_H_
