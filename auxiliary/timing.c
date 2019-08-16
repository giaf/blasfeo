/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2016-2018 by Gianluca Frison.                                                     *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* This program is free software: you can redistribute it and/or modify                            *
* it under the terms of the GNU General Public License as published by                            *
* the Free Software Foundation, either version 3 of the License, or                               *
* (at your option) any later version                                                              *.
*                                                                                                 *
* This program is distributed in the hope that it will be useful,                                 *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                   *
* GNU General Public License for more details.                                                    *
*                                                                                                 *
* You should have received a copy of the GNU General Public License                               *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.                          *
*                                                                                                 *
* The authors designate this particular file as subject to the "Classpath" exception              *
* as provided by the authors in the LICENSE file that accompained this code.                      *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include "../include/blasfeo_timing.h"

#if (defined _WIN32 || defined _WIN64) && !(defined __MINGW32__ || defined __MINGW64__)

	void blasfeo_tic(blasfeo_timer* t) {
		QueryPerformanceFrequency(&t->freq);
		QueryPerformanceCounter(&t->tic);
	}

	double blasfeo_toc(blasfeo_timer* t) {
		QueryPerformanceCounter(&t->toc);
		return ((t->toc.QuadPart - t->tic.QuadPart) / (double)t->freq.QuadPart);
	}

#elif(defined __APPLE__)
	void blasfeo_tic(blasfeo_timer* t) {
		/* read current clock cycles */
		t->tic = mach_absolute_time();
	}

	double blasfeo_toc(blasfeo_timer* t) {
		uint64_t duration; /* elapsed time in clock cycles*/

		t->toc = mach_absolute_time();
		duration = t->toc - t->tic;

		/*conversion from clock cycles to nanoseconds*/
		mach_timebase_info(&(t->tinfo));
		duration *= t->tinfo.numer;
		duration /= t->tinfo.denom;

		return (double)duration / 1e9;
	}

#elif(defined __DSPACE__)

	void blasfeo_tic(blasfeo_timer* t) {
		ds1401_tic_start();
		t->time = ds1401_tic_read();
	}

	double blasfeo_toc(blasfeo_timer* t) {
		return ds1401_tic_read() - t->time;
	}

#elif defined(__XILINX_NONE_ELF__)

	void blasfeo_tic(blasfeo_timer* t) {
		XTime_GetTime(&(t->tic));
	}

	double blasfeo_toc(blasfeo_timer* t) {
		uint64_t toc;
		XTime_GetTime(&toc);
		t->toc = toc;

		/* time in s */
		return (double) (toc - t->tic) / (COUNTS_PER_SECOND);  
	}
#else

	#if __STDC_VERSION__ >= 199901L  // C99 Mode

		/* read current time */
		void blasfeo_tic(blasfeo_timer* t) {
			gettimeofday(&t->tic, 0);
		}

		/* return time passed since last call to tic on this timer */
		double blasfeo_toc(blasfeo_timer* t) {
			struct timeval temp;

			gettimeofday(&t->toc, 0);

			if ((t->toc.tv_usec - t->tic.tv_usec) < 0) {
				temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec - 1;
				temp.tv_usec = 1000000 + t->toc.tv_usec - t->tic.tv_usec;
			} else {
				temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
				temp.tv_usec = t->toc.tv_usec - t->tic.tv_usec;
			}

			return (double)temp.tv_sec + (double)temp.tv_usec / 1e6;
		}

	#else  // ANSI C Mode

		/* read current time */
		void blasfeo_tic(blasfeo_timer* t) {
			clock_gettime(CLOCK_MONOTONIC, &t->tic);
		}


		/* return time passed since last call to tic on this timer */
		double blasfeo_toc(blasfeo_timer* t) {
			struct timespec temp;

			clock_gettime(CLOCK_MONOTONIC, &t->toc);

			if ((t->toc.tv_nsec - t->tic.tv_nsec) < 0) {
				temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec - 1;
				temp.tv_nsec = 1000000000+t->toc.tv_nsec - t->tic.tv_nsec;
			} else {
				temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
				temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
			}

			return (double)temp.tv_sec + (double)temp.tv_nsec / 1e9;
		}

	#endif  // __STDC_VERSION__ >= 199901L

#endif  // (defined _WIN32 || _WIN64)
