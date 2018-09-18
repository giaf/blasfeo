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

#ifndef BLASFEO_M_AUX_H_
#define BLASFEO_M_AUX_H_

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif



void blasfeo_cvt_d2s_vec(int m, struct blasfeo_dvec *vd, int vdi, struct blasfeo_svec *vs, int vsi);
void blasfeo_cvt_s2d_vec(int m, struct blasfeo_svec *vs, int vsi, struct blasfeo_dvec *vd, int vdi);
void blasfeo_cvt_d2s_mat(int m, int n, struct blasfeo_dmat *Md, int mid, int nid, struct blasfeo_smat *Ms, int mis, int nis);
void blasfeo_cvt_s2d_mat(int m, int n, struct blasfeo_smat *Ms, int mis, int nis, struct blasfeo_dmat *Md, int mid, int nid);


#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_M_AUX_H_
