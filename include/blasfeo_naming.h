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


/*
 * ----------- Naming conventions
 *
 *  (precision)(data)
 *
 * 1) d(double)
 *    s(single)
 *
 * 2) ge(general)
 *    tr(triangular)
 *    vec(vector)
 *    row(row)
 *    col(column)
 *    dia(diagonal)
 *
 * 3) se(set)
 *    cp(copy)
 *    sc(scale)
 *    ad(add)
 *    tr(transpose)
 *    in(insert)
 *    ex(extract)
 *    pe(premute)
 *    sw(swap)
 *
 *    f(factorization)
 *
 *    lqf(LQ factorization)
 *    qrf (factorization)
 *    trf (LU factorization using partial pivoting with row interchanges.)
 *
 * 4) _l(lower) / _u(upper)
 *    _lib8 (hp implementation, 8 rows kernel)
 *    _lib4 (hp implementation, 4 rows kernel)
 *    _lib0 (hp interface with reference implentation)
 *    _lib (reference implementation)
 *    _libref (reference implementation with dedicated namespace)
 *
 * 5) _sp(sparse)
 *    _exp(exponential format)
 */
