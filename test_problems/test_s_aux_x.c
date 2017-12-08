#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

#include "../include/blasfeo_common.h"
#include "../include/blasfeo_s_aux_ext_dep.h"
#include "../include/blasfeo_s_aux.h"
#include "../include/blasfeo_s_aux_test.h"
#include "../include/blasfeo_i_aux_ext_dep.h"
#include "../include/blasfeo_v_aux_ext_dep.h"
#include "../include/blasfeo_s_kernel.h"
#include "../include/blasfeo_s_blas.h"

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x));

#include "test_s_common.h"
#include "test_x_common.c"


#define REAL float
#define STRMAT s_strmat
#define STRVEC s_strvec

