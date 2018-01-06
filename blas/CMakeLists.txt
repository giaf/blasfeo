

set(BLAS_SRC
    d_blas2_diag_lib.c
    s_blas2_diag_lib.c)

if(${BLASFEO_LA_MODE} MATCHES HIGH_PERFORMANCE)

    list(APPEND BLAS_SRC
        d_blas1_lib4.c
        d_blas2_lib4.c
        d_blas3_lib4.c
        d_blas3_diag_lib4.c
        d_lapack_lib4.c)

	if(${BLASFEO_TARGET} MATCHES X64_INTEL_HASWELL OR
	   ${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

		list(APPEND BLAS_SRC
			s_blas1_lib8.c
			s_blas2_lib8.c
			s_blas3_lib8.c
			s_blas3_diag_lib8.c
			s_lapack_lib8.c)

	elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE OR
		   ${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER OR
		   ${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57 OR
		   ${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15 OR
		   ${BLASFEO_TARGET} MATCHES GENERIC)

		list(APPEND BLAS_SRC
			s_blas1_lib4.c
			s_blas2_lib4.c
			s_blas3_lib4.c
			s_blas3_diag_lib4.c
			s_lapack_lib4.c)

	endif()

else() # Reference BLAS

	list(APPEND BLAS_SRC
		d_blas1_lib.c
		d_blas2_lib.c
		d_blas3_lib.c
		d_blas3_diag_lib.c
		d_lapack_lib.c
		s_blas1_lib.c
		s_blas2_lib.c
		s_blas3_lib.c
		s_blas3_diag_lib.c
		s_lapack_lib.c)

endif()

add_library(blas OBJECT ${BLAS_SRC})
