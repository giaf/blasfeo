
set(C99_SRC
    kernel_dgemv_4_lib4.c
    kernel_sgemm_4x4_lib4.c
    kernel_sgemm_diag_lib4.c
    kernel_sgemv_4_lib4.c
    kernel_ssymv_4_lib4.c
    kernel_sgetrf_pivot_4_lib4.c)

if(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE OR
   ${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER OR
   ${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57 OR
   ${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15 OR
   ${BLASFEO_TARGET} MATCHES GENERIC)

    list(APPEND C99_SRC
        kernel_dgemm_4x4_lib4.c
        kernel_dgemm_diag_lib4.c
        kernel_dsymv_4_lib4.c
        kernel_dgetrf_pivot_4_lib4.c
        kernel_dgeqrf_4_lib4.c
        kernel_dgecp_lib4.c
        kernel_dgetr_lib4.c
        
        kernel_sgecp_lib4.c
        kernel_sgetr_lib4.c)

endif()

add_library(c99 OBJECT ${C99_SRC})