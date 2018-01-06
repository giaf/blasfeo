
if(${BLASFEO_LA_MODE} MATCHES HIGH_PERFORMANCE)

    set(KERNEL_SRC "")

    if(${BLASFEO_TARGET} MATCHES X64_INTEL_HASWELL)
        list(APPEND KERNEL_SRC
            avx2/kernel_dgemm_12x4_lib4.S
            avx2/kernel_dgemm_8x8_lib4.S
            avx2/kernel_dgemm_8x4_lib4.S
            avx2/kernel_dgemm_4x4_lib4.S
            avx/kernel_dgemm_diag_lib4.c
            avx2/kernel_dgemv_8_lib4.S
            avx/kernel_dgemv_4_lib4.S
            avx2/kernel_dsymv_6_lib4.S
            c99/kernel_dsymv_4_lib4.c
            avx2/kernel_dgetrf_pivot_4_lib4.c
            avx/kernel_dgeqrf_4_lib4.c
            avx2/kernel_dgebp_lib4.S
            avx2/kernel_dgelqf_4_lib4.S
            avx2/kernel_sgemm_24x4_lib8.S
            avx2/kernel_sgemm_16x4_lib8.S
            avx2/kernel_sgemm_8x8_lib8.S
            avx2/kernel_sgemm_8x4_lib8.S
            avx/kernel_sgemm_diag_lib8.c
            avx/kernel_sgead_lib8.S
            avx/kernel_sgecpsc_lib8.S
            avx/kernel_sgetr_lib8.S
            #avx/kernel_sgesc_lib8.S
            avx/kernel_sgemv_8_lib8.S
            avx/kernel_sgemv_4_lib8.S)

    elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)
        list(APPEND KERNEL_SRC
            avx/kernel_dgemm_8x4_lib4.S
            avx/kernel_dgemm_4x4_lib4.S
            avx/kernel_dgemm_diag_lib4.c
            avx/kernel_dgemv_12_lib4.S
            avx/kernel_dgemv_8_lib4.S
            avx/kernel_dgemv_4_lib4.S
            avx/kernel_dsymv_6_lib4.S
            c99/kernel_dsymv_4_lib4.c
            avx/kernel_dgetrf_pivot_4_lib4.c
            avx/kernel_dgeqrf_4_lib4.c
            avx/kernel_dgebp_lib4.S
            avx/kernel_sgemm_16x4_lib8.S
            avx/kernel_sgemm_8x8_lib8.S
            avx/kernel_sgemm_8x4_lib8.S
            avx/kernel_sgemm_diag_lib8.c
            avx/kernel_sgead_lib8.S
            avx/kernel_sgecpsc_lib8.S
            avx/kernel_sgetr_lib8.S
            #avx/kernel_sgesc_lib8.S
            avx/kernel_sgemv_8_lib8.S
            avx/kernel_sgemv_4_lib8.S)

    elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE)

        list(APPEND KERNEL_SRC
            sse3/kernel_dgemm_4x4_lib4.S
            c99/kernel_dgemm_4x4_lib4.c
            c99/kernel_dgemm_diag_lib4.c
            c99/kernel_dgemv_4_lib4.c
            c99/kernel_dsymv_4_lib4.c
            c99/kernel_dgetrf_pivot_4_lib4.c
            c99/kernel_dgeqrf_4_lib4.c
            c99/kernel_sgemm_4x4_lib4.c
            c99/kernel_sgemm_diag_lib4.c
            c99/kernel_sgemv_4_lib4.c
            c99/kernel_sgetrf_pivot_4_lib4.c
            c99/kernel_sgecp_lib4.c)

    elseif(${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER)

        list(APPEND KERNEL_SRC
            fma/kernel_dgemm_4x4_lib4.S
            c99/kernel_dgemm_4x4_lib4.c
            c99/kernel_dgemm_diag_lib4.c
            c99/kernel_dgemv_4_lib4.c
            c99/kernel_dsymv_4_lib4.c
            c99/kernel_dgetrf_pivot_4_lib4.c
            c99/kernel_dgeqrf_4_lib4.c
            c99/kernel_sgemm_4x4_lib4.c
            c99/kernel_sgemm_diag_lib4.c
            c99/kernel_sgemv_4_lib4.c
            c99/kernel_sgetrf_pivot_4_lib4.c
            c99/kernel_sgecp_lib4.c)

    elseif(${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)

        list(APPEND KERNEL_SRC
            armv8a/kernel_dgemm_8x4_lib4.S
            armv8a/kernel_dgemm_4x4_lib4.S
            c99/kernel_dgemm_4x4_lib4.c
            c99/kernel_dgemm_diag_lib4.c
            c99/kernel_dgemv_4_lib4.c
            c99/kernel_dsymv_4_lib4.c
            c99/kernel_dgetrf_pivot_4_lib4.c
            c99/kernel_dgeqrf_4_lib4.c
            armv8a/kernel_sgemm_16x4_lib4.S
            armv8a/kernel_sgemm_12x4_lib4.S
            armv8a/kernel_sgemm_8x8_lib4.S
            armv8a/kernel_sgemm_8x4_lib4.S
            armv8a/kernel_sgemm_4x4_lib4.S
            c99/kernel_sgemm_4x4_lib4.c
            c99/kernel_sgemm_diag_lib4.c
            c99/kernel_sgemv_4_lib4.c
            c99/kernel_sgetrf_pivot_4_lib4.c
            c99/kernel_sgecp_lib4.c)

    elseif(${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)

        list(APPEND KERNEL_SRC
            armv7a/kernel_dgemm_4x4_lib4.S
            c99/kernel_dgemm_4x4_lib4.c
            c99/kernel_dgemm_diag_lib4.c
            c99/kernel_dgemv_4_lib4.c
            c99/kernel_dsymv_4_lib4.c
            c99/kernel_dgetrf_pivot_4_lib4.c
            c99/kernel_dgeqrf_4_lib4.c
            armv7a/kernel_sgemm_12x4_lib4.S
            armv7a/kernel_sgemm_8x4_lib4.S
            armv7a/kernel_sgemm_4x4_lib4.S
            c99/kernel_sgemm_4x4_lib4.c
            c99/kernel_sgemm_diag_lib4.c
            c99/kernel_sgemv_4_lib4.c
            c99/kernel_sgetrf_pivot_4_lib4.c
            c99/kernel_sgecp_lib4.c)

    elseif(${BLASFEO_TARGET} MATCHES GENERIC)

        list(APPEND KERNEL_SRC
            c99/kernel_dgemm_4x4_lib4.c
            c99/kernel_dgemm_diag_lib4.c
            c99/kernel_dgemv_4_lib4.c
            c99/kernel_dsymv_4_lib4.c
            c99/kernel_dgetrf_pivot_4_lib4.c
            c99/kernel_dgeqrf_4_lib4.c
            c99/kernel_sgemm_4x4_lib4.c
            c99/kernel_sgemm_diag_lib4.c
            c99/kernel_sgemv_4_lib4.c
            c99/kernel_sgetrf_pivot_4_lib4.c
            c99/kernel_sgecp_lib4.c)

    endif()

endif()

add_library(kernel OBJECT ${KERNEL_SRC})
