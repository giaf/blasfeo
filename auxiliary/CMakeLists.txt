
if(${BLASFEO_LA_MODE} MATCHES HIGH_PERFORMANCE)

    set(AUX_SRC d_aux_lib4.c)

    if(${BLASFEO_TARGET} MATCHES X64_INTEL_HASWELL)
        list(APPEND AUX_SRC
            ../kernel/avx/kernel_dgecp_lib4.c
            ../kernel/avx2/kernel_dgetr_lib4.c
            s_aux_lib8.c
            m_aux_lib48.c)
    
    elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

		list(APPEND AUX_SRC
            ../kernel/avx/kernel_dgecp_lib4.c
			../kernel/avx/kernel_dgetr_lib4.c
			s_aux_lib8.c
            m_aux_lib48.c)

    elseif(${BLASFEO_TARGET} MATCHES 
        X64_INTEL_CORE | X64_AMD_BULLDOZER | ARMV8A_ARM_CORTEX_A57 | ARMV7A_ARM_CORTEX_A15 | GENERIC )
	
        list(APPEND AUX_SRC
            ../kernel/c99/kernel_dgecp_lib4.c
            ../kernel/c99/kernel_dgetr_lib4.c 
            s_aux_lib4.c
            ../kernel/c99/kernel_sgetr_lib4.c
            m_aux_lib44.c)

    endif()

else() # Reference BLAS

    set(AUX_SRC
        d_aux_lib.c
        s_aux_lib.c
        m_aux_lib.c)

endif()

if(EXT_DEP)

	set(EXT_SRC
		d_aux_ext_dep_lib.c
		s_aux_ext_dep_lib.c
		v_aux_ext_dep_lib.c
		i_aux_ext_dep_lib.c)

endif()

add_library(auxiliary OBJECT ${AUX_SRC} ${EXT_SRC})
