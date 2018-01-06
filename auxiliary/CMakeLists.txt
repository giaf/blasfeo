
if(${BLASFEO_LA_MODE} MATCHES HIGH_PERFORMANCE)

    set(AUX_SRC d_aux_lib4.c)

    if(${BLASFEO_TARGET} MATCHES X64_INTEL_HASWELL OR
       ${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

       list(APPEND AUX_SRC
            s_aux_lib8.c
            m_aux_lib48.c)

    elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE OR
           ${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER OR
           ${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57 OR
           ${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15 OR
           ${BLASFEO_TARGET} MATCHES GENERIC)
	
        list(APPEND AUX_SRC
            s_aux_lib4.c
            m_aux_lib44.c)

    endif()

else() # Reference BLAS

    set(AUX_SRC
        d_aux_lib.c
        s_aux_lib.c
        m_aux_lib.c)

endif()

if(BLASFEO_EXT_DEP)

	set(EXT_SRC
        d_aux_ext_dep_lib.c
        d_aux_ext_dep_lib4.c
        s_aux_ext_dep_lib.c
        s_aux_ext_dep_lib4.c
		v_aux_ext_dep_lib.c
        i_aux_ext_dep_lib.c)

endif()

if(BLASFEO_TESTING)
    list(APPEND AUX_SRC
        d_aux_libref.c
        s_aux_libref.c)
endif()

add_library(auxiliary OBJECT ${AUX_SRC} ${EXT_SRC})
