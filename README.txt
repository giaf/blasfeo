BLASFEO - BLAS For Embedded Optimization

BLASFEO provides a set of basic linear algebra routines, performance-optimized for matrices of moderate size (up to a couple hundreds size in each dimension), as typically encountered in embedded optimization applications.

The currently supported compter architectures (TARGET) are:
- X64_INTEL_HASWELL: Intel Haswell architecture or newer, AVX2 and FMA ISA, 64-bit OS.
- X64_INTEL_SANDY_BRIDGE: Intel Sandy-Bridge architecture or newer, AVX ISA, 64-bit OS.
- X64_INTEL_CORE: Intel Core architecture or newer, SSE3 ISA, 64-bit OS.
- X64_AMD_BULLDOZER: AMD Bulldozer architecture, AVX and FMA ISAs, 64-bit OS.
- X86_AMD_JAGUAR: AMD Jaguar architecture, AVX ISA, 32-bit OS.
- X86_AMD_BARCELONA: AMD Barcelona architecture, SSE3 ISA, 32-bit OS.
- ARMV8A_ARM_CORTEX_A57: ARMv8A architecture, VFPv4 and NEONv2 ISAs, 64-bit OS.
- ARMV8A_ARM_CORTEX_A53: ARMv8A architecture, VFPv4 and NEONv2 ISAs, 64-bit OS.
- ARMV7A_ARM_CORTEX_A15: ARMv7A architecture, VFPv3 and NEON ISAs, 32-bit OS.
- GENERIC: generic target, coded in C, giving better performance if the architecture provides more than 16 scalar FP registers (e.g. many RISC such as ARM).

The BLASFEO backend provides three possible implementations of each linear algebra routine (LA):
- HIGH_PERFORMANCE: target-tailored; performance-optimized for cache resident matrices; panel-major matrix format
- REFERENCE: target-unspecific lightly-optimizated; small code footprint; column-major matrix format
- BLAS_WRAPPER: call to external BLAS and LAPACK libraries; column-major matrix format

The optimized linear algebra kernels are currently provided for OS_LINUX (x86_64 64-bit, x86 32-bit, ARMv8A 64-bit, ARMv7A 32-bit), OS_WINDOWS (x86_64 64-bit) and OS_MAC (x86_64 64-bit).

BLASFEO employes structures to describe matrices (blasfeo_dmat) and vectors (blasfeo_dvec), defined in include/blasfeo_common.h.
The actual implementation of blasfeo_dmat and blasfeo_dvec depends on the LA and TARGET choice.

More information about BLASFEO can be found in the ArXiv paper at the URL
https://arxiv.org/abs/1704.02457
or in the slides at the URL
www.cs.utexas.edu/users/flame/BLISRetreat2017/slides/Gianluca_BLIS_Retreat_2017.pdf
or in the video at the URL
https://utexas.app.box.com/s/yt2d693v8xc37yyjklnf4a4y1ldvyzon

--------------------------------------------------

Notes:

- 06-01-2018: BLASFEO employs now a new naming convention.
The bash script change_name.sh can be used to automatically change the source code of any software using BLASFEO to adapt it to the new naming convention.
