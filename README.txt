BLASFEO - BLAS For Embedded Optimization

BLASFEO provides a set of linear algebra routines optimized for use in embedded optimization.
It is for example employed in the Model Predictive Control software package HPMPC.

BLASFEO provides three implementations of each linear algebra routine (LA):
- HIGH_PERFORMANCE: a high-performance implementation hand-optimized for different computer architectures.
- REFERENCE: a lightly-optimized version, coded entirely in C withou assumptions about the computer architecture.
- BLAS: a wrapper to BLAS and LAPACK routines.

The currently supported compter architectures (TARGET) are:
- X64_INTEL_HASWELL: Intel Haswell architecture or newer, AVX2 and FMA ISA, 64-bit OS.
- X64_INTEL_SANDY_BRIDGE: Intel Sandy-Bridge architecture or newer, AVX ISA, 64-bit OS.
- X64_INTEL_CORE: Intel Core architecture or newer, SSE3 ISA, 64-bit OS.
- X64_AMD_BULLDOZER: AMD Bulldozer architecture, AVX and FMA ISAs, 64-bit OS.
- ARMV78_ARM_CORTEX_A57: ARMv78 architecture, VFPv4 and NEONv2 ISAs, 64-bit OS.
- ARMV7A_ARM_CORTEX_A15: ARMv7A architecture, VFPv3 and NEON ISAs, 32-bit OS.
- GENERIC: generic target, coded in C, giving better performance if the architecture provides more than 16 scalar FP registers (e.g. many RISC such as ARM).

The optimized linear algebra kernels are currently provided for OS_LINUX (x86_64 64-bit, ARMv8A 64-bit, ARMv7A 32-bit), OS_WINDOWS (x86_64 64-bit) and OS_MAC (x86_64 64-bit).

BLASFEO employes structures to describe matrices (d_strmat) and vectors (d_strvec), defined in include/blasfeo_common.h.
The actual implementation of d_strmat and d_strvec depends on the LA and TARGET choice.

More information about BLASFEO can be found in the ArXiv paper at the URL
https://arxiv.org/abs/1704.02457
or in the slides at the URL
www.cs.utexas.edu/users/flame/BLISRetreat2017/slides/Gianluca_BLIS_Retreat_2017.pdf
or in the video at the URL
https://utexas.app.box.com/s/yt2d693v8xc37yyjklnf4a4y1ldvyzon
