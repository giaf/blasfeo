# BLASFEO - BLAS For Embedded Optimization

BLASFEO provides a set of basic linear algebra routines, performance-optimized for matrices that fit in cache (i.e. generally up to a couple hundred size in each dimension), as typically encountered in embedded optimization applications.

## BLASFEO APIs

BLASFEO provides two APIs (Application Programming Interfaces):
- BLAS API: the standard BLAS and LAPACK APIs, with matrices stored in column-major.
- BLASFEO API: BLASFEO's own API is optimized to reduce overhead for small matrices.
It employes structures to describe matrices (```blasfeo_dmat```) and vectors (```blasfeo_dvec```), defined in ```include/blasfeo_common.h```.
The actual implementation of ```blasfeo_dmat``` and ```blasfeo_dvec``` depends on the ```TARGET```, ```LA``` (Linear Algebra) and ```MF``` (Matrix Format) choice.
The API is non-destructive, and compared to the BLAS API it has an additional matrix/vector argument reserved for the output.

## Supported Computer Architectures

The architecture for BLASFEO to use is specified using the ```TARGET``` build variable.
Currently BLASFEO supports the following architectures:

| TARGET                       | Description |
| ---------------------------- | ------------------------------------------------------------- |
| ```X64_INTEL_HASWELL```      | Intel Haswell or AMD Zen architectures or newer, AVX2 and FMA ISA, 64-bit OS |
| ```X64_INTEL_SANDY_BRIDGE``` | Intel Sandy-Bridge architecture or newer, AVX ISA, 64-bit OS |
| ```X64_INTEL_CORE```         | Intel Core architecture or newer, SSE3 ISA, 64-bit OS |
| ```X64_AMD_BULLDOZER```      | AMD Bulldozer architecture, AVX and FMA ISAs, 64-bit OS |
| ```X86_AMD_JAGUAR```         | AMD Jaguar architecture, AVX ISA, 32-bit OS |
| ```X86_AMD_BARCELONA```      | AMD Barcelona architecture, SSE3 ISA, 32-bit OS |
| ```ARMV8A_ARM_CORTEX_A57```  | ARMv8A architecture, VFPv4 and NEONv2 ISAs, 64-bit OS |
| ```ARMV8A_ARM_CORTEX_A53```  | ARMv8A architecture, VFPv4 and NEONv2 ISAs, 64-bit OS |
| ```ARMV7A_ARM_CORTEX_A15```  | ARMv7A architecture, VFPv4 and NEON ISAs, 32-bit OS |
| ```ARMV7A_ARM_CORTEX_A9```   | ARMv7A architecture, VFPv3 and NEON ISAs, 32-bit OS |
| ```ARMV7A_ARM_CORTEX_A7```   | ARMv7A architecture, VFPv4 and NEON ISAs, 32-bit OS |
| ```GENERIC```                | Generic target, coded in C, giving better performance if the architecture provides more than 16 scalar FP registers (e.g. many RISC such as ARM) |

Note that the ```X86_AMD_JAGUAR``` and ```X86_AMD_BARCELONA``` architectures are not currently supported by the CMake build system and can only be used through the included Makefile.

### Automatic Target Detection

When using the CMake build system, it is possible to automatically detect the X64 target the current computer can use.
This can be enabled by specifying the ```X64_AUTOMATIC``` target.
In this mode, the build system will automatically search through the X64 targets to find the best one that can both compile and run on the host machine.

### Target Testing

When using the CMake build system, tests will automatically be performed to see if the current compiler can compile the needed code for the selected target and that the current computer can execute the code compiled for the current target.
The execution test can be disabled by setting the ```BLASFEO_CROSSCOMPILING``` flag to true.
This is automatically done when CMake detects that cross compilation is happening.

## Linear Algebra Routines

The BLASFEO backend provides three possible implementations of each linear algebra routine, specified using the ```LA``` build variable:

| LA                          | Description |
| --------------------------- | ------------------------------------------------------------- |
| ```HIGH_PERFORMANCE```      | Target-tailored; performance-optimized for cache resident matrices; panel- or column-major matrix format. Currently provided for OS_LINUX (x86_64 64-bit, x86 32-bit, ARMv8A 64-bit, ARMv7A 32-bit), OS_WINDOWS (x86_64 64-bit) and OS_MAC (x86_64 64-bit). |
| ```REFERENCE```             | Target-unspecific lightly-optimizated; small code footprint; panel- or column-major matrix format |
| ```EXTERNAL_BLAS_WRAPPER``` | Call to external BLAS and LAPACK libraries; column-major matrix format |

## Matrix Format

Currently there are two matrix formats used in the BLASFEO matrix structures ```blasfeo_dmat``` and ```blasfeo_smat```, specified using the ```LA``` build variable:
- column-major: (or FORTRAN-style) the standard matrix format used in the BLAS and LAPACK libraries.
- panel-major: BLASFEO's own matrix format, which is designed to improve performance for matrices fitting in cache.
Each matrix is divided into panels with fixed height, and within each panel the matrix elements are stored in column-major matrix format.
For performance reasons, a similar format is internally used also in the high-performance implementation for ```MF=COLMAJ```, however at the additional computational cost due to packing and unpacking overhead.

## Recommended guidelines

Guidelines to use of BLASFEO routines and avoid known performance issues can be found in the file
[guidelines.md](https://github.com/giaf/blasfeo/blob/master/guidelines.md). <br/>
We strongly recommend the user to read it.

## More Information

More information can be found on the BLASFEO wiki at https://blasfeo.syscop.de, including more detailed installation instructions, examples, and a rich collection of benchmarks and comparisions.

More scientific information can be found in:
- the original BLASFEO paper describes the BLASFEO API and the backend (comprising the panel-major matrix format): <br/>
G. Frison, D. Kouzoupis, T. Sartor, A. Zanelli, M. Diehl, *BLASFEO: basic linear algebra subroutines for embedded optimization*. ACM Transactions on Mathematical Software (TOMS), 2018. <br/>
(arXiv preprint https://arxiv.org/abs/1704.02457 )
- the second BLASFEO paper describes the BLAS API implementation and the assembly framework with its custom function calling convention: <br/>
G. Frison, T. Sartor, A. Zanelli, M. Diehl, *The BLAS API of BLASFEO: optimizing performance for small matrices*, 2019. <br/>
(arXiv preprint https://arxiv.org/abs/1902.08115 )
- the slides introduce BLASFEO: <br/>
www.cs.utexas.edu/users/flame/BLISRetreat2017/slides/Gianluca_BLIS_Retreat_2017.pdf
- video with comments to the slides: <br/>
https://utexas.app.box.com/s/yt2d693v8xc37yyjklnf4a4y1ldvyzon

## Notes

- BLASFEO is released under the 2-Clause BSD License.

- 06-01-2018: BLASFEO employs now a new naming convention.
The bash script change_name.sh can be used to automatically change the source code of any software using BLASFEO to adapt it to the new naming convention.
