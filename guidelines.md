The performance of BLASFEO routines can be affected by many factor, and some can have a large impact on performance.


--------------------------------------------------

Known performance issues:
- use of denormals.
In some computer architectures (like e.g. the widespread x86_64) computations involving denormals floating point numbers are handled in microcode, and therefore can incur in a very large performance penalty (10x or more).
Unless computation on denormals is on purpose, the user should pay attention to avoid denormals on the data matrices as well as to the __memory passed to create BLASFEO matrices or vectors__ (as the padding memory is still used in internal computations, even if it is discarded in the result).
As a good practice, it is __recommended to zero out__ the memory passed to create a BLASFEO matrix or vector.
