// void test_neon()

//  .p2align 4,,15
#if defined(OS_LINUX)
  .global test_neon
  .type test_neon, %function
test_neon:
#elif defined(OS_MAC)
  .global _test_neon
_test_neon:
#endif



#if defined(OS_LINUX)
  .size test_neon, .-test_neon
#endif
