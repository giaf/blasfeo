// TODO remove
//
void strcp_l_lib(int m, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
void sgead_lib(int m, int n, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
void sgetr_lib(int m, int n, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void strtr_l_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void strtr_u_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void sdiareg_lib(int kmax, float reg, int offset, float *pD, int sdd);
void sdiain_sqrt_lib(int kmax, float *x, int offset, float *pD, int sdd);
void sdiaex_lib(int kmax, float alpha, int offset, float *pD, int sdd, float *x);
void sdiaad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
void sdiain_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
void sdiaex_libsp(int kmax, int *idx, float alpha, float *pD, int sdd, float *x);
void sdiaad_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
void sdiaadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD, int sdd);
void srowin_lib(int kmax, float alpha, float *x, float *pD);
void srowex_lib(int kmax, float alpha, float *pD, float *x);
void srowad_lib(int kmax, float alpha, float *x, float *pD);
void srowin_libsp(int kmax, float alpha, int *idx, float *x, float *pD);
void srowad_libsp(int kmax, int *idx, float alpha, float *x, float *pD);
void srowadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD);
void srowsw_lib(int kmax, float *pA, float *pC);
void scolin_lib(int kmax, float *x, int offset, float *pD, int sdd);
void scolad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
void scolin_libsp(int kmax, int *idx, float *x, float *pD, int sdd);
void scolad_libsp(int kmax, float alpha, int *idx, float *x, float *pD, int sdd);
void scolsw_lib(int kmax, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void svecin_libsp(int kmax, int *idx, float *x, float *y);
void svecad_libsp(int kmax, int *idx, float alpha, float *x, float *y);
