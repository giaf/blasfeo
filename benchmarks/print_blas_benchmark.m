% print blas benchmark

% DGEMM

dgemm_nt
save -mat dgemm_nt.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');
hold on

dgemm_nn
save -mat dgemm_nn.mat A B

plot(B(:,1), B(:,2), 'b');
hold off

axis([0 300 0 Gflops_max]);
legend('dgemm\_nt', 'dgemm\_nn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['dgemm.eps'];
print(f1, file_name, '-depsc') 

% DSYRK

dsyrk_ln
save -mat dsyrk_ln.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('dsyrk\_ln', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['dsyrk.eps'];
print(f1, file_name, '-depsc') 

% DTRMM

dtrmm_rlnn
save -mat dtrmm_rlnn.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('dtrmm\_rlnn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['dtrmm.eps'];
print(f1, file_name, '-depsc') 

% DTRSM

dtrsm_rltn
save -mat dtrsm_rltn.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('dtrsm\_rltn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['dtrsm.eps'];
print(f1, file_name, '-depsc') 

% DPOTRF

dpotrf_l
save -mat dpotrf_l.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('dpotrf\_l', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['dpotrf.eps'];
print(f1, file_name, '-depsc') 

% DGEMV

dgemv_n
save -mat dgemv_n.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');
hold on

dgemv_t
save -mat dgemv_t.mat A B

plot(B(:,1), B(:,2), 'b');
hold off

axis([0 300 0 Gflops_max]);
legend('dgemv\_n', 'dgemv\_t', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['dgemv.eps'];
print(f1, file_name, '-depsc') 

% DTRMV

dtrmv_lnn
save -mat dtrmv_lnn.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');
hold on

dtrmv_ltn
save -mat dtrmv_ltn.mat A B

plot(B(:,1), B(:,2), 'b');
hold off

axis([0 300 0 Gflops_max]);
legend('dtrmv\_lnn', 'dtrmv\_ltn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['dtrmv.eps'];
print(f1, file_name, '-depsc') 

% SGEMM

sgemm_nt
save -mat sgemm_nt.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');
hold on

sgemm_nn
save -mat sgemm_nn.mat A B

plot(B(:,1), B(:,2), 'b');
hold off

axis([0 300 0 Gflops_max]);
legend('sgemm\_nt', 'sgemm\_nn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['sgemm.eps'];
print(f1, file_name, '-depsc') 

% SSYRK

ssyrk_ln
save -mat ssyrk_ln.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('ssyrk\_ln', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['ssyrk.eps'];
print(f1, file_name, '-depsc') 

% STRMM

strmm_rlnn
save -mat strmm_rlnn.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('strmm\_rlnn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['strmm.eps'];
print(f1, file_name, '-depsc') 

% DTRSM

strsm_rltn
save -mat strsm_rltn.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('strsm\_rltn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['strsm.eps'];
print(f1, file_name, '-depsc') 

% SPOTRF

spotrf_l
save -mat spotrf_l.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');

axis([0 300 0 Gflops_max]);
legend('spotrf\_l', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['spotrf.eps'];
print(f1, file_name, '-depsc') 

% SGEMV

sgemv_n
save -mat sgemv_n.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');
hold on

sgemv_t
save -mat sgemv_t.mat A B

plot(B(:,1), B(:,2), 'b');
hold off

axis([0 300 0 Gflops_max]);
legend('sgemv\_n', 'sgemv\_t', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['sgemv.eps'];
print(f1, file_name, '-depsc') 

% STRMV

strmv_lnn
save -mat strmv_lnn.mat A B

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'r');
hold on

strmv_ltn
save -mat strmv_ltn.mat A B

plot(B(:,1), B(:,2), 'b');
hold off

axis([0 300 0 Gflops_max]);
legend('strmv\_lnn', 'strmv\_ltn', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name = ['strmv.eps'];
print(f1, file_name, '-depsc') 


