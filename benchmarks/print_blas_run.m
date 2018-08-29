% print blas

tmp_run

Gflops_max = A(1)*A(2);

f1 = figure();
plot(B(:,1), B(:,2), 'b');
hold on
plot(B(:,1), B(:,4), 'g');
plot(B(:,1), B(:,6), 'r');
hold off

axis([0 300 0 Gflops_max]);
legend('blas', 'blas_api', 'blasfeo_api', 'Location', 'SouthEast');
xlabel('matrix size n')
ylabel('Gflops')
grid on

file_name_eps = ['tmp_run.eps'];
file_name_pdf = ['tmp_run.pdf'];
print(f1, file_name_eps, '-depsc') 
system(['epstopdf ', file_name_eps, ' -out ', file_name_pdf]);
system(['rm ', file_name_eps]);

