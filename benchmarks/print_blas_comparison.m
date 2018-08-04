% print blas comparision

d_Gflops_max = 0;
s_Gflops_max = 0;


routines = {
'gemm_nt',
'gemm_nn',
'syrk_ln',
'trmm_rlnn',
'trsm_rltn',
'potrf_l',
'gemv_n',
'gemv_t',
'trmv_lnn',
'trmv_ltn',
};

routine_names = {
'gemm\_nt',
'gemm\_nn',
'syrk\_ln',
'trmm\_rlnn',
'trsm\_rltn',
'potrf\_l',
'gemv\_n',
'gemv\_t',
'trmv\_lnn',
'trmv\_ltn',
};

for jj=1:2*length(routines)
	
	fig(jj) = figure();

end


targets = {
'HIGH_PERFORMANCE/X64_INTEL_HASWELL',
%'HIGH_PERFORMANCE/X64_INTEL_SANDY_BRIDGE',
%'HIGH_PERFORMANCE/X64_INTEL_CORE',
%'HIGH_PERFORMANCE/X86_AMD_JAGUAR',
'HIGH_PERFORMANCE/X86_AMD_BARCELONA',
%'HIGH_PERFORMANCE/GENERIC',
%'REFERENCE/X64_INTEL_SANDY_BRIDGE',
%'REFERENCE/X86_AMD_JAGUAR',
%'BLAS_WRAPPER/X64_INTEL_HASWELL',
%'BLAS_WRAPPER/X64_INTEL_SANDY_BRIDGE',
%'BLAS_WRAPPER/X86_AMD_JAGUAR',
};

target_names = {
'HP\_X64\_HW',
%'HP\_X64\_SB',
%'HP\_X64\_CR',
%'HP\_X86\_JG',
'HP\_X86\_BC',
%'HP\_GE',
%'RF',
%'OB',
};


for ii=1:length(targets)

	path = ['build/', targets{ii}, '/data/'];

	for jj=1:length(routines)

		% double
		file = [path, 'd', routines{jj}, '.mat']

		load(file)

		d_Gflops_max = max(d_Gflops_max, A(1)*A(2));

		figure(fig(2*(jj-1)+1))
		hold all
		plot(B(:,1), B(:,2));
		hold off

		% single

		file = [path, 's', routines{jj}, '.mat']

		load(file)

		s_Gflops_max = max(s_Gflops_max, A(1)*A(2));

		figure(fig(2*(jj-1)+2))
		hold all
		plot(B(:,1), B(:,2));
		hold off
	
	end

end



% finalize & print

system('mkdir -p figures');

for jj=1:length(routines)
	
	% double
	figure(fig(2*(jj-1)+1))
	title(['d', routine_names{jj}])
	axis([0 300 0 d_Gflops_max]);
	xlabel('matrix size n')
	ylabel('Gflops')
	grid on
	legend(target_names)

	file_name_eps = ['figures/d', routines{jj}, '.eps'];
	file_name_pdf = ['figures/d', routines{jj}, '.pdf'];
	print(fig(2*(jj-1)+1), file_name_eps, '-depsc')
	system(['epstopdf ', file_name_eps, ' -out ', file_name_pdf]);
	system(['rm ', file_name_eps]);

	% single
	figure(fig(2*(jj-1)+2))
	title(['s', routine_names{jj}])
	axis([0 300 0 s_Gflops_max]);
	xlabel('matrix size n')
	ylabel('Gflops')
	grid on
	legend(target_names)

	file_name_eps = ['figures/s', routines{jj}, '.eps'];
	file_name_pdf = ['figures/s', routines{jj}, '.pdf'];
	print(fig(2*(jj-1)+2), file_name_eps, '-depsc')
	system(['epstopdf ', file_name_eps, ' -out ', file_name_pdf]);
	system(['rm ', file_name_eps]);

end

