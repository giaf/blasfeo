% print blas comparision

d_Gflops_max = 0;
s_Gflops_max = 0;


routines = {
'gemm_nn',
'gemm_nt',
'gemm_tn',
'gemm_tt',
'syrk_ln',
'trmm_rlnn',
'trmm_rutn',
'trsm_llnu',
'trsm_lunn',
'trsm_rltn',
'trsm_rltu',
'trsm_rutn',
'potrf_l',
'potrf_u',
'getrf_rowpivot',
'gemv_n',
'gemv_t',
'gemv_nt',
'trmv_lnn',
'trmv_ltn',
'trsv_lnn',
'trsv_ltn',
'symv_l',
};

routine_names = {
'gemm\_nn',
'gemm\_nt',
'gemm\_tn',
'gemm\_tt',
'syrk\_ln',
'trmm\_rlnn',
'trmm\_rutn',
'trsm\_llnu',
'trsm\_lunn',
'trsm\_rltn',
'trsm\_rltu',
'trsm\_rutn',
'potrf\_l',
'potrf\_u',
'getrf\_rowpivot',
'gemv\_n',
'gemv\_t',
'gemv\_nt',
'trmv\_lnn',
'trmv\_ltn',
'trsv\_lnn',
'trsv\_ltn',
'symv\_l',
};

for jj=1:2*length(routines)
	
	fig(jj) = figure();

end


targets = {
'HIGH_PERFORMANCE/X64_INTEL_HASWELL/BLASFEO_API',
'HIGH_PERFORMANCE/X64_INTEL_HASWELL/BLAS_API',
%'HIGH_PERFORMANCE/X64_INTEL_SANDY_BRIDGE/BLASFEO_API',
%'HIGH_PERFORMANCE/X64_INTEL_CORE/BLASFEO_API',
%'HIGH_PERFORMANCE/X86_AMD_JAGUAR/BLASFEO_API',
%'HIGH_PERFORMANCE/X86_AMD_BARCELONA/BLASFEO_API',
%'HIGH_PERFORMANCE/GENERIC/BLASFEO_API',
%'REFERENCE/X64_INTEL_SANDY_BRIDGE/BLASFEO_API',
%'REFERENCE/X86_AMD_JAGUAR/BLASFEO_API',
%'BLAS_WRAPPER/X64_INTEL_HASWELL/BLASFEO_API',
%'BLAS_WRAPPER/X64_INTEL_SANDY_BRIDGE/BLASFEO_API',
%'BLAS_WRAPPER/X86_AMD_JAGUAR/BLASFEO_API',
};

target_names = {
'BF\_HP\_X64\_HW',
'SB\_HP\_X64\_HW',
%'BF\_HP\_X64\_SB',
%'BF\_HP\_X64\_CR',
%'BF\_HP\_X86\_JG',
%'BF\_HP\_X86\_BC',
%'BF\_HP\_GE',
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

