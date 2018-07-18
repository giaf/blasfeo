% print blas comparision

d_Gflops_max = 0;
s_Gflops_max = 0;


routines = {
'gemm_nt',
'gemm_nn',
'potrf_l',
'gemv_n',
'gemv_t',
};

routine_names = {
'gemm\_nt',
'gemm\_nn',
'potrf\_l',
'gemv\_n',
'gemv\_t',
};

for jj=1:2*length(routines)
	
	fig(jj) = figure();

end


targets = {
'HIGH_PERFORMANCE/X64_INTEL_SANDY_BRIDGE',
'HIGH_PERFORMANCE/X64_INTEL_CORE',
'HIGH_PERFORMANCE/X86_AMD_JAGUAR',
'REFERENCE/X64_INTEL_SANDY_BRIDGE',
'BLAS_WRAPPER/X64_INTEL_SANDY_BRIDGE',
};

target_names = {
'HP\_X64\_SB',
'HP\_X64\_CR',
'HP\_X86\_JG',
'RF',
'OB',
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



% finalize

for jj=1:length(routines)
	
	% double
	figure(fig(2*(jj-1)+1))
	title(['d', routine_names{jj}])
	axis([0 300 0 d_Gflops_max]);
	xlabel('matrix size n')
	ylabel('Gflops')
	grid on
	legend(target_names)

	% single
	figure(fig(2*(jj-1)+2))
	title(['s', routine_names{jj}])
	axis([0 300 0 s_Gflops_max]);
	xlabel('matrix size n')
	ylabel('Gflops')
	grid on
	legend(target_names)

end

