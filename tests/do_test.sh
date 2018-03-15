#! /bin/bash

# TARGET

# intel
declare -a TARGETS=(
	# "GENERIC"
	# "X64_INTEL_CORE"
	"X64_INTEL_SANDY_BRIDGE"
	)
# arm
# amd

# LA
# routines
declare -a D_BLAS3_ROUTINES=(
	# "GEMM|blasfeo_dgemm_nn"
	# "GEMM|blasfeo_dgemm_nt"
	# "GEMM|blasfeo_dsyrk_ln_mn"
	"SYRK|blasfeo_dsyrk_ln"
	"TRM|blasfeo_dtrsm_llnu"
	"TRM|blasfeo_dtrsm_rltu"
	"TRM|blasfeo_dtrsm_rltn"
	"TRM|blasfeo_dtrsm_rutn"
	"TRM|blasfeo_dtrmm_rutn"
	"TRM|blasfeo_dtrmm_rlnn"
	"TRM|blasfeo_dtrsm_llunn"
	)


# echo "Cleaning .."
# make -C .. clean
# make clean

DONE=0
TOTAL=$((${#D_BLAS3_ROUTINES[@]} * ${#TARGETS[@]}))

## now loop through the above array
for TARGET in "${TARGETS[@]}"
do
	echo
	echo "Testing $TARGET"
	echo

	# make -C .. TARGET=$TARGET >>log.txt 2>&1
	# make -C .. TARGET=$TARGET deploy_to_tests >>log.txt 2>&1

	for ROUTINE_SPEC in "${D_BLAS3_ROUTINES[@]}"
	do
		ROUTINE="${ROUTINE_SPEC##*|}"
		ROUTINE_CLASS="${ROUTINE_SPEC%|*}"
		make -s TARGET=$TARGET ROUTINE=$ROUTINE ROUTINE_CLASS=$ROUTINE_CLASS TEST_VERBOSITY=0 update_blas
		status=$?
		let "DONE+=1"

		if [ $status -ne 0 ]; then

			echo
			echo "error with target $TARGET $ROUTINE ($DONE/$TOTAL)" >&2
			echo

			exit
		fi

		echo
		echo "Completed  $TARGET $ROUTINE ($DONE/$TOTAL) "
		echo
	done
done
