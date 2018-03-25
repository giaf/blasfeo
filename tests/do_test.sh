#! /bin/bash

# TARGET

# intel
declare -a TARGETS=(
	"GENERIC"
	# "X64_INTEL_CORE"
	# "X64_INTEL_SANDY_BRIDGE"
	)

# LA=BLAS_WRAPPER
LA=HIGH_PERFORMANCE
# LA=REFERENCE
# REF_BLAS=OPENBLAS
REF_BLAS=0
VERBOSE=3

# arm
# amd

# LA
# routines
declare -a D_BLAS3_ROUTINES=(
	# "GEMM|blasfeo_dgemm_nn"
	# "GEMM|blasfeo_dgemm_nt"
	# "GEMM|blasfeo_dsyrk_ln_mn"
	# "SYRK|blasfeo_dsyrk_ln"
	"TRM|blasfeo_dtrsm_llnu"
	"TRM|blasfeo_dtrsm_rltu"
	"TRM|blasfeo_dtrsm_rltn"
	"TRM|blasfeo_dtrsm_rutn"
	# "TRM|blasfeo_dtrmm_rutn"
	"TRM|blasfeo_dtrmm_rlnn"
	"TRM|blasfeo_dtrsm_lunn"
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
	echo "Testing $LA:$TARGET"
	echo

	# make -s -C .. REF_BLAS=$REF_BLAS LA=$LA TARGET=$TARGET static_library
	make -s -C .. REF_BLAS=$REF_BLAS LA=$LA TARGET=$TARGET deploy_to_tests

	for ROUTINE_SPEC in "${D_BLAS3_ROUTINES[@]}"
	do
		ROUTINE="${ROUTINE_SPEC##*|}"
		ROUTINE_CLASS="${ROUTINE_SPEC%|*}"
		make -s REF_BLAS=$REF_BLAS LA=$LA TARGET=$TARGET ROUTINE=$ROUTINE ROUTINE_CLASS=$ROUTINE_CLASS TEST_VERBOSITY=$VERBOSE update_blas
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
