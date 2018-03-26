#! /bin/bash

# default config
LA=HIGH_PERFORMANCE
REF_BLAS=0
VERBOSE=1
BUILD_LIBS=1

declare -a TARGETS=()
declare -a D_BLAS3_ROUTINES=()

# import config for batch test
source batch_test_local.sh

DONE=0
TOTAL=$((${#D_BLAS3_ROUTINES[@]} * ${#TARGETS[@]}))

## now loop through the above array
for TARGET in "${TARGETS[@]}"
do
	echo
	echo "Testing $LA:$TARGET"
	echo

	if [ $BUILD_LIBS -eq 1 ]; then
	make -s -C .. REF_BLAS=$REF_BLAS LA=$LA TARGET=$TARGET
	make -s -C .. REF_BLAS=$REF_BLAS LA=$LA TARGET=$TARGET deploy_to_tests
	fi

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
