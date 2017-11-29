# intel
declare -a TARGETS=(
	"GENERIC"
	"X64_INTEL_CORE"
	"X64_INTEL_SANDY_BRIDGE"
	)
# arm
# amd

echo "Cleaning .."
make -C .. clean
make clean

DONE=0
TOTAL=${#TARGETS[@]}

## now loop through the above array
for TARGET in "${TARGETS[@]}"
do
	echo "Testing $TARGET"
	TESTING=1 TARGET=$TARGET make -C .. -e run_test_aux_clean
	status=$?
	let "DONE+=1"

	if [ $status -ne 0 ]; then

		echo
		echo "error with target $TARGET ($DONE/$TOTAL)" >&2
		echo

		exit
	fi

	echo
	echo "Completed  $DONE/$TOTAL tests"
	echo
done
