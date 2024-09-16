#!/bin/bash
slopes=(
	00
	02
	05
	07
	10
	12
	15
	17
	20
	22
	25
	27
	30	
)
slope_ref=00
comWeight=(
	# 0
	# 10
	# 100
	# 250
	# 500
	# 1000
	# 2500
	# 5000
	# 10000
)
comWeight_ref=1000
velocities=(
	# 10
	# 15
	# 20
	# 25
	# 30
	# 32
	# 34
	# 35
	# 36
	# 38
	# 40
	# 42
	# 44
	# 45
	# 46
	# 48
	# 50
	# 55
	# 60
	# 65
	# 70
)
vel_ref=40
AUTOSAVE="True"
timestamp=$(date +%T)

OPEN="python walk_virgile_open.py"
CLOSED="python walk_virgile_closed.py"
CLOSED2OPEN="python warmstart_closed_to_open.py"
OPEN2CLOSED="python warmstart_open_to_closed.py"

RESULTS_DIR="results"

run_all () {
	local s=$1
	local b=$2
	local v=$3

	SAVE_DIR="${RESULTS_DIR}/walk_slope_${s}_comWeight_${b}_vel_${v}"
	if [ -d "$SAVE_DIR" ]; then
		echo "Directory $SAVE_DIR exists, moving it to save folder"
		mv "$SAVE_DIR" "${SAVE_DIR}_${timestamp}"
	fi
	mkdir "$SAVE_DIR"

	# Save the results
	$OPEN "" "$SAVE_DIR/open.npy" "" "${b}" "${v}" "${s}" $AUTOSAVE
	$CLOSED "" "$SAVE_DIR/closed.npy" "" "${b}" "${v}" "${s}" $AUTOSAVE
	#
	$CLOSED2OPEN "$SAVE_DIR/closed.npy" "$SAVE_DIR/open_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	$OPEN2CLOSED "$SAVE_DIR/open.npy" "$SAVE_DIR/closed_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	#
	# $OPEN "True" "$SAVE_DIR/open_warmstarted.npy" "$SAVE_DIR/open_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	# $CLOSED "True" "$SAVE_DIR/closed_warmstarted.npy" "$SAVE_DIR/closed_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	# #
	# $CLOSED2OPEN "$SAVE_DIR/closed_warmstarted.npy" "$SAVE_DIR/open_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	# $OPEN2CLOSED "$SAVE_DIR/open_warmstarted.npy" "$SAVE_DIR/closed_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	# #
	# $OPEN "True" "$SAVE_DIR/open_warmstarted_warmstarted.npy" "$SAVE_DIR/open_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	# $CLOSED "True" "$SAVE_DIR/closed_warmstarted_warmstarted.npy" "$SAVE_DIR/closed_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
}

for s in "${slopes[@]}"; do
	run_all $s "00" "30"
done
for w in "${comWeight[@]}"; do
	run_all "${slope_ref}" $w "${vel_ref}"
done
for v in "${velocities[@]}"; do
	run_all "${slope_ref}" "${comWeight_ref}" $v
done
