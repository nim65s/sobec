#!/bin/bash
slopes=(
	00
	05
	10
	15
	20
	25
	30
)
slope_ref=00
base=(
	500
	525
	550
	575
	600
	625
	650
)
base_ref=605
velocities=(
	10
	15
	20
	25
	30
	35
	40
	45
	50
)
vel_ref=20
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

	SAVE_DIR="${RESULTS_DIR}/walk_slope_${s}_base_${b}_vel_${v}"
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
	$OPEN "True" "$SAVE_DIR/open_warmstarted.npy" "$SAVE_DIR/open_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	$CLOSED "True" "$SAVE_DIR/closed_warmstarted.npy" "$SAVE_DIR/closed_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	#
	$CLOSED2OPEN "$SAVE_DIR/closed_warmstarted.npy" "$SAVE_DIR/open_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	$OPEN2CLOSED "$SAVE_DIR/open_warmstarted.npy" "$SAVE_DIR/closed_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	#
	$OPEN "True" "$SAVE_DIR/open_warmstarted_warmstarted.npy" "$SAVE_DIR/open_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
	$CLOSED "True" "$SAVE_DIR/closed_warmstarted_warmstarted.npy" "$SAVE_DIR/closed_warmstarted_ws.npy" "${b}" "${v}" "${s}" $AUTOSAVE
}

for s in "${slopes[@]}"; do
	run_all $s "${base_ref}" "${vel_ref}"
done
for b in "${base[@]}"; do
	run_all "${slope_ref}" $b "${vel_ref}"
done
for v in "${velocities[@]}"; do
	run_all "${slope_ref}" "${base_ref}" $v
done