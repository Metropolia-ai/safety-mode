#!/bin/bash
# build.sh <name> <onnx> <shapes> [extra trtexec args...]  -> trt/<name>.plan, log jetson-bench/logs/export_trtexec_<name>.log
name=$1; onnx=$2; shapes=$3; shift 3
T=${TRT_DIR:-/mnt/data/work/export/efficientsam3-ev-m/trt}
L=/mnt/data/work/aivm-safety-mode/jetson-bench/logs/export_trtexec_$name.log
cmd=(/usr/src/tensorrt/bin/trtexec --onnx=$onnx --shapes=$shapes --saveEngine=$T/$name.plan --timingCacheFile=$T/timing.cache --exportLayerInfo=$T/$name.layers.json --profilingVerbosity=detailed --separateProfileRun "$@")
{ echo "# $(date -Is)"; printf '$ '; printf '%q ' "${cmd[@]}" | cut -c1-4000; echo; } > $L
s=$(date +%s.%N); "${cmd[@]}" >> $L 2>&1; rc=$?; e=$(date +%s.%N)
echo "exit=$rc wall_s=$(echo "$e - $s" | bc)" >> $L
ls -la $T/$name.plan >> $L 2>&1
grep -E "Engine built in|GPU Compute Time: min|Latency: min|PASSED|FAILED|exit=|Error|error" $L | grep -v "^\$" | cut -c1-300 | tail -12
