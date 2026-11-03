#!/usr/bin/env bash
# Muse Glimmer 30B (official K-Quant-17GB Q4_K_M) via llama-server from the bench stand build, text only.
# Usage: server.sh start|stop|status|log   (env: PORT=8090 CTX=16384 EXTRA="...")
set -u
B=/mnt/data/work/bench/build/llama.cpp/build/bin/llama-server
M=/mnt/data/work/glimmer/model/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
PORT=${PORT:-8090}; CTX=${CTX:-16384}; LOG=/mnt/data/work/glimmer/server.log; PID=/mnt/data/work/glimmer/server.pid
case "${1:-}" in
  start)
    [ -f $PID ] && kill -0 $(cat $PID) 2>/dev/null && { echo "running pid $(cat $PID)"; exit 0; }
    nohup $B -m $M -a muse-glimmer-30B -ngl 99 -c $CTX -np 1 --host 127.0.0.1 --port $PORT --jinja \
      -fa on --temp 1.0 --top-p 0.95 --top-k 64 --metrics ${EXTRA:-} >>$LOG 2>&1 &
    echo $! > $PID
    for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "up on :$PORT pid $(cat $PID) after ${i}s"; exit 0; }; sleep 1; done
    echo "not up after 240 s"; tail -20 $LOG; exit 1 ;;
  stop) [ -f $PID ] && kill $(cat $PID) 2>/dev/null; rm -f $PID; echo stopped ;;
  status) curl -s http://127.0.0.1:$PORT/health; echo; curl -s http://127.0.0.1:$PORT/v1/models | head -c 300; echo ;;
  log) tail -n ${2:-40} $LOG ;;
  *) echo "usage: $0 start|stop|status|log"; exit 1 ;;
esac
