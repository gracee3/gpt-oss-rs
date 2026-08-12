#!/usr/bin/env bash
# Full-model explicit-Xe API, cancellation, graceful-drain, and soak gate.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 SERVER_BIN MODEL_SNAPSHOT REPACK_CACHE RESULTS_DIR [SOAK_SECONDS]" >&2
  exit 2
fi

server_bin=$1
model_snapshot=$2
repack_cache=$3
results_dir=$4
soak_seconds=${5:-1800}
port=${GPT_OSS_XE_GATE_PORT:-18009}
base_url="http://127.0.0.1:${port}"
served_model=openai/gpt-oss-20b

mkdir -p "$results_dir"
server_log="$results_dir/server.log"
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd \
  "$server_bin" serve \
  --model "$model_snapshot" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --device xe \
  --xe-max-resident-mib 128 \
  --cpu-threads 4 \
  --cpu-repack-cache "$repack_cache" \
  --max-model-len 1024 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 512 \
  --max-prefill-chunk 512 \
  --drain-deadline-seconds 180 \
  --evidence-dir "$results_dir/runtime-evidence" \
  >"$server_log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 360); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    echo "server exited before readiness" >&2
    exit 1
  fi
  if curl --silent --show-error --fail "$base_url/ready" \
    --output "$results_dir/ready.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ $ready -ne 1 ]]; then
  echo "server did not become ready within 360 seconds" >&2
  exit 1
fi

start_swap_kib=$(awk '/^VmSwap:/ {print $2}' "/proc/$server_pid/status")
start_system_swap_free_kib=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
start_major_faults=$(awk '{print $12}' "/proc/$server_pid/stat")
wayland_processes=$(pgrep -cx kwin_wayland || true)

cache_program=$(find "$repack_cache/xe/native" -type f -name program.bin -print -quit 2>/dev/null || true)
if [[ -z "$cache_program" ]]; then
  echo "Xe native cache program was not created during startup" >&2
  exit 1
fi
cache_program_sha256=$(sha256sum "$cache_program" | awk '{print $1}')
cache_probe_dir="$results_dir/cache-probe"
mkdir -p "$cache_probe_dir"
cp "$cache_program" "$cache_probe_dir/program.bin.original"
printf 'corrupt native cache\n' >"$cache_program"
corrupt_cache_sha256=$(sha256sum "$cache_program" | awk '{print $1}')

chat_payload="$results_dir/chat-payload.json"
responses_payload="$results_dir/responses-payload.json"
jq -n --arg model "$served_model" '{model:$model,messages:[{role:"user",content:"Summarize: alpha beta gamma delta."}],max_tokens:2,temperature:0}' >"$chat_payload"
jq -n --arg model "$served_model" '{model:$model,input:"Summarize: alpha beta gamma delta.",max_output_tokens:2,temperature:0}' >"$responses_payload"

curl --silent --show-error --fail "$base_url/v1/chat/completions" \
  -H 'content-type: application/json' --data-binary "@$chat_payload" \
  --output "$results_dir/chat-nonstream.json"
jq -e '.id and (.choices | length > 0)' "$results_dir/chat-nonstream.json" >/dev/null

jq '.stream=true' "$chat_payload" >"$results_dir/chat-stream-payload.json"
curl --silent --show-error --fail --no-buffer "$base_url/v1/chat/completions" \
  -H 'content-type: application/json' --data-binary "@$results_dir/chat-stream-payload.json" \
  --output "$results_dir/chat-stream.sse"
grep -q 'data: \[DONE\]' "$results_dir/chat-stream.sse"

curl --silent --show-error --fail "$base_url/v1/responses" \
  -H 'content-type: application/json' --data-binary "@$responses_payload" \
  --output "$results_dir/responses-nonstream.json"
jq -e '.id and (.status == "completed" or .status == "incomplete")' \
  "$results_dir/responses-nonstream.json" >/dev/null

jq '.stream=true' "$responses_payload" >"$results_dir/responses-stream-payload.json"
curl --silent --show-error --fail --no-buffer "$base_url/v1/responses" \
  -H 'content-type: application/json' --data-binary "@$results_dir/responses-stream-payload.json" \
  --output "$results_dir/responses-stream.sse"
grep -Eq 'event: response\.(completed|incomplete)' "$results_dir/responses-stream.sse"

curl --silent --show-error --fail --no-buffer "$base_url/v1/chat/completions" \
  -H 'content-type: application/json' --data-binary "@$results_dir/chat-stream-payload.json" \
  --output "$results_dir/concurrent-chat.sse" &
concurrent_chat_pid=$!
curl --silent --show-error --fail "$base_url/v1/responses" \
  -H 'content-type: application/json' --data-binary "@$responses_payload" \
  --output "$results_dir/concurrent-responses.json" &
concurrent_responses_pid=$!
wait "$concurrent_chat_pid"
wait "$concurrent_responses_pid"
grep -q 'data: \[DONE\]' "$results_dir/concurrent-chat.sse"
jq -e '.id' "$results_dir/concurrent-responses.json" >/dev/null

long_prompt=$(printf 'alpha beta gamma delta %.0s' {1..120})
jq -n --arg model "$served_model" --arg prompt "$long_prompt" \
  '{model:$model,input:$prompt,max_output_tokens:8,temperature:0}' \
  >"$results_dir/cancel-payload.json"
curl --silent --no-buffer --limit-rate 1000 "$base_url/v1/responses" \
  -H 'content-type: application/json' \
  --data-binary "@$results_dir/cancel-payload.json" \
  --output "$results_dir/cancel-before.out" &
cancel_before_pid=$!
sleep 0.5
kill -TERM "$cancel_before_pid" 2>/dev/null || true
wait "$cancel_before_pid" 2>/dev/null || true

curl --silent --no-buffer "$base_url/v1/responses" -H 'content-type: application/json' \
  --data-binary "@$results_dir/cancel-payload.json" \
  --output "$results_dir/cancel-after.out" &
cancel_after_pid=$!
sleep 2
kill -TERM "$cancel_after_pid" 2>/dev/null || true
wait "$cancel_after_pid" 2>/dev/null || true

curl --silent --show-error --fail "$base_url/ready" \
  --output "$results_dir/ready-after-cancellation.json"

soak_started=$(date +%s)
soak_deadline=$((soak_started + soak_seconds))
soak_count=0
: >"$results_dir/soak.ndjson"
while (( $(date +%s) < soak_deadline )); do
  soak_count=$((soak_count + 1))
  jq '.max_output_tokens=1 | .stream=false' "$responses_payload" \
    >"$results_dir/soak-payload.json"
  curl --silent --show-error --fail "$base_url/v1/responses" \
    -H 'content-type: application/json' --data-binary "@$results_dir/soak-payload.json" \
    >>"$results_dir/soak.ndjson"
  printf '\n' >>"$results_dir/soak.ndjson"
done
soak_finished=$(date +%s)

end_swap_kib=$(awk '/^VmSwap:/ {print $2}' "/proc/$server_pid/status")
end_system_swap_free_kib=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
end_major_faults=$(awk '{print $12}' "/proc/$server_pid/stat")
peak_rss_kib=$(awk '/^VmHWM:/ {print $2}' "/proc/$server_pid/status")

kill -TERM "$server_pid"
set +e
wait "$server_pid"
server_status=$?
set -e
server_pid=
if [[ $server_status -ne 0 ]]; then
  echo "server graceful shutdown returned $server_status" >&2
  exit 1
fi
grep -q 'attached serialized CPU+Xe projection engine' "$server_log"
grep -q 'server shut down gracefully' "$server_log"
if grep -Eq 'Xe projection failed|service entered failed|panicked at' "$server_log"; then
  echo "server log contains a runtime, lifecycle, or panic failure" >&2
  exit 1
fi

cache_log="$results_dir/cache-recovery-server.log"
OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd \
  "$server_bin" serve \
  --model "$model_snapshot" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --device xe \
  --xe-max-resident-mib 128 \
  --cpu-threads 4 \
  --cpu-repack-cache "$repack_cache" \
  --max-model-len 1024 \
  --drain-deadline-seconds 180 \
  --evidence-dir "$results_dir/cache-recovery-runtime-evidence" \
  >"$cache_log" 2>&1 &
server_pid=$!
cache_ready=0
for _ in $(seq 1 360); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    echo "cache-recovery server exited before readiness" >&2
    exit 1
  fi
  if curl --silent --show-error --fail "$base_url/ready" \
    --output "$results_dir/cache-recovery-ready.json" 2>/dev/null; then
    cache_ready=1
    break
  fi
  sleep 1
done
if [[ $cache_ready -ne 1 ]]; then
  echo "cache-recovery server did not become ready" >&2
  exit 1
fi
recovered_cache_sha256=$(sha256sum "$cache_program" | awk '{print $1}')
if [[ "$recovered_cache_sha256" == "$corrupt_cache_sha256" ]]; then
  echo "corrupt native cache was not replaced" >&2
  exit 1
fi
cp "$cache_program" "$cache_probe_dir/program.bin.recovered"
kill -TERM "$server_pid"
set +e
wait "$server_pid"
cache_server_status=$?
set -e
server_pid=
if [[ $cache_server_status -ne 0 ]]; then
  echo "cache-recovery server shutdown returned $cache_server_status" >&2
  exit 1
fi
grep -q 'attached serialized CPU+Xe projection engine' "$cache_log"
grep -q 'server shut down gracefully' "$cache_log"

auto_log="$results_dir/auto-cpu-server.log"
GPT_OSS_XE_OPENCL_LIBRARY=/definitely/missing/gpt-oss-rs/libOpenCL.so \
  OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd \
  "$server_bin" serve \
  --model "$model_snapshot" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --device auto \
  --cpu-threads 4 \
  --cpu-repack-cache "$repack_cache" \
  --max-model-len 1024 \
  --drain-deadline-seconds 180 \
  --evidence-dir "$results_dir/auto-runtime-evidence" \
  >"$auto_log" 2>&1 &
server_pid=$!
auto_ready=0
for _ in $(seq 1 360); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    echo "automatic CPU server exited before readiness" >&2
    exit 1
  fi
  if curl --silent --show-error --fail "$base_url/ready" \
    --output "$results_dir/auto-cpu-ready.json" 2>/dev/null; then
    auto_ready=1
    break
  fi
  sleep 1
done
if [[ $auto_ready -ne 1 ]]; then
  echo "automatic CPU server did not become ready" >&2
  exit 1
fi
if [[ $(jq -c 'keys' "$results_dir/ready.json") != $(jq -c 'keys' "$results_dir/auto-cpu-ready.json") ]]; then
  echo "explicit Xe and automatic CPU readiness wire keys differ" >&2
  exit 1
fi
kill -TERM "$server_pid"
set +e
wait "$server_pid"
auto_server_status=$?
set -e
server_pid=
if [[ $auto_server_status -ne 0 ]]; then
  echo "automatic CPU server shutdown returned $auto_server_status" >&2
  exit 1
fi
grep -q 'native CPU runtime selected' "$auto_log"
grep -q 'server shut down gracefully' "$auto_log"
if grep -Eq 'automatic Xe probe|attached serialized CPU\+Xe|OpenCL loader unavailable' "$auto_log"; then
  echo "disabled automatic promotion unexpectedly probed or attached OpenCL" >&2
  exit 1
fi

sha256sum \
  "$results_dir/ready.json" \
  "$results_dir/chat-nonstream.json" \
  "$results_dir/chat-stream.sse" \
  "$results_dir/responses-nonstream.json" \
  "$results_dir/responses-stream.sse" \
  "$results_dir/concurrent-chat.sse" \
  "$results_dir/concurrent-responses.json" \
  "$results_dir/ready-after-cancellation.json" \
  "$results_dir/cache-probe/program.bin.original" \
  "$results_dir/cache-probe/program.bin.recovered" \
  "$results_dir/cache-recovery-ready.json" \
  "$results_dir/cache-recovery-server.log" \
  "$results_dir/auto-cpu-ready.json" \
  "$results_dir/soak.ndjson" \
  "$results_dir/server.log" \
  "$results_dir/auto-cpu-server.log" \
  >"$results_dir/SHA256SUMS"

jq -n \
  --arg schema gpt-oss-rs.xe-lifecycle-gate/v1 \
  --arg status pass \
  --argjson soak_seconds "$((soak_finished - soak_started))" \
  --argjson soak_requests "$soak_count" \
  --argjson wayland_processes "$wayland_processes" \
  --argjson start_swap_kib "$start_swap_kib" \
  --argjson end_swap_kib "$end_swap_kib" \
  --argjson start_system_swap_free_kib "$start_system_swap_free_kib" \
  --argjson end_system_swap_free_kib "$end_system_swap_free_kib" \
  --argjson start_major_faults "$start_major_faults" \
  --argjson end_major_faults "$end_major_faults" \
  --argjson peak_rss_kib "$peak_rss_kib" \
  --arg original_cache_sha256 "$cache_program_sha256" \
  --arg recovered_cache_sha256 "$recovered_cache_sha256" \
  '{schema:$schema,status:$status,ready_wire_shape_unchanged:true,streaming_and_nonstreaming_chat_and_responses:true,concurrent_requests:2,cancellation_before_and_after_submission_exercised:true,graceful_shutdown_exit_status:0,native_cache_reopen_and_corruption_recovery:true,original_cache_sha256:$original_cache_sha256,recovered_cache_sha256:$recovered_cache_sha256,automatic_disabled_record_skips_opencl_probe:true,automatic_effective_backend:"cpu",soak_seconds:$soak_seconds,soak_requests:$soak_requests,wayland_processes:$wayland_processes,start_swap_kib:$start_swap_kib,end_swap_kib:$end_swap_kib,swap_growth_kib:($end_swap_kib-$start_swap_kib),start_system_swap_free_kib:$start_system_swap_free_kib,end_system_swap_free_kib:$end_system_swap_free_kib,system_swap_used_growth_kib:($start_system_swap_free_kib-$end_system_swap_free_kib),start_major_faults:$start_major_faults,end_major_faults:$end_major_faults,major_fault_growth:($end_major_faults-$start_major_faults),peak_rss_kib:$peak_rss_kib,artifact_index:"SHA256SUMS"}' \
  >"$results_dir/summary.json"

trap - EXIT
echo "$results_dir/summary.json"
