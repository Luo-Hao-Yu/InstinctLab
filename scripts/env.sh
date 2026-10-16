#!/usr/bin/env bash
set -euo pipefail
container_name=instinctlab-dev
if ! docker inspect "$container_name" >/dev/null 2>&1; then
    echo 'Environment container is missing. See ENVIRONMENT.md.' >&2
    exit 1
fi
if [ "$(docker inspect -f '{{.State.Running}}' "$container_name")" != true ]; then
    docker start "$container_name" >/dev/null
fi
exec_flags=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    exec_flags+=(-t)
fi
if [ "$#" -eq 0 ]; then
    exec docker exec "${exec_flags[@]}" -w /workspace/instinctlab "$container_name" bash
fi
exec docker exec "${exec_flags[@]}" -w /workspace/instinctlab "$container_name" /isaac-sim/python.sh "$@"
