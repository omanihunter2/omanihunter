#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

export CAMELTOOLS_DATA="/opt/render/project/src/camel_tools_data"
mkdir -p "$CAMELTOOLS_DATA"

camel_data -i light

find "$CAMELTOOLS_DATA" -maxdepth 6 -type f | grep -E 'morphology.db|mle|calima' || true
