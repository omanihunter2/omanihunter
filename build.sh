#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

export CAMELTOOLS_DATA="$PWD/camel_tools_data"

camel_data -i morphology-db-msa-r13
camel_data -i disambig-mle-calima-msa-r13

echo "CAMeL data directory:"
find "$CAMELTOOLS_DATA" -maxdepth 5 -type f | head -50
