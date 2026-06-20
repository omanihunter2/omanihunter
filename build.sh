#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

python -m camel_tools.data -i disambig-mle-calima-msa-r13
python -m camel_tools.data -i morphology-db-msa-r13
