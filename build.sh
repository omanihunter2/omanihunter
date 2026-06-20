#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

camel_data -i morphology-db-msa-r13
camel_data -i disambig-mle-calima-msa-r13
