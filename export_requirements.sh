#!/usr/bin/env bash
set -e
uv export --format requirements-txt --no-hashes -o requirements.txt
sed -i '/^-e \.$\|^\.$/ d' requirements.txt
echo "krunic @ file:///home/ljubomir/github/krunic" >> requirements.txt
echo "requirements.txt updated"
