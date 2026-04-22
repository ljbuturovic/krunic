#!/usr/bin/env bash                                                                                                                                                                                                                                      
uv export --format requirements-txt --no-hashes -o requirements.txt
echo "krunic @ file:///home/ljubomir/github/krunic" >> requirements.txt
