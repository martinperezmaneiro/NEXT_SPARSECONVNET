#!/usr/bin/bash
export SCN=$( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )/
export PYTHONPATH=$SCN:$PYTHONPATH
export TEST_DATA=$SCN//next_sparseconvnet/test_files/
export PATH=$SCN/bin:$PATH
export PATH=$SCN/scripts:$PATH
