#!/bin/bash

# Run sphinx-apidoc
sphinx-apidoc -f -o . ../src/fado/

# Build the HTML docs
make html
