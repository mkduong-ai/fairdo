#!/bin/bash

# Find and remove all .rst files in the current directory and subdirectories,
# except those named 'index.rst'
find . -name "*.rst" ! -name "index.rst" -type f -exec rm -f {} \;

# Run sphinx-apidoc
sphinx-apidoc --module-first -f -o . ../src/fado/

# Remove older builds if they exist
if [ -d "_build" ]; then
    rm -r _build
fi

# Build the HTML docs
make html
