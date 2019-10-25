#!/bin/bash

DIR=$(dirname "${BASH_SOURCE[0]}" )

pushd $DIR

echo 'backend : Agg' > matplotlibrc

for script in RXJ1713*.py; do
    echo ''
    echo "Running example $script..."
    time python $script
done

popd
