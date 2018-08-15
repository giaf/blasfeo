#! /bin/bash

find .. -name "*.h" -exec bash -c 'for file; do cat new_lic.h >temp && tail -n +28 $file >> temp && cat temp > $file; done;' {} +
find .. -name "Makefile.*" -exec bash -c 'for file; do cat new_lic.makefile >temp && tail -n +28 $file >> temp && cat temp > $file; done;' {} +
