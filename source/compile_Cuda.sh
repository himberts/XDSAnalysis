#! /bin/bash

echo "Build Program ..."
nvcc -I/usr/local/include/gsl -L/usr/local/lib -lgsl libs/GraphicsLib.cu libs/CudaCoreFunctions.cu libs/diffusetoolbox.cpp libs/diffuselibCuda.cu diffuse_analysis_v22_CUDA.cu -o ../bin/diffuse_analysis_v22_CUDA -lgsl -lgslcblas -lm -lboost_program_options
echo "Done"
