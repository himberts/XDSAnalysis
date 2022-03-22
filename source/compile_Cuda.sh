#! /bin/bash

echo "Build Program ..."
/usr/local/cuda-11.2/bin/nvcc -arch=sm_80 -Wno-deprecated-gpu-targets -I/usr/local/include/gsl -L/usr/local/lib -lgsl libs/GraphicsLib.cu libs/CudaCoreFunctions.cu libs/diffusetoolbox.cpp libs/diffuselibCuda.cu diffuse_analysis_v21_CUDA.cu -o ../bin/diffuse_analysis_v21_CUDA -lgsl -lgslcblas -lm -lboost_program_options
echo "Done"
