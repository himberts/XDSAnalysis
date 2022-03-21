#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void CreateLogScaleVec(double* r, double StartR, double StopR, double StepSize){
    int Rlength = (StopR - StartR)/StepSize;
    for(int k = 0; k<=Rlength; k++){
      r[k] = pow(10,StartR + k*StepSize);
    }
}
