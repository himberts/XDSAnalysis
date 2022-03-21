#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <string.h>
#include "GraphicsLib.h"

void PrintLogo(){
  std::cout<<std::endl;
  PrintStarLine(2);
  std::cout<<std::endl;
  PrintSpaces((GRAPHWIDTH-27)/2);
  std::cout<<"Diffuse XRD Analysis - CUDA\n"<<std::endl;
  PrintSpaces((GRAPHWIDTH-10)/2);
  std::cout<<"Version "<<DIFFVERSION<<"\n"<<std::endl;
  PrintStarLine(2);
  std::cout<<std::endl;
  std::cout<<std::endl;
}

void PrintSpaces(int NumSpaces){
  for(int n=0; n<NumSpaces; n++){
      std::cout<<" ";
  }
}

void PrintLine(){
  std::cout<<std::endl;
  for(int n=0; n<GRAPHWIDTH; n++){
      std::cout<<"-";
  }
  std::cout<<std::endl;
  std::cout<<std::endl;
}

void PrintStarLine(int NumLines){
  for(int k=0;k<NumLines; k++){
    for(int n=0; n<GRAPHWIDTH; n++){
        std::cout<<"*";
    }
    std::cout<<std::endl;
  }
}

void PrintProgress(double Percentage){
  std::cout<<"\rProgress: "<<Percentage<<"%\t[";
  for(int n=0; n<floor(Percentage/2); n++){
    std::cout<<"#";
  }
  for(int n=0; n<ceil((100-Percentage)/2); n++){
    std::cout<<" ";
  }
  std::cout<<"]"<<std::flush;
}

void PrintFitTableHeader(){
  std::cout<<"|"<<std::setw(10)<<"Iteration"<<"|"<<std::setw(10)<<"Eta"<<"|"<<std::setw(10)<<"Zeta"<<"|"<<std::setw(10)<<"Xisquare"<<"|"<<std::setw(10)<<"Size"<<"|"<<std::endl;
  for(int k=1; k<=56; k++){
    std::cout<<"_";
  }
  std::cout<<std::endl;
}
void PrintFitTableData(int iteration, double eta, double zeta, double XiSquare, double size){
  std::cout<<"|"<<std::setw(10)<<iteration<<"|"<<std::setw(10)<<eta<<"|"<<std::setw(10)<<zeta<<"|"<<std::setw(10)<<XiSquare<<"|"<<std::setw(10)<<size<<"|"<<std::endl;
}


void PrintProcessInfoHeader(){
  std::cout<<"|"<<std::setw(20)<<"Function"<<"|"<<std::setw(10)<<"Processor"<<"|"<<std::setw(15)<<"Current Device"<<"|"<<std::setw(9)<<"# Blocks"<<"|"<<std::setw(9)<<"# Threats"<<"|"<<std::endl;
  for(int k=1; k<=69; k++){
    std::cout<<"_";
  }
  std::cout<<std::endl;
}

void PrintProcessInfo(std::string Function, std::string Processor, int CurrDevice, int numBlocks, int blockSize){
  std::cout<<"|"<<std::setw(20)<<Function<<"|"<<std::setw(10)<<Processor<<"|"<<std::setw(15)<<CurrDevice<<"|"<<std::setw(9)<<numBlocks<<"|"<<std::setw(9)<<blockSize<<"|"<<std::endl;
}
