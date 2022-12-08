#ifndef __GRAPHICSLIB_INCL__
#define __GRAPHICSLIB_INCL__

#define DIFFVERSION 22
#define GRAPHWIDTH 69

void PrintLogo();
void PrintLine();
void PrintSpaces(int);
void PrintStarLine(int);
void PrintProgress(double n);
void PrintFitTableHeader();
void PrintFitTableData(int, double, double, double, double, double, double, double, double);
void PrintProcessInfoHeader();
void PrintProcessInfo(std::string, std::string, int, int, int);
#endif
