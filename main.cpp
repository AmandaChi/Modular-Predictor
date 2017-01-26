#include "Predictor.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace Predictor;

int main()
{
	
	string fname = "C:\\Users\\bingychi\\Documents\\GitHub\\Modular-Predictor\\Debug\\vector.txt";
	ifstream fin(fname.c_str());
	for (int i = 0; i < 10; i ++)
	{
		DataChunk *sample = new DataChunk1D();
		sample->read_line(fin);
		sample->show_name();
		sample->show_value();
	}
	
}

