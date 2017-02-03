#include "Predictor.h"


using namespace std;
using namespace Predictor;

int main()
{
	
	string fname = "C:\\Users\\bingychi\\Documents\\GitHub\\Modular-Predictor\\Debug\\vector.txt";
	ifstream fin(fname.c_str());
	Model CNN("C:\\Users\\bingychi\\Documents\\GitHub\\Modular-Predictor\\Debug\\model_config.txt", 1);
	for (int i = 0; i < 10; i ++)
	{
		DataChunk *sample = new DataChunk1D();
		sample->read_line(fin);
		sample->show_name();
		sample->show_value();
		cout << to_string(CNN.compute_output(sample)[0]) << endl;
	}
}

