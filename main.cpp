#include "Predictor.h"


using namespace std;
using namespace Predictor;

int main()
{
	
	string fname = "C:\\Users\\bingychi\\Documents\\GitHub\\Modular-Predictor\\Debug\\CTRVec.txt";
	ifstream fin(fname.c_str());
	ofstream fout("res.txt");
	Model CNN("C:\\Users\\bingychi\\Documents\\GitHub\\Modular-Predictor\\Debug\\ctr_model_config.txt", 1);
	for (int i = 0; i < 9; i ++)
	{
		DataChunk *sample = new DataChunk1D();
		sample->read_line(fin,0);
		sample->show_name();
		sample->show_value();
		vector<float> res;
		res = CNN.compute_output(sample);
		//cout << "Output Vector" << endl;
		for (unsigned int j = 0; j < res.size(); j++)
			fout << to_string(res[j]) << " ";
		fout << endl;
	}
}
