#include "Predictor.h"


using namespace std;
using namespace Predictor;

vector<float> read_1d_array(ifstream &fin, int cols)
{
	vector<float> arr;
	float tmp_float;
	for (int n = 0; n < cols; n++)
	{
		fin >> tmp_float;
		arr.push_back(tmp_float);
	}
	return arr;
}

void missing_activation_impl(const string &act) {
	cout << "Activation " << act << " not defined!" << endl;
	cout << "Please add its implementation before use." << endl;
	exit(1);
}

void DataChunk1D::read_line(ifstream &fin,int field)
{
	string line;
	getline(fin, line);
	stringstream ss(line);
	for (int i = 0; i <= field; i++)
		getline(ss, line, '\t');
	stringstream ss_field(line);
	string buf;
	while (ss_field >> buf)
		data.push_back(atof(buf.c_str()));
	DataChunk *out = new DataChunk1D();

}

void DataChunk2D::read_from_file(const string &fname)
{
	ifstream fin(fname.c_str());
	fin >> m_depth >> m_length;
	for (int d = 0; d < m_depth; d++) {
		vector<float> tmp_single_depth = read_1d_array(fin, m_length);
		data.push_back(tmp_single_depth);
	}
	fin.close();
}

void DataChunk3D::read_from_file(const string &fname)
{
	ifstream fin(fname.c_str());
	fin >> m_depth >> m_row >> m_col;
	for (int d = 0; d < m_depth; d++) {
		vector<vector<float>> tmp_single_depth;
		for (int r = 0; r < m_row; r++) {
			vector<float> tmp_row = read_1d_array(fin, m_col);
			tmp_single_depth.push_back(tmp_row);
		}
		data.push_back(tmp_single_depth);
	}
	fin.close();
}

//Embedding Layer
void LayerEmbedding::load_weights(ifstream &fin, ifstream &confin) {
	fin >> m_vocab_size >> m_dimension;
	for (int i = 0; i < m_vocab_size; i++)
	{
		vector<float> tmp_emb = read_1d_array(fin, m_dimension);
		m_embs.push_back(tmp_emb);
	}
}

DataChunk* LayerEmbedding::compute_output(DataChunk* dc) {
	vector<vector<float>> tmp_embs;
	vector<float> input = dc -> get_1d();
	for (unsigned int i = 0; i < input.size(); i++)
		tmp_embs.push_back(m_embs[input[i]]);
	DataChunk *out = new DataChunk2D();
	out->set_data(tmp_embs);
	return out;
}

//Dense Layer
void LayerDense::load_weights(ifstream &fin, ifstream &confin) { // Dimensions: input * output
	fin >> m_input_cnt >> m_output_cnt;
	for (int i = 0; i < m_input_cnt; i++)
	{
		vector<float> tmp_single_row = read_1d_array(fin, m_output_cnt);
		m_weights.push_back(tmp_single_row);
	}
	m_bias = read_1d_array(fin, m_output_cnt);
}

DataChunk* LayerDense::compute_output(DataChunk* dc) {
	vector<float> y_ret(m_weights[0].size(), 0.0);
	vector<float> im = dc->get_1d();
	for (unsigned int i = 0; i < m_weights.size(); i++)  //iter over input
		for (unsigned int j = 0; j < m_weights[0].size(); j++) //iter over output
			y_ret[j] += m_weights[i][j] * im[i];
	for (unsigned int j = 0; j < m_bias.size(); j++)
		y_ret[j] += m_bias[j];
	DataChunk *out = new DataChunk1D();
	out->set_data(y_ret);
	return out;
}

//Activation Layer
DataChunk* LayerActivation::compute_output(DataChunk* dc) {
	if (dc->get_data_dim() == 3){
		vector<vector<vector<float>>> y = dc->get_3d();
		if (m_activation_type == "relu") {
			for (unsigned int i = 0; i < y.size(); i++)
				for (unsigned int j = 0; j < y[0].size(); j++)
					for (unsigned int k = 0; k < y[0][0].size(); k++)
						if (y[i][j][k] < 0) y[i][j][k] = 0;
			DataChunk *out = new DataChunk3D();
			out->set_data(y);
			return out;
		}
		else {
			missing_activation_impl(m_activation_type);
		}
	}
	else if (dc->get_data_dim() == 2) {
		vector<vector<float>> y = dc->get_2d();
		if (m_activation_type == "relu") {
			for (unsigned int i = 0; i < y.size(); i++)
				for (unsigned int j = 0; j < y[0].size(); j++)
					if (y[i][j] < 0) y[i][j] = 0;
			DataChunk *out = new DataChunk2D();
			out->set_data(y);
			return out;
		}
		else {
			missing_activation_impl(m_activation_type);
		}
	}
	else if (dc->get_data_dim() == 1) {
		vector<float> y = dc->get_1d();
		if (m_activation_type == "relu") {
			for (unsigned int i = 0; i < y.size(); i++)
				if (y[i] < 0) y[i] = 0;
			DataChunk *out = new DataChunk1D();
			out->set_data(y);
			return out;
		}
		else if (m_activation_type == "sigmoid")
		{
			for (unsigned int i = 0; i < y.size(); i++)
				y[i] = 1.0/(1+exp(-y[i]));
			DataChunk *out = new DataChunk1D();
			out->set_data(y);
			return out;
		}
		else{
			missing_activation_impl(m_activation_type);
		}
	}
	else { throw "Data dim not supported."; }
	return dc;
}
//Conv1D
void LayerConv1D::load_weights(ifstream &fin, ifstream &confin) {
	confin >> m_border_mode;
	fin >> m_filter_length >> m_input_dim >> m_nb_filter;
	for (int i = 0; i < m_filter_length; i++)
	{
		vector<vector<float>> tmp_filter;
		for (int j = 0; j < m_input_dim; j++)
		{
			vector<float> tmp_single_row = read_1d_array(fin, m_nb_filter);
			tmp_filter.push_back(tmp_single_row);
		}
		m_filters.push_back(tmp_filter);
	}
	m_bias = read_1d_array(fin, m_nb_filter);
}

DataChunk* LayerConv1D::compute_output(DataChunk* dc)
{
	unsigned int st_x = (m_filter_length - 1)/2;
	auto const & vec = dc->get_2d();
	size_t size_x = (m_border_mode == "valid") ? vec.size() - 2 * st_x : vec.size();
	vector<vector<float>> ret(size_x, vector<float>(m_nb_filter, 0));
	for(int i = 0; i < m_filter_length; i++) //Loop over filter length
		for (int j = 0; j < m_nb_filter; j++) //Loop over filter
			for (unsigned int k = 0; k < size_x; k++) //Loop over output vec
				for (int l = 0; l < m_input_dim; l++) //Loop over input dim
				{
					int tmp = (m_border_mode == "valid") ? k + i : k + i - st_x;
					if (tmp >= 0)
						ret[k][j] += vec[tmp][l] * m_filters[i][l][j];
				}
	for (unsigned int i = 0; i < size_x; i++)
		for (int j = 0; j < m_nb_filter; j++)
			ret[i][j] += m_bias[j];
	DataChunk *out = new DataChunk2D();
	out->set_data(ret);
	return out;
}
//MaxPooling : May not be right, easist version
DataChunk* LayerMaxPooling::compute_output(DataChunk* dc)
{
	if (dc->get_data_dim() == 2)
	{
		vector<vector<float>> y = dc->get_2d();
		vector<float> tmpMax(y[0].size());
		for (unsigned int i = 0; i < y.size(); i++)
			for (unsigned int j = 0; j < y[0].size(); j++)
				if (i == 0 || y[i][j] > tmpMax[j])
					tmpMax[j] = y[i][j];
		DataChunk *out = new DataChunk1D();
		out->set_data(tmpMax);
		return out;
	}
	else { throw "Data dim not supported."; }


}

DataChunk* LayerFlatten::compute_output(DataChunk* dc)
{
	if (dc->get_data_dim() == 2)
	{
		DataChunk *out = new DataChunk1D();
		out->set_data(dc->get_2d()[0]);
		return out;
	}
	else
	{
		throw "Data dim not supported.";
	}
}
//Whole Model
Model::Model(const string &input_fname, bool verbose) :m_verbose(verbose) {
	load_weights(input_fname);
}

vector<float> Model::compute_output(DataChunk *dc)
{
	DataChunk *in = dc;
	DataChunk *out;
	for (unsigned int l = 0; l < m_layers.size(); l++)
	{
		out = m_layers[l]->compute_output(in);
//		out->show_name();
//		out->show_value();
		if (in != dc) delete in;
		in = out;
	}

	vector<float> output = out->get_1d();
	delete out;
	return output;
}

bool Model::load_weights(const string &input_config) {
	if (m_verbose) cout << "Reading model from " << input_config << endl;
	ifstream fin(input_config.c_str());
	fin >> m_layers_cnt;
	string layer_type = "";
	int tmp_layerNo;
	int layer_number;
	if (m_verbose) cout << "Layers count " << m_layers_cnt << endl;
	for (int layer = 0; layer < m_layers_cnt; layer++)
	{
		fin >> tmp_layerNo >> layer_type >> layer_number;
		if (m_verbose) cout << "Layer " << tmp_layerNo << " " << layer_type << " " << layer_number << endl;
		Layer *l = 0L;
		string layerFileName = layer_type + "_" + to_string(layer_number) + ".txt";
		ifstream layerfile(layerFileName);
		if (layerfile && m_verbose)
			cout << "Reading Layer" << tmp_layerNo << " From " << layerFileName << endl;
		if (layer_type == "convolution1d")
			l = new LayerConv1D();
		else if (layer_type == "dense")
			l = new LayerDense();
		else if (layer_type == "activation")
			l = new LayerActivation();
		else if (layer_type == "maxpooling1d")
			l = new LayerMaxPooling();
		else if (layer_type == "embedding")
			l = new LayerEmbedding();
		else if (layer_type == "flatten")
			l = new LayerFlatten();
		if (l == 0L) {
			cout << "Layer is empty, maybe it is not defined? Cannot define network. " << endl;
			return false;
		}
		l->load_weights(layerfile,fin);
		m_layers.push_back(l);
		layerfile.close();
	}
	fin.close();
	return true;
}