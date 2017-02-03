#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>

using namespace std;
namespace Predictor
{
	class DataChunk;
	class DataChunk1D;
	class DataChunk2D;
	class DataChunk3D;

	class Layer;
	class LayerEmbedding;
	class LayerMaxPooling;
	class LayerActivation;
	class LayerConv1D;
	class LayerDense;
	

	class Model;
}

class Predictor::DataChunk {
public:
	virtual ~DataChunk() {}
	virtual void set_data(vector<vector<vector<float>>> const &) { throw "not implemented"; };
	virtual void set_data(vector<vector<float>> const &) { throw "not implemented"; };
	virtual void set_data(vector<float> const &) { throw "not implemented"; };
	virtual void read_from_file(const string &fname) {};
	virtual void read_line(ifstream &fin) {};
	virtual int get_data_dim(void) const { return 0; }
	virtual vector<float> const & get_1d() const { throw "not implemented"; };
	virtual vector<vector<float>> const & get_2d() const { throw "not implemented"; };
	virtual vector<vector<vector<float>>> const & get_3d() const { throw "not implemented"; };
	virtual void show_name() = 0;
	virtual void show_value() = 0;
};

class Predictor::DataChunk1D: public Predictor::DataChunk {
public:
	vector<float> data;
	vector<float> const & get_1d() const { return data; }
	void set_data(vector<float> const & d) { data = d; };
	int get_data_dim(void) const { return 1; }
	void show_name() {
		cout << "DataChunk1D " << data.size() << endl;
	}
	void show_value()
	{
		cout << "DataChunk1D values:" << endl;
		for (unsigned i = 0; i < data.size(); i++)
			cout << data[i] << " ";
		cout << endl;
	}
	void read_from_file(const string &fname) {};
	void read_line(ifstream &fin);
};

class Predictor::DataChunk2D : public Predictor::DataChunk {
public:
	vector<vector<float>> const & get_2d() const { return data; };
	void set_data(vector<vector<float>> const & d) { data = d; };
	int get_data_dim(void) const { return 2; }
	void show_name()
	{
		cout << "DataChunk2D " << data.size() << "x" << data[0].size() << endl;
	}
	void show_value()
	{
		cout << "DataChunk2D values:" << endl;
		for (unsigned int i = 0; i < data.size(); i++) {
			cout << "Kernel " << i << endl;
			for (unsigned int j = 0; j < data[0].size(); j++) {
				cout << data[i][j] << " ";
			}
			cout << endl;
		}
	}
	void read_from_file(const string &fname);
	vector<vector<float>> data;
	int m_depth;
	int m_length;
};

class Predictor::DataChunk3D: public Predictor::DataChunk {
public:
	vector<vector<vector<float>>> const & get_3d() const { return data; };
	void set_data(vector<vector<vector<float>>> const & d) { data = d; };
	int get_data_dim(void) const { return 3; }
	void show_name()
	{
		cout << "DataChunk3D " << data.size() << "x" << data[0].size() << "x" << data[0][0].size() << endl;
	}
	void show_value()
	{
		cout << "DataChunck3D values:" << endl;
		for (unsigned int i = 0; i < data.size(); i++) {
			cout << "Kernel " << i << endl;
			for (unsigned int j = 0; j < data[0].size(); j++) {
				for (unsigned int k = 0; k < data[0][0].size(); k++)
					cout << data[i][j][k] << " ";
				cout << endl;
			}
		}
	}
	void read_from_file(const string &fname);
	vector<vector<vector<float>>> data;
	int m_depth;
	int m_row;
	int m_col;

};

class Predictor::Layer {
public:
	virtual void load_weights(ifstream &fin, ifstream &confin) = 0;
	virtual Predictor::DataChunk* compute_output(Predictor::DataChunk*) = 0;
	Layer(string name) : m_name(name) {}
	virtual ~Layer() {}
	string get_name() { return m_name; }
	string m_name;
};

class Predictor::LayerEmbedding : public Predictor::Layer {
public:
	LayerEmbedding() : Layer("Embedding") {};
	void load_weights(ifstream &fin, ifstream &confin);
	Predictor::DataChunk* compute_output(Predictor::DataChunk*);
	vector<vector<float>> m_embs;
	int m_vocab_size;
	int m_dimension;
};
class Predictor::LayerMaxPooling : public Predictor::Layer {
public:
	LayerMaxPooling() : Layer("MaxPooling") {};
	void load_weights(ifstream &fin, ifstream &confin) { }
	Predictor::DataChunk* compute_output(Predictor::DataChunk*);
	//int m_pool;
};

class Predictor::LayerActivation : public Predictor::Layer {
public:
	LayerActivation() : Layer("Activation") {};
	void load_weights(ifstream &fin, ifstream &confin) { confin >> m_activation_type; }
	Predictor::DataChunk* compute_output(Predictor::DataChunk*);
	string m_activation_type;
};

class Predictor::LayerConv1D : public Predictor::Layer {
public:
	LayerConv1D() : Layer("Conv1D") {}
	void load_weights(ifstream &fin, ifstream &confin);
	Predictor::DataChunk* compute_output(Predictor::DataChunk*);
	vector<vector<vector<float>>> m_filters;
	vector<float> m_bias;
	string m_border_mode;
	int m_filter_length;
	int m_nb_filter;
	int m_input_dim;
};

class Predictor::LayerDense : public Predictor::Layer {
public:
	LayerDense() : Layer("Dense") {}
	void load_weights(ifstream &fin, ifstream &confin);
	Predictor::DataChunk* compute_output(Predictor::DataChunk*);
	vector<vector<float>> m_weights; //input, output
	vector<float> m_bias;

	int m_input_cnt;
	int m_output_cnt;
};

class Predictor::Model {
public:
	Model(const string &input_config, bool verbose);
	~Model() { 
		for (unsigned int i = 0; i < m_layers.size(); i++)
			delete m_layers[i];
	}
	vector<float> compute_output(Predictor::DataChunk *dc);
private:
	bool load_weights(const string &input_config);
	int m_layers_cnt;
	vector<Predictor::Layer *> m_layers;
	bool m_verbose;

};


#endif