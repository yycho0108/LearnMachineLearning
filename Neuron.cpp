#include "Neuron.h"

static auto f = std::bind(std::uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(time(0)));

Neuron::Connection::Connection(){
	weight = f();
	delta = 0;
}

Neuron::Connection::Connection(double w){
	weight = w;
	delta = 0;
}

Neuron::Neuron(int index, int outputSize)
	:index(index),outputSize(outputSize),weight(outputSize){
		//for(int i=0;i<outputSize;++i){
		//	weight.push_back(Connection());
		//}
}
Neuron::Neuron(int index, std::string& s)
	:index(index)
{
	outputSize = 0;
	std::stringstream ss(s);
	double w;
	while(ss >> w){
		weight.push_back(Connection(w));
		++outputSize;
	}
}
std::vector<double> Neuron::feedForward(double in){
	val = in;
	std::vector<double> res;
	for(auto &w : weight){
		res.push_back(val * w.weight);
	}
	return res; 
}

void Neuron::print(std::ostream& f_out){
	//f_out << '[';
	for(auto &w : weight){
		f_out << w.weight << ' ';
	}
	//f_out << ']' << std::endl;
	f_out << std::endl;
}

void Neuron::print(){
	this->print(std::cout);
	print(std::cout);
}
