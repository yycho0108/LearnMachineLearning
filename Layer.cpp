#include "Layer.h"


std::vector<double> Layer::feedForward(std::vector<double>& input){
	assert(layer.size() == input.size());
	std::vector<double> res(outputSize);
	for(uint i=0; i < layer.size(); ++i){
		auto& neuron = layer[i];
		auto o = neuron.feedForward(input[i]);
		accumulate(res,o);
	}
	return transfer(res);
}
Layer& Layer::feedForward(Layer& input){
	for(uint i=0; i<layer.size(); ++i){
		auto& neuron = layer[i];
		double sum = 0.0;
		for(int j=0;j<input.inputSize;++j){
			sum += input.layer[j].val * input.layer[j].weight[i].weight;
		}
		neuron.val = transfer(sum);
	}
	return *this;
}
void Layer::accumulate(std::vector<double>& dst, std::vector<double>& add){
	assert(dst.size() == add.size());
	for(uint i=0;i<dst.size();++i){
		dst[i] += add[i];
	}
}

Layer::Layer(int index, int inputSize, int outputSize)
	:index(index),inputSize(inputSize),
	outputSize(outputSize){
		for(int i=0;i<inputSize;++i){
			layer.push_back(Neuron(i,outputSize));
		}
}
Layer::Layer(int index, std::istream& f_in):
	index(index)
{
	std::string s;
	int i=0;
	while(std::getline(f_in,s)){
		if(s=="<") //beginning
			continue;
		if(s==">")
			break;
		layer.push_back(Neuron(i,s));
		++i;
	}
	outputSize = layer.front().outputSize;
	inputSize = layer.size();
}
std::vector<double> Layer::calcGradient(std::vector<double>& next){
	gradient.clear();
	for(uint i=0;i<next.size();++i){
		auto& neuron = this->layer[i];
		auto delta = next[i] - neuron.val;
		gradient.push_back(delta * transferPrime(neuron.val));
	}
	return gradient;
}

std::vector<double> Layer::calcGradient(Layer& next){
	gradient.clear();
	for(auto& n : layer){ // n = neuron
		double delta = 0;
		for(int i=0; i < next.inputSize; ++i){
			delta += next.gradient[i] * n.weight[i].weight;
		}
		gradient.push_back(delta * transferPrime(n.val));
	}

	return gradient;
}

double LAMBDA = 0.001;
void Layer::update(Layer& prev){
	for(int i=0; i<inputSize;++i){
		//auto &n = layer[i];
		auto &g = gradient[i];
		for(int j=0;j<prev.inputSize;++j){
			auto &neuron = prev.layer[j];
			double oldDelta = neuron.weight[i].delta;
			double newDelta = ETA * neuron.val * g + ALPHA * oldDelta;
			//- ETA*LAMBDA*neuron.weight[i].weight; //regularization
			neuron.weight[i].delta = newDelta;
			neuron.weight[i].weight += newDelta;
		}
	}
}
void Layer::print(std::ostream& f_out){
	f_out << "<" << std::endl;
	for(auto& n : layer){
		n.print(f_out);
	}
	f_out << ">" << std::endl;
}

void Layer::print(){
	print(std::cout);
}
