#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <functional>
#include <cmath>
#include <sstream>
#include <list>
#include <fstream>
#include <cassert>

#define ETA 0.15
#define ALPHA 0.3

void print(std::vector<double>& v){
	for(auto i : v){
		std::cout << i << ' ';
	}
	std::cout << std::endl;
}

using namespace std;
double transfer(double in){
	//return tanh(in);
	return 1 / (1 +exp(-in));
}
double transferDerivative(double in){
	return 1-in*in;
	//return exp(-in) / (pow((1+exp(-in)),2));
}
std::vector<double> transfer(std::vector<double>& in){
	std::vector<double> res;
	res.reserve(in.size());
	for(auto i:in){
		res.push_back(transfer(i));
	}
	return res;
}
std::vector<double> transferDerivative(std::vector<double>& in){
	std::vector<double> res;
	res.reserve(in.size());
	for(auto i:in){
		res.push_back(transferDerivative(i));
	}
	return res;
}

class Neuron{
	friend class Layer;
	friend class Net;
	private:
		int index , outputSize;
		double val;
		std::vector<double> weight;
		std::vector<double> deltaWeight;
	public:
		Neuron(int index, int outputSize);
		std::vector<double> output(double in);
};

Neuron::Neuron(int index, int outputSize):index(index),outputSize(outputSize){
	static auto f = std::bind(uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(time(0))); //random
	for(int i=0;i<outputSize;++i){
		weight.push_back(f());
		deltaWeight.push_back(0);
	}
}

std::vector<double> Neuron::output(double in){
	val = in; //what I deploy
	std::vector<double> out;
	for(int i=0;i<outputSize;++i){
		out.push_back(weight[i] * in);
	}
	return out;
}


class Layer{
	friend class Net;
	private:
		int index, inputSize ,outputSize;
		std::vector<Neuron> layer;
		std::vector<double> gradient;
		void accumulate(std::vector<double>& dst, std::vector<double>& add);
	public:
		std::vector<double> feedForward(std::vector<double> input);
		Layer(int index, int inputSize, int outputSize);
		void calcGradient(const Layer& next);
		void calcGradient(const std::vector<double>& next);
		static std::vector<double> calcGradient(const std::vector<double>& targ, const std::vector<double>& res);
		void update(Layer& prev);
};

void Layer::accumulate(std::vector<double>& dst, std::vector<double>& add){
	assert(dst.size() == add.size());
	for(int i=0;i<(int)dst.size();++i){
		dst[i] += add[i];
	}
}

Layer::Layer(int index, int inputSize, int outputSize):
	index(index),inputSize(inputSize),outputSize(outputSize){
	for(int i=0;i<inputSize;++i){
		layer.push_back(Neuron(i,outputSize)); //outputsize fix
	}
}

std::vector<double> Layer::feedForward(std::vector<double> input){
	assert(input.size() == inputSize);
	std::vector<double> res(outputSize);
		for(int i=0;i<inputSize;++i){
			auto &neuron = layer[i];
			auto out = neuron.output(input[i]);
			accumulate(res,out);
		}
	return transfer(res);
}

std::vector<double> Layer::calcGradient(const std::vector<double>& targ, const std::vector<double>& res){
	std::vector<double> gradient;
	for(int i=0;i<(int)res.size();++i){
		auto delta = targ[i] - res[i];
		gradient.push_back(delta * transferDerivative(res[i]));
	}
	return gradient;
}

void Layer::calcGradient(const Layer& next){
	calcGradient(next.gradient);
}
void Layer::calcGradient(const std::vector<double>& next){ //gradient of next layer
	gradient.clear();
	for(auto& n : layer){ //per neuron
		double delta = 0;
		for(int i=0;i<(int)n.weight.size();++i){
			delta += n.weight[i] * next[i];
		}
		gradient.push_back(delta * transferDerivative(n.val));
	}

}
void Layer::update(Layer& prev){
	for(int i=0; i<(int)prev.layer.size(); ++i){
		auto& neuron = prev.layer[i];
		double oldDelta = neuron.deltaWeight[index];
		double newDelta = ETA * neuron.val * gradient[i]
						  + ALPHA * oldDelta;
		neuron.deltaWeight[index] = newDelta;
		neuron.weight[index] += newDelta;
	}
}
class Net{
	private:
		std::vector<Layer> layers;
	public:
		std::vector<double> feedForward(const std::vector<double>& input);
		void backProp(const std::vector<double>& inputVals,const std::vector<double>& targetVals);
		Net(std::vector<int>& topology);
};

Net::Net(std::vector<int>& topology){
	int l = topology.size(); //3
	for(int i=0;i<l-1;++i){
		layers.push_back(Layer(i,topology[i],topology[i+1]));
	}
}

std::vector<double> Net::feedForward(const std::vector<double>& input){
	std::vector<double> res(input.begin(), input.end());
	for(auto& l : layers){
		res = l.feedForward(res);
	}
	return res;
}
void Net::backProp(
		const std::vector<double>& output,
		const std::vector<double>& targetVals
		){
	//Calculate Net Error
	double err=0.0;
	for(int i=0;i<(int)output.size();++i){
		err += abs(targetVals[i] - output[i]);
	}
	err = sqrt(err/output.size());
	std::cout << "ERROR : " << err << std::endl;
	
	//Calculate layer gradients
	auto gradient = Layer::calcGradient(targetVals,output);
	layers.back().calcGradient(gradient);	
	//Calculate gradients on hidden layers
	for(int i = layers.size()-2; i>=0; --i){
		auto& l = layers[i];
		l.calcGradient(layers[i+1]);
	}

	for(int i= layers.size()-1; i > 0; --i){
		auto& l = layers[i];
		l.update(layers[i-1]);
	}

	/*
	 * std::cout << "___" << std::endl;
	for(auto& l : layers){
		for(auto& n : l.layer){
			print(n.weight);
		}
		std::cout << std::endl;
	}
	std::cout << "___" << std::endl;
	*/
}

std::vector<int> parseTopology(std::ifstream& f_in){
	std::string s;
	std::vector<int> topology;
	std::getline(f_in,s);
	std::stringstream ss(s);
	int i;
	while(ss >> i){
		topology.push_back(i);
	}
	return topology;

}

std::pair<std::vector<double>,std::vector<double>> parseInput(std::ifstream& f_in, int in, int out, bool& success){
	double a;
	std::vector<double> input;
	std::vector<double> output;
	for(int i=0;i<in;++i){
		if(!(f_in >> a)){
			success = false;
			return std::make_pair(input,output);
		}
		input.push_back(a);
	}
	for(int i=0;i<out;++i){
		if(!(f_in >> a)){
			success = false;
			return std::make_pair(input,output);
		}
		output.push_back(a);
	}
	return std::make_pair(input,output);
}

int main(){
	std::ifstream f_in("../train/training.txt");
	auto topology = parseTopology(f_in);
	auto net = Net(topology);
	bool success = true;
	while(1){
		auto data = parseInput(f_in,topology.front(),topology.back(),success);
		if(!success)
			break;
		for(auto i : data.first){
			std::cout << i << ' ';
		}
		std::cout << "-->";
		auto res = net.feedForward(data.first);
		for(auto i : res){
			std::cout << i << ' ';
		}
		std::cout << std::endl;
		net.backProp(res,data.second);	
	}

	return 0;
}
