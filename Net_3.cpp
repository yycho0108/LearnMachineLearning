#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <functional>
#include <ctime>
#include <sstream>
#include <cassert>

#define ETA (.15)
#define ALPHA (.50)

/* ***** Utility Functions ***** */
typedef std::string str;
typedef unsigned uint;
template<typename T>
void print(std::vector<T> v){
	for(auto i : v){
		std::cout << i << ' ';
	}
	std::cout << std::endl;

}

double transfer(double in){
	return tanh(in);
}
double transferPrime(double in){
	return 1.0-in*in;
}


/*double transfer(double in){
	
	//return 1/(1+exp(-in));
	//return (1 / (1 +exp(-in)) - 0.5) *4;
}
double transferPrime(double in){
	return exp(-in)/pow(1+exp(-in),2);

	//return 4*exp(-in) / (pow((1+exp(-in)),2));
}*/

std::vector<double> transfer(std::vector<double>& in){
	std::vector<double> res;
	for(auto i : in){
		res.push_back(transfer(i));
	}
	return res;
}
std::vector<double> transferPrime(std::vector<double>& in){
	std::vector<double> res;
	for(auto i : in){
		res.push_back(transferPrime(i));
	}
	return res;
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

/* ***** Class Declarations ***** */
struct Connection{
	double weight,delta;
	Connection(){
		static auto f = std::bind(std::uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(time(0)));
		weight = f();
		delta = 0;
	}
};

class Neuron{
	friend class Net;
	friend class Layer;
	private:
		int index;
		int outputSize;
		double val;
		std::vector<Connection> weight;
	public:
		Neuron(int index, int outputSize);
		std::vector<double> feedForward(double in);
		void print();
};

class Layer{
	friend class Net;
	private:
		int index;
		int inputSize,outputSize;
		std::vector<Neuron> layer;
		std::vector<double> gradient;
	public:
		Layer(int index, int inputSize, int outputSize);
		std::vector<double> feedForward(std::vector<double>& input);
		void accumulate(std::vector<double>& dst, std::vector<double>& add);
		std::vector<double> calcGradient(std::vector<double>& next);
		std::vector<double> calcGradient(Layer& next);
		void update(Layer& prev);
		void print();
};

class Net{
	private:
		std::vector<Layer> net;
	public:
		Net(std::vector<int>& topology);
		std::vector<double> feedForward(std::vector<double>& input);
		void backProp(std::vector<double>& target);
		void print();
		void report(std::pair<std::vector<double>,std::vector<double>>& input);
};
/* ***** Class Definitions ***** */

//Neuron
Neuron::Neuron(int index, int outputSize)
	:index(index),outputSize(outputSize),weight(outputSize){
		//for(int i=0;i<outputSize;++i){
		//	weight.push_back(Connection());
		//}
}

std::vector<double> Neuron::feedForward(double in){
	val = in;
	std::vector<double> res;
	for(auto &w : weight){
		res.push_back(val * w.weight);
	}
	return res; 
}

void Neuron::print(){
	std::cout << '[';
	for(auto &w : weight){
		std::cout << w.weight << ' ';
	}
	std::cout << ']' << std::endl;
}
//Layer

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
	/*for(int i=0;i<next.inputSize;++i){
		double delta = 0;
		for(auto& n : layer){
			delta += n.weight[i].weight * next.gradient[i];
		}
		gradient.push_back(delta * transferPrime(this->layer[i].val));
	}*/
	return gradient;
}

void Layer::update(Layer& prev){
	for(int i=0; i<inputSize;++i){
		auto &n = layer[i];
		auto &g = gradient[i];
		for(int j=0;j<prev.inputSize;++j){
			auto &neuron = prev.layer[j];
			double oldDelta = neuron.weight[n.index].delta;
			double newDelta = ETA * neuron.val * g + ALPHA * oldDelta;
			neuron.weight[i].delta = newDelta;
			neuron.weight[i].weight += newDelta;
		}
	}
	/*for(auto& n : layer){
		for(int i=0; i < prev.inputSize;++i){
			auto &neuron = prev.layer[i];
			double oldDelta = neuron.weight[n.index].delta;
			double newDelta = ETA * neuron.val * gradient[i]
		}
	}

	for(int i=0; i<prev.inputSize;++i){
		auto &neuron = prev.layer[i];
		double oldDelta = neuron.weight[index].delta;
		double newDelta = ETA * neuron.val * prev.gradient[i]
		+ ALPHA * oldDelta;
		//std::cout << '[' << index <<',' << i <<  ':' << neuron.val << ']';
		neuron.weight[i].delta = newDelta;
		neuron.weight[i].weight += newDelta;
	}*/
}
void Layer::print(){
	std::cout << "LAYER <";
	for(auto& n : layer){
		n.print();
		std::cout << ' ';
	}
	std::cout << ">" << std::endl;
}
//Net

std::vector<double> Net::feedForward(std::vector<double>& input){
	std::vector<double> next(input.begin(),input.end());

	for(uint i=0; i<net.size(); ++i){
		auto& l = net[i];
		next = l.feedForward(next);
	}
	return next;	

}
void Net::backProp(std::vector<double>& target){
	net.back().calcGradient(target);
	for(int i = net.size() - 2; i > 0; --i){
		net[i].calcGradient(net[i+1]);
	}
	for(int i = net.size() - 1; i > 0; --i){
		net[i].update(net[i-1]);
	}
}
Net::Net(std::vector<int>& topology){
	for(uint i=0;i<topology.size();++i){
		net.push_back(Layer(i,topology[i], (i == topology.size()-1)?0:topology[i+1]));
	}
}
void Net::print(){
	for(auto& l : net){
		l.print();
	}
}

void Net::report(std::pair<std::vector<double>,std::vector<double>>& input){
	static int count = 0;
	std::cout << '[' << ++count << ']';
	std::cout << "INPUT : ";
	for(auto &d : input.first){
		std::cout << d << ' ';
	}
	std::cout << " | ";
	std::cout << "OUTPUT : ";
	for(auto &o : input.second){
		std::cout << o << ' ';
	}
	std::cout << " | ";
	std::cout << "RESULT : ";
	for(auto&n : net.back().layer){
		std::cout << n.val << ' ';
	}
	std::cout << std::endl;
}


int main(){
	std::ifstream f_in("training.txt");
	auto topology = parseTopology(f_in);
	auto inNum = topology.front();
	auto outNum = topology.back();
	Net net(topology);
	net.print();	
	while(1){
		bool success = true;
		auto input = parseInput(f_in, inNum,outNum,success);
		if(!success)
			break;
		net.feedForward(input.first);
		net.report(input);
		//net.print();
		net.backProp(input.second);
	}
	
	//
	return 0;
}
