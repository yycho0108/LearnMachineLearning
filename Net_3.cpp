#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <functional>
#include <ctime>
#include <sstream>
#include <cassert>

//#define ETA (.10)
//#define ALPHA (.30)

double ETA = .10, ALPHA = .30;

#define IMGSIZE (28*28)
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
	return 1 / (1 + exp(-in));
	//return tanh(in);
}
double transferPrime(double in){
	auto tmp = exp(-in);
	return tmp/((1+tmp)*(1+tmp));
	//return 1.0-in*in;
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

//FOR MNIST DATA SET
//
struct Parser{
	using BYTE = unsigned char;
	std::ifstream f_data,f_label;
	Parser(const char dat[], const char lab[]){
		f_data.open(dat);
		f_label.open(lab);
		//skip metadata
		BYTE buf[16];
		f_data.read((char*)buf,16);
		f_label.read((char*)buf,8);
		//skip metadata
	}
	~Parser(){
		f_data.close();
		f_label.close();
	}
	std::vector<int> parseTopology();
	bool parseInput(std::pair<std::vector<double>,std::vector<double>>& res);

};

std::vector<int> Parser::parseTopology(){
	return std::vector<int>({IMGSIZE,100,10});
}

bool Parser::parseInput(std::pair<std::vector<double>,std::vector<double>>& res){
	if(res.first.size() != IMGSIZE)
		res.first.resize(IMGSIZE);
	if(res.second.size() != 10)
		res.second.resize(10);
	//in = IMGSIZE
	//out = 10
	
	BYTE buf_dat[IMGSIZE];
	BYTE buf_lab;

	f_data.read((char*)buf_dat,IMGSIZE);
	f_label.read((char*)&buf_lab,1);

	for(int i=0;i<IMGSIZE;++i){
		res.first[i] = (buf_dat[i] / 256.0);
	}
	for(int i=0;i<10;++i){
		res.second[i]=0.0;
	}
	res.second[buf_lab] = 1.0; // probability 100%, others 0%
	return f_data && f_label;
	//return std::make_pair(res_dat,res_lab);
}
void visualize(std::vector<double>& input){
	for(int i=0;i<28;++i){
		for(int j=0; j<28; ++j){
			std::cout << (input[i*28+j]>0?1:0);
		}
		std::cout << std::endl;
	}
}
	/* ***** Class Declarations ***** */
struct Connection{
	double weight,delta;
	Connection(){
		static auto f = std::bind(std::uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(time(0)));
		weight = f();
		delta = 0;
	}
	Connection(double w){
		weight = w;
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
		Neuron(int index, std::string& s);
		std::vector<double> feedForward(double in);
		void print(std::ostream& f_out);
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
		Layer(int index, std::istream& f_in);
		std::vector<double> feedForward(std::vector<double>& input);
		void accumulate(std::vector<double>& dst, std::vector<double>& add);
		std::vector<double> calcGradient(std::vector<double>& next);
		std::vector<double> calcGradient(Layer& next);
		void update(Layer& prev);
		void print(std::ostream& f_out);
		void print();
};

class Net{
	private:
		std::vector<Layer> net;
	public:
		Net(); //EMPTY NETWORK
		Net(std::vector<int>& topology);
		Net(std::string file); //LOAD
		std::vector<double> feedForward(std::vector<double>& input);
		void backProp(std::vector<double>& target);
		void print(std::ostream& f_out);
		void print();
		void report(std::pair<std::vector<double>,std::vector<double>>& input);
		void report_2(std::vector<double>& target);
};
/* ***** Class Definitions ***** */

//Neuron
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
Net::Net(){
}
Net::Net(std::vector<int>& topology){
	for(uint i=0;i<topology.size();++i){
		net.push_back(Layer(i,topology[i], (i == topology.size()-1)?0:topology[i+1]));
	}
}
Net::Net(std::string file){
	std::ifstream f_in(file);
	std::string s;
	int index = 0;
	while(std::getline(f_in,s)){ // 1 line = node
		if(s == "<"){
			net.push_back(Layer(index, f_in));
			++index;
			continue;
		}
		if(s == ">")
			continue;
	}
	f_in.close();
}
void Net::print(std::ostream& f_out){
	for(auto& l : net){
		l.print(f_out);
	}
}
void Net::print(){
	print(std::cout);
}
void Net::report(std::pair<std::vector<double>,std::vector<double>>& input){
	static int count = 0;
	std::cout << '[' << ++count << ']';
	//std::cout << "INPUT : ";
	//for(auto &d : input.first){
	//	std::cout << d << ' ';
	//}
	//std::cout << " | ";
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

void Net::report_2(std::vector<double>& target){
	static int wrong=0;
	int targetVal;

	for(int i=0;i<10;++i){
		if(target[i] > 0.5){
			targetVal=i;
			break;
		}
	}
	auto& result = net.back().layer;
	auto guess = std::make_pair(0,0.0); //num, probability
	for(int i=0;i<10;++i){
		if(result[i].val > guess.second){
			guess.first = i;
			guess.second = result[i].val;
		}
	}
	if(guess.first != targetVal)
		++wrong;
	std::cout << targetVal << ':' <<  guess.first << '(' << guess.second << ')' << "WRONG : " << wrong << std::endl;
}


void train(Net& net){
	/* *** TRAINING PHASE *** */

	//std::ifstream f_in("training.txt");
	//auto topology = parseTopology(f_in);
	//auto inNum = topology.front();
	//auto outNum = topology.back();
	const char trainDat[] = "train/trainData";
	const char trainLab[] = "train/trainLabel";
	Parser p(trainDat,trainLab);
	auto topology = p.parseTopology();
	net = Net(topology);
	net.print();


	auto input = std::make_pair(std::vector<double>(), std::vector<double>());	
	while(1){
		bool success = p.parseInput(input);
		//visualize(input.first);
		//print(input.first);
		//auto input = parseInput(f_in, inNum,outNum,success);
		if(!success)
			break;
		net.feedForward(input.first);
		//net.report(input);
		//net.print();
		net.backProp(input.second);
		//break;
	}
	std::ofstream weight_map("weight_map.txt");
	net.print(weight_map);

	std::cout << "---- TRAINING COMPLETE!! ----" << std::endl;
}

void test(Net& net){

	/* *** TEST NET *** */
	auto input = std::make_pair(std::vector<double>(), std::vector<double>());	
	const char testDat[] = "train/testData";
	const char testLab[] = "train/testLabel";
	Parser p_2(testDat,testLab);	
	char next;
	while(p_2.parseInput(input)){
		visualize(input.first);
		net.feedForward(input.first);
		net.report_2(input.second);
		std::cin >> next;
		if(next == 'n' || next == 'N')
			break;
	}

}

int main(int argc, char* argv[]){
	
	/* *** SPECIFY CONSTANTS *** */
	if(argc == 3){
		ETA = std::atof(argv[1]);
		ALPHA = std::atof(argv[2]);
	}

#ifdef TRAIN
	Net net;
	train(net);
#else
	Net net("weight_map.txt");	
#endif
	test(net);
	return 0;
}
