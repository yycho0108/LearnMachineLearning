#include "Utility.h"

/* ***** Utility Functions ***** */
double ETA = 0.15;
double ALPHA = 0.3;

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
//
Parser::Parser(const char dat[], const char lab[]){
	f_data.open(dat);
	f_label.open(lab);
	//skip metadata
	BYTE buf[16];
	f_data.read((char*)buf,16);
	f_label.read((char*)buf,8);
	//skip metadata
}

Parser::~Parser(){
	f_data.close();
	f_label.close();
}
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


