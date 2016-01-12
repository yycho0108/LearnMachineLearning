#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>


#define IMGSIZE (28*28)
/* ***** Utility Functions ***** */
typedef std::string str;
typedef unsigned uint;
extern double ETA, ALPHA;

template<typename T>
void print(std::vector<T> v);


double transfer(double in);
double transferPrime(double in);
std::vector<double> transfer(std::vector<double>& in);

std::vector<double> transferPrime(std::vector<double>& in);
std::vector<int> parseTopology(std::ifstream& f_in);


std::pair<std::vector<double>,std::vector<double>> parseInput(std::ifstream& f_in, int in, int out, bool& success);


//FOR MNIST DATA SET
//
struct Parser{
	using BYTE = unsigned char;
	std::ifstream f_data,f_label;
	Parser(const char dat[], const char lab[]);
	~Parser();
	std::vector<int> parseTopology();
	bool parseInput(std::pair<std::vector<double>,std::vector<double>>& res);

};

void visualize(std::vector<double>& input);
#endif
