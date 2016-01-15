#ifndef __NEURON_H__
#define __NEURON_H__

#include <vector>
#include <random>
#include <functional>
#include <ctime>
#include <sstream>
#include <fstream>
#include <iostream>

class Neuron{
	struct Connection{
		double weight,delta;
		Connection();
		Connection(double w);
	};
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

#endif
