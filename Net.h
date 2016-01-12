#ifndef __NET_H__
#define __NET_H__

#include "Neuron.h"
#include "Layer.h"

class Net{
	private:
		std::vector<Layer> net;
	public:
		Net(); //EMPTY NETWORK
		Net(std::vector<int>& topology);
		Net(std::string file); //LOAD
		std::vector<double> feedForward(std::vector<double>& input);
		void backProp(std::vector<double>& target);
		// REPRESENTATION
		void print(std::ostream& f_out);
		void print();
		void report(std::pair<std::vector<double>,std::vector<double>>& input);
		void report_2(std::vector<double>& target);
		std::vector<double> getResults();
		// I/O
		void save(std::string fileName);
		void load(std::string fileName);
};

void train(Net& net);
void test(Net& net);
#endif
