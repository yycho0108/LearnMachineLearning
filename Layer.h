#ifndef __LAYER_H__
#define __LAYER_H__

#include "Utility.h"
#include "Neuron.h"
#include <cassert>

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

#endif
