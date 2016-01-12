#include "Net.h"

#define TRAIN
int main(int argc, char* argv[]){
	/* *** SPECIFY CONSTANTS *** */
	Net net;

#ifdef TRAIN
	if(argc == 3){
		//LEARNING RATE
		ETA = std::atof(argv[1]);
		//MOMENTUM
		ALPHA = std::atof(argv[2]);
	}
	train(net);
	net.save("net.txt");
#else //LOAD
	net.load("net.txt");
#endif


	test(net);
	return 0;
}
