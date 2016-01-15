#include "Net.h"

#define TRAIN
int main(int argc, char* argv[]){
	/* *** SPECIFY CONSTANTS *** */
	std::vector<int> topology({28*28, 75, 10});
	Net net(topology);

#ifdef TRAIN
	if(argc == 3){
		//LEARNING RATE
		ETA = std::atof(argv[1]);
		//MOMENTUM
		ALPHA = std::atof(argv[2]);
	}
	for(int i=0;i<1;++i){
		train(net);
	}	

	net.save("net.txt");
#else //LOAD
	net.load("net.txt");
#endif


	test(net);
	return 0;
}
