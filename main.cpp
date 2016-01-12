#include "Net.h"

#define TRAIN
int main(int argc, char* argv[]){
	/* *** SPECIFY CONSTANTS *** */
	if(argc == 3){
		ETA = std::atof(argv[1]);
		ALPHA = std::atof(argv[2]);
	}

#ifdef TRAIN
	Net net;
	train(net);
#else //LOAD
	Net net("weight_map.txt");	
#endif
	test(net);
	return 0;
}
