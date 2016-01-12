#include "Net.h"
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
	load(file);
}
void Net::print(std::ostream& f_out){
	for(auto& l : net){
		l.print(f_out);
	}
}
void Net::print(){
	print(std::cout);
}

std::vector<double> Net::getResults(){
	std::vector<double> res;
	for(auto& n : net.back().layer){
		res.push_back(n.val);
	}
	return res;
}
void Net::report(std::pair<std::vector<double>,std::vector<double>>& input){
	static int count = 0;
	std::cout << '[' << ++count << ']';
	std::cout << "OUTPUT : ";
	for(auto &o : input.second){
		std::cout << o << ' ';
	}
	std::cout << " | ";
	std::cout << "RESULT : ";

	for(auto& n : getResults()){
		std::cout << n << ' ';
	}
	std::cout << std::endl;
}

bool wrongFlag = false;
void Net::report_2(std::vector<double>& target){
	static int wrong=0;
	int targetVal;

	for(int i=0;i<10;++i){
		if(target[i] > 0.5){
			targetVal=i;
			break;
		}
	}
	auto res = getResults();
	auto guess = std::make_pair(0,0.0); //num, probability
	for(int i=0;i<10;++i){
		if(res[i] > guess.second){
			guess.first = i;
			guess.second = res[i];
		}
	}
	if(guess.first != targetVal){
		++wrong;
		wrongFlag = true;
	}
	std::cout << targetVal << ':' <<  guess.first << '(' << guess.second << ')' << "WRONG : " << wrong << std::endl;
}


void Net::save(std::string fileName){
	std::ofstream f_out(fileName);
	print(f_out);
	f_out.close();
}

void Net::load(std::string fileName){
	net.clear();
	std::ifstream f_in(fileName);
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
/* *** OTHER FUNCTIONS *** */

void train(Net& net){
	/* *** TRAINING PHASE *** */

	std::cout << "---- TRAINING START!! ----" << std::endl;
	//std::ifstream f_in("training.txt");
	//auto topology = parseTopology(f_in);
	//auto inNum = topology.front();
	//auto outNum = topology.back();
	const char trainDat[] = "train/trainData";
	const char trainLab[] = "train/trainLabel";
	Parser p(trainDat,trainLab);
	auto topology = p.parseTopology();
	net = Net(topology);
	//net.print();
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


	std::cout << "---- TRAINING COMPLETE!! ----" << std::endl;
}

bool continueFlag = false; //do not continue
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
		if(wrongFlag && !continueFlag){
			wrongFlag = false;
			std::cin >> next;
			if(next == 'n' || next == 'N')
				break;
			if(next == 'c')
				continueFlag = true;
		}
	}

}
