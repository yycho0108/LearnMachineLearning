#include <iostream>
#include <fstream>
#include <random>
#include <functional>
#include <ctime>

/*bool alternate = false;
int src[] = {0,1,1,1,1,0,0,0};
int index =-1;
*/
double createData(){
	static auto f = std::bind(std::uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(time(0)));
	return f();
	//return f()>0.5?1:0;
	//if(++index >=8) index=0;
	//return src[index];
}
int main(int argc, char* argv[]){
	int lim = 200;
	if(argc >1)
		lim = std::atoi(argv[1]);
	std::ofstream f_out("training.txt");
	f_out << "2 4 2" << std::endl;
	for(int i=0;i<lim;++i){
		auto left = createData();
		auto right = createData();
		//auto res = (left*right);
		f_out << left << ' ' << right << ' ' << right << ' ' << left << std::endl;
		//f_out << double(left) << ' ' << double(right) << ' '<< double(res) << std::endl;
	}
}
