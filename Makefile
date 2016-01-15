CC = g++
FLAGS = -std=c++11 -O3
LIBS=-L/usr/lib
INCS=-I/usr/include -I.
TARGET = main
all: $(TARGET)

$(TARGET): Neuron.o Layer.o Net.o Utility.o $(TARGET).cpp
	$(CC) $(FLAGS) $(INCS) $^ -o $@ $(LIBS)
Neuron.o: Neuron.cpp
	$(CC) -c $(FLAGS) $(INCS) $^ -o $@ $(LIBS)
Layer.o: Neuron.o Utility.o Layer.cpp
	$(CC) -c $(FLAGS) $(INCS) $^ -o $@ $(LIBS)
Net.o: Neuron.o Layer.o Utility.o Net.cpp
	$(CC) -c $(FLAGS) $(INCS) $^ -o $@ $(LIBS)
Utility.o : Utility.cpp
	$(CC) -c $(FLAGS) $(INCS) $^ -o $@ $(LIBS)

clean:
	rm main Neuron.o Layer.o Net.o Utility.o
