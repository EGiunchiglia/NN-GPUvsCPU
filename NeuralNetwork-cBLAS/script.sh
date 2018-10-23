#!/bin/sh

#  script.sh
#  sequential_neural_net_BLAS
#
#  Created by E.Giunchiglia on 29/01/18.
#  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.


for i in $(seq 25 20 100) ; do
g++ -O2  -lopenblas  -std=c++11 main.cpp ./src/neuralNetworkModel.cpp ./src/neuralNetworkTrainer.cpp ./src/dataReader.cpp || exit;
./a.out $i
done

for i in $(seq 120 50 550) ; do
g++ -O2 -lopenblas  -std=c++11 main.cpp ./src/neuralNetworkModel.cpp ./src/neuralNetworkTrainer.cpp ./src/dataReader.cpp || exit;
./a.out $i
done

for i in $(seq 720 200 1920) ; do
g++ -O2 -lopenblas  -std=c++11 main.cpp ./src/neuralNetworkModel.cpp ./src/neuralNetworkTrainer.cpp ./src/dataReader.cpp || exit;
./a.out $i
done
