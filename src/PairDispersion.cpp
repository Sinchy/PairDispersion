//============================================================================
// Name        : PairDispersion.cpp
// Author      : Shiyong Tan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <chrono>
#include <omp.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "NumMATIO.h"
#include "Array.h"
using namespace std;

double VectorNorm(double x, double y, double z) {
	return sqrt(x * x + y * y + z * z);
}

void SavePairs(char* file_path, vector<vector<vector<int>>>& pairs){
	// remove the file name in file_path

	string file_path_s(file_path);
	int pos = file_path_s.find_last_of("/\\");
	string save_path0 = file_path_s.substr(0, pos);
//	cout<<save_path0<<endl;

	NumMATIO mat_io;
	mat_io.SetVarName("pairs");
	int num_gp = pairs.size();
	for (int i = 0; i < num_gp; ++i) {
		int num_rows = pairs[i].size();
		string save_path = save_path0 + "/\\pairs" + to_string(i) + ".mat";
		mat_io.SetFilePath((char*)save_path.c_str());
		mat_io.WriteData(pairs[i]);
	}
}


int main(int argc, char **argv) {
	cout << "!!!Hello World!!!!" << endl;

	auto start = std::chrono::system_clock::now();

	// Read the track data from the .mat file
	NumMATIO mat_io;
	mat_io.SetFilePath(argv[1]);
	mat_io.SetVarName("filter_data");
	vector<int> dim;
	double* data = NULL;
	mat_io.ReadData(data, dim);
	Array track(data, dim);

	//Read the bin information
	mat_io.SetVarName("bin");
	vector<int> bin_dim;
	double* bin_data = NULL;
	mat_io.ReadData(bin_data, bin_dim);
	Array bin(bin_data, bin_dim);

	//Obtain unique frame number
	vector<double> frameNo = track.GetColumn(3);

    std::sort(frameNo.begin(), frameNo.end());
    auto last = std::unique(frameNo.begin(), frameNo.end());
    frameNo.erase(last, frameNo.end());
	random_shuffle(frameNo.begin(), frameNo.end());

	//Obtain the end frame of each trajectory which is used to calculat the pair length
	// NOTE: the track must be sorted according to the track ID.
	vector<double> trID = track.GetColumn(4);
	last = std::unique(trID.begin(), trID.end());
	trID.erase(last, trID.end());
	vector<double> tr_endframe(trID.size());
	int ID = 0;
	for (int i = 0; i < dim[0]; ++i) {
		if (track(i, 4) != trID[ID]){
			tr_endframe[ID] = track(i - 1, 3);
			++ID;
		}
	}


	// Setting for pair searching
	int max_num_pairs = 50000;
	int min_pair_length = 100;
	bool* pairs_full = new bool[bin_dim[0]];
	for(int i = 0; i < bin_dim[0]; ++i) pairs_full[i] = false;

	//Obtain pairs within a specific distance
	vector<vector<vector<int>>> pairs(bin_dim[0]); // first dimension: groups of bins;
	//second dimension: number of pairs; third dimension: pair ID and frame no and length

#pragma omp parallel shared(pairs) //num_threads(10)
	{
#pragma omp for
//		for (int i = 0; i < 1000; ++i) {
	for (int i = 0; i < frameNo.size(); ++i) {
#pragma omp critical
		{
		cout<<i<<endl;
		}
		vector<int> index = track.GetElementRowIndex(frameNo[i], 3); // row index of the particles in one frame
		int num_point = index.size();
		if (num_point <= 1)
			continue; // skip when there is only one particle in that frame

		vector<double> X = track.GetElement_RowsCol(index, 0);
		vector<double> Y = track.GetElement_RowsCol(index, 1);
		vector<double> Z = track.GetElement_RowsCol(index, 2);

		// calculate the inter-partile distance
//		int total_combination = num_point * (num_point - 1) / 2;
//		vector<vector<int>> index_combination(total_combination, vector<int>(2));
//		vector<double> dist(total_combination);
		double dist = 0;
		vector<int> pair_candidate(4, 0);
		for (int j = 0; j < num_point; ++j) {
			for (int k = j + 1; k < num_point; ++k) {
//				dist(j * (num_point - 1 - j) + k) = VectorNorm(X[j] - X[k],
//						Y[j] - Y[k], Z[j] - Z[k]);
//				index_combination[j * (num_point - 1 - j) + k][0] = j;
//				index_combination[j * (num_point - 1 - j) + k][1] = k;

				dist = VectorNorm(X[j] - X[k], Y[j] - Y[k], Z[j] - Z[k]);
				if (dist > bin(bin_dim[0] - 1, 1)) // if dist is larger than the largest, then continue
					continue;

				for (int n = 0; n < bin_dim[0]; ++n) {
					if (pairs_full[n] || dist < bin(n, 0)) // NOTE: bin should start from small to large
															// if enough pairs have been found, then skip
						break;
					if (dist > bin(n, 1))
						continue;
					pair_candidate[0] = track(index[j], 4); //get the track ID
					pair_candidate[1] = track(index[k], 4);
					pair_candidate[2] = frameNo[i];  // frame no

					// Get the track length
					// get the time sequence for each particle
//					vector<double> frame_sequence0 = track.GetElement_RowsCol(track.GetElementRowIndex(pair_candidate[0], 4), 3);
//					vector<double> frame_sequence1 = track.GetElement_RowsCol(track.GetElementRowIndex(pair_candidate[1], 4), 3);
//					int end_frame = min(frame_sequence0.back(), frame_sequence1.back());
					int trID_index = distance(trID.begin(), find(trID.begin(), trID.end(), pair_candidate[0]));
					double end_frame0 = tr_endframe[trID_index];
					trID_index = distance(trID.begin(), find(trID.begin(), trID.end(), pair_candidate[1]));
					double end_frame1 = tr_endframe[trID_index];
					double end_frame = min(end_frame0, end_frame1);
					pair_candidate[3] = end_frame - frameNo[i] + 1;   // length of the pair.

#pragma omp critical(pairs)
					{
						if (pair_candidate[3] > min_pair_length) { // short pairs would be neglected.
							pairs[n].push_back(pair_candidate);
							if (pairs[n].size() > max_num_pairs)
								pairs_full[n] = true;
						}
						}

//					cout<<n<<" "<<pairs[n][0][0]<<" "<<pairs[n][0][1]<<" "<<pairs[n][0][2]<<" "<<pairs[n][0][3]<<" "<<endl;
				}
			}
		}
	}
	}

	// save pairs to .mat file
	SavePairs(argv[1], pairs);

	delete[] pairs_full;

	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	cout << "Total time: "<<elapsed.count() << '\n';
	return 0;
}
