#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <iostream>
#include <time.h>
#include <stdint.h>
using namespace cv;
#define BLOCK_SIZE 5
#define BLOCK_SIZE_MINUS_ONE BLOCK_SIZE-1
#define DEBUG 1
#define DELAY_SIZE 0
/* 0~1.0 */
#define MIN_THRESHOLD 0.9
#define DEFAULT_THRESHOLD 8*BLOCK_SIZE*BLOCK_SIZE;
#define MAX_THRESHOLD 0.95

void createBlocks(uchar* data, int width, int height, int* horizontal, int* block) {
	clock_t start, end;
	start = clock();

	int div = BLOCK_SIZE/2;
	for(int y=0; y<height + BLOCK_SIZE_MINUS_ONE; y++) {
		int rows = y*width;
		int blockRows = y*(width + BLOCK_SIZE_MINUS_ONE);

		int x = div;
		horizontal[rows] = 0;
		for(int i=-div; i<=div; i++)
			horizontal[rows] += data[blockRows + x + i];
		for(int idx=1; idx < width; idx++, x++) {
			horizontal[rows + idx] = horizontal[rows + idx - 1];
			horizontal[rows + idx] -= data[blockRows + x - div];
			horizontal[rows + idx] += data[blockRows + x + div + 1];
		}
	}

	for(int x=0; x<width; x++) {
		block[x] = 0;

		for(int i=0; i<BLOCK_SIZE; i++)
			block[x] += horizontal[i*width + x];
		for(int y=1; y<height; y++) {
			block[y*width + x] = block[(y-1)*width + x];
			block[y*width + x] -= horizontal[(y-1)*width + x];
			block[y*width + x] += horizontal[(y+BLOCK_SIZE_MINUS_ONE)*width + x];
			//if(block[y*width + x] < 0 || block[(y-1)*width + x] < horizontal[(y-1)*width + x])
			//	std::cout << "a";
		}
	}
	end = clock();
	std::cout << "createBlocks : " << end - start << "ms" << std::endl;
}

void findMatching(uchar* data, float* threshold, int* prev_block, int* next_block, int width, int height) {
	clock_t start, end;
	start = clock();
	for(int y=0; y<height; y++) {
		int rows = y*width;
		for(int x=0; x<width; x++) {
			int diff = abs(next_block[rows + x] - prev_block[rows + x]);
			if(diff >= threshold[rows + x])
				data[rows + x] = 255;
			else
				data[rows + x] = 0;
			/*
			float divide = next_block[rows + x] / (float)prev_block[rows + x];
			if(divide <= threshold[rows + x] || divide >= 1/threshold[rows + x]) {
				data[rows + x] = 255;
 				//threshold[rows + x] += (exp(threshold[rows + x]/100) - 1);
				//if(threshold[rows + x] > MAX_THRESHOLD)
				//	threshold[rows + x] = MAX_THRESHOLD;
			}
			else {
				data[rows + x] = 0;
				//threshold[rows + x] -= (exp(threshold[rows + x]/100) - 1);
				//if(threshold[rows + x] < MIN_THRESHOLD)
				//	threshold[rows + x] = MIN_THRESHOLD;
			}
			*/
		}
	}
	end = clock();

	std::cout << "findMatching : " << end - start << "ms" << std::endl;
}

int main() {
	VideoCapture cap("expressway.mp4");
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int fps = cap.get(CV_CAP_PROP_FPS);
	if(fps == 0)
		fps = 5;

	Mat frame, expandFrame(height + BLOCK_SIZE - 1, width + BLOCK_SIZE - 1, CV_8UC1), result(height, width, CV_8UC1);
	expandFrame = Scalar(128);
	int div = BLOCK_SIZE/2;
	vector<int*> blocks;
		
	int* hori = new int[width*(height+BLOCK_SIZE_MINUS_ONE)];

	float* threshold = new float[width*height];
	for(int i=0; i<width*height; i++) {
		threshold[i] = DEFAULT_THRESHOLD;
	}

	while(cap.read(frame)) {
		/*
		clock_t start = clock();
		pyrMeanShiftFiltering(frame, frame, 10, 10);
		clock_t end = clock();
		std::cout << "meanshift filtering: " << end - start << "ms" << std::endl;
		*/
		cvtColor(frame, frame, CV_RGB2GRAY);
		frame.copyTo(expandFrame.colRange(div, width + div).rowRange(div, height + div));

		int* block = new int[width*height];
		createBlocks(expandFrame.data, width, height, hori, block);
		blocks.push_back(block);

		if(blocks.size() >= DELAY_SIZE + 2) {
			findMatching(result.data, threshold, blocks.at(0), block, width, height);
			delete[] blocks.at(0);
			blocks.erase(blocks.begin()+0);

			imshow("frame", frame);
			imshow("expandFrame", expandFrame);
			//morphologyEx(result, result, MORPH_OPEN, Mat(5, 5, CV_8U, cv::Scalar(1)));
			//medianBlur(result, result, 11);
			imshow("result", result);
			waitKey(1000/fps);
		}
	}

	delete[] hori;
	for(int i=0; i<blocks.size(); i++)
		delete[] blocks.at(0);
	delete[] threshold;
	/*
	int *prev_hori, *next_hori, *prev_block, *next_block;
#ifdef DEBUG 1
	clock_t start, end;
#endif
	if(cap.read(frame)) {
		cvtColor(frame, frame, CV_RGB2GRAY);
		frame.copyTo(expandFrame.colRange(div, width + div).rowRange(div, height + div));

		prev_hori = new int[width*(height+BLOCK_SIZE_MINUS_ONE)];
		prev_block = new int[width*height];
		createBlocks(expandFrame.data, width, height, prev_hori, prev_block);
		while(cap.read(frame)) {
#ifdef DEBUG 1
			start = clock();
#endif

			next_hori = new int[width*(height+BLOCK_SIZE_MINUS_ONE)];
			next_block = new int[width*height];

			cvtColor(frame, frame, CV_RGB2GRAY);
			frame.copyTo(expandFrame.colRange(div, width + div).rowRange(div, height + div));

			createBlocks(expandFrame.data, width, height, next_hori, next_block);
			findMatching(result.data, prev_block, next_block, width, height);

			delete[] prev_hori;
			delete[] prev_block;
			prev_hori = next_hori;
			prev_block = next_block;

#ifdef DEBUG 1
			end = clock();
			std::cout << end - start << "ms" << std::endl;
#endif
			imshow("frame", frame);
			imshow("expandFrame", expandFrame);
			imshow("result", result);
			waitKey(1000/fps);
		}
		delete[] prev_hori;
		delete[] prev_block;
	}
*/

	return 0;
}