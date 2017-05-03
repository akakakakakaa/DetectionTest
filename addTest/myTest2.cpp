#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <iostream>
#include <time.h>
#include <stdint.h>
using namespace cv;
#define BLOCK_SIZE 3
#define SKIP_SIZE_X 0
#define SKIP_SIZE_Y 0
#define DEBUG 1

//gray color
struct Node {
	int avg_value;
	int mask[BLOCK_SIZE*BLOCK_SIZE];
	int refresh;

	Node() {
		avg_value = 0;
		refresh = 10;
	} 
};

struct NodeMat {
	int width;
	int height;
	Node* nodes;

	NodeMat(int m_height, int m_width) {
		width = m_width;
		height = m_height;
		nodes = new Node[height*width];
	}

	void release() {
		delete[] nodes;
	}
};

void createMask(Mat frame, NodeMat& nodeFrame, int threshold) {
#ifdef DEBUG 1
	clock_t start = clock();
#endif
	uchar* data = frame.data;
	Node* nodes = nodeFrame.nodes;
	int k;
	for(int y=1; y<frame.rows-1; y++) {
		int first = (y-1)*frame.cols;
		int second = y*frame.cols;
		int third = (y+1)*frame.cols;
		for(int x=1; x<frame.cols-1; x++) {
			if(nodes[second + x].refresh >= 10) {
				uchar curr = data[second + x];
				//Node node; 해서 지역변수로 하는것보다 이게더빠르네;
				Node* node = &nodes[second + x];
				node->avg_value = 0;
				k = abs(curr - data[first   + x - 1]);
				k < threshold ? node->mask[0] = k, node->avg_value+=data[first   + x - 1] :  node->mask[0] = 0;
				k = abs(curr - data[first   + x    ]);
				k < threshold ? node->mask[1] = k, node->avg_value+=data[first   + x    ] :  node->mask[1] = 0;
				k = abs(curr - data[first   + x + 1]);
				k < threshold ? node->mask[2] = k, node->avg_value+=data[first   + x + 1] :  node->mask[2] = 0;
				k = abs(curr - data[second  + x - 1]);
				k < threshold ? node->mask[3] = k, node->avg_value+=data[second  + x - 1] :  node->mask[3] = 0;
				k = abs(curr - data[second  + x    ]);
				k < threshold ? node->mask[4] = k, node->avg_value+=data[second  + x    ] :  node->mask[4] = 0;
				k = abs(curr - data[second  + x + 1]);
				k < threshold ? node->mask[5] = k, node->avg_value+=data[second  + x + 1] :  node->mask[5] = 0;
				k = abs(curr - data[third   + x - 1]);
				k < threshold ? node->mask[6] = k, node->avg_value+=data[third   + x - 1] :  node->mask[6] = 0;
				k = abs(curr - data[third   + x    ]);
				k < threshold ? node->mask[7] = k, node->avg_value+=data[third   + x    ] :  node->mask[7] = 0;
				k = abs(curr - data[third   + x + 1]);
				k < threshold ? node->mask[8] = k, node->avg_value+=data[third   + x + 1] :  node->mask[8] = 0;
				node->refresh = 0;
				node->avg_value /= 9;
			}
		}
	}

#ifdef DEBUG 1
	clock_t end = clock();
	std::cout << "createMask " << end - start << "ms" << std::endl;
#endif
};

void check(Mat frame, NodeMat& nodeFrame, float threshold) {
#ifdef DEBUG 1
	clock_t start = clock();
#endif
	uchar* data = frame.data;
	Node* nodes = nodeFrame.nodes;
	int k;

	for(int y=1; y<frame.rows-1; y++) {
		int first = (y-1)*frame.cols;
		int second = y*frame.cols;
		int third = (y+1)*frame.cols;
		for(int x=1; x<frame.cols-1; x++) {
			if(nodes[second + x].avg_value != 0) {
				uchar curr = data[second + x];
				//Node node; 해서 지역변수로 하는것보다 이게더빠르네;
				Node* node = &nodes[second + x];
			
				float diff = 0;
				int avg_value = 0;
				node->mask[0] != 0 ? diff += abs(abs(curr - data[first  + x - 1]) / (float)node->mask[0] - 1), avg_value += data[first  + x - 1] : NULL;
				node->mask[1] != 0 ? diff += abs(abs(curr - data[first  + x    ]) / (float)node->mask[1] - 1), avg_value += data[first  + x    ] : NULL;
				node->mask[2] != 0 ? diff += abs(abs(curr - data[first  + x + 1]) / (float)node->mask[2] - 1), avg_value += data[first  + x + 1] : NULL;
				node->mask[3] != 0 ? diff += abs(abs(curr - data[second + x - 1]) / (float)node->mask[3] - 1), avg_value += data[second + x - 1] : NULL;
				node->mask[4] != 0 ? diff += abs(abs(curr - data[second + x    ]) / (float)node->mask[4] - 1), avg_value += data[second + x    ] : NULL;
				node->mask[5] != 0 ? diff += abs(abs(curr - data[second + x + 1]) / (float)node->mask[5] - 1), avg_value += data[second + x + 1] : NULL;
				node->mask[6] != 0 ? diff += abs(abs(curr - data[third  + x - 1]) / (float)node->mask[6] - 1), avg_value += data[third  + x - 1] : NULL;
				node->mask[7] != 0 ? diff += abs(abs(curr - data[third  + x    ]) / (float)node->mask[7] - 1), avg_value += data[third  + x    ] : NULL;
				node->mask[8] != 0 ? diff += abs(abs(curr - data[third  + x + 1]) / (float)node->mask[8] - 1), avg_value += data[third  + x + 1] : NULL;
				if(diff >= threshold || abs(avg_value - node->avg_value) >= 90) {
					data[second + x] = 0;
					nodes[second + x].refresh++;
				}
				else {
					data[second + x] = 255;
				}
			}
			else {
				data[second + x] = 255;
				nodes[second + x].refresh++;
			}
		}
	}

#ifdef DEBUG 1
	clock_t end = clock();
	std::cout << "check " << end - start << "ms" << std::endl;
#endif
};

int main() {
	VideoCapture cap("test1111.mp4");
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int resizeWidth = width / (SKIP_SIZE_X+1);
	int resizeHeight = height / (SKIP_SIZE_Y+1);
	int fps = cap.get(CV_CAP_PROP_FPS);

	Mat prev_frame, next_frame, result;
	NodeMat nodeFrame(height, width);

	cap >> prev_frame;
	cvtColor(prev_frame, prev_frame, CV_RGB2GRAY);
	createMask(prev_frame, nodeFrame, 500);

	while(cap.read(next_frame)) {
		cvtColor(next_frame, next_frame, CV_RGB2GRAY);

		check(next_frame, nodeFrame, 8);
		createMask(next_frame, nodeFrame, 500);
		subtract(prev_frame, next_frame, result);
		//cv::morphologyEx(result, result, cv::MORPH_OPEN, Mat(5, 5, CV_8U, cv::Scalar(1)));
		imshow("result", result);
		imshow("frame", next_frame);
		waitKey(1000/fps);

		next_frame.copyTo(prev_frame);
	}

	nodeFrame.release();

	return 0;
}