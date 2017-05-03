#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <iostream>
#include <time.h>
using namespace cv;
#define BLOCK_SIZE 3
#define SKIP_SIZE_X 0
#define SKIP_SIZE_Y 0

void calculate(uchar* data, uchar* idx, int width, int height, int resizeWidth) {
	int div = BLOCK_SIZE/2;

	for(int y=div, idx_y=0; y<height-div; y+=SKIP_SIZE_Y+1, idx_y++)
		for(int x=div, idx_x=0; x<width-div; x+=SKIP_SIZE_X+1, idx_x++) {
			char pixel = data[y*width + x];
			//float sum = 0;
			for(int i=-div, count = 0; i<=div; i++)
				for(int j=-div; j<=div; j++, count++) {
					idx[idx_y*resizeWidth + idx_x] += (data[(y+i)*width + (x+j)] > pixel) ? 1 << count : 0;
					//idx[idx_y*resizeWidth + idx_x] += abs((float)data[(y+i)*width + (x+j)] - pixel) / (BLOCK_SIZE*BLOCK_SIZE);
					//sum += abs((float)data[(y+i)*width + (x+j)] - pixel) / (BLOCK_SIZE*BLOCK_SIZE-1);
					//sum += (float)data[(y+i)*width + (x+j)] / (BLOCK_SIZE*BLOCK_SIZE);
				}
			//idx[idx_y*resizeWidth + idx_x] = (int)sum;
		}
}

void main() {
	VideoCapture cap("test1111.mp4");
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int resizeWidth = width / (SKIP_SIZE_X+1);
	int resizeHeight = height / (SKIP_SIZE_Y+1);
	Mat frame, result(resizeHeight, resizeWidth, CV_8UC1), prev(resizeHeight, resizeWidth, CV_8UC1), diff;
	clock_t start, end;
	
	uchar* idx = result.data;
	cap >> frame;
	GaussianBlur(frame, frame, Size(3, 3), 0, 0);
	medianBlur(frame, frame, 3);
	cvtColor(frame, frame, CV_RGB2GRAY);
	calculate(frame.data, idx, width, height, resizeWidth);
	result.copyTo(prev);

	while(true) {
		start = clock();
		memset(idx, 0, resizeWidth*resizeHeight);

		cap >> frame;
		GaussianBlur(frame, frame, Size(3, 3), 0, 0);
		medianBlur(frame, frame, 3);
		cvtColor(frame, frame, CV_RGB2GRAY);

		calculate(frame.data, idx, width, height, resizeWidth);

		end = clock();
		std::cout << end - start << std::endl;
		subtract(prev, result, diff);
		medianBlur(diff, diff, 5);
		imshow("test", result);
		imshow("original", frame);
		imshow("diff", diff);
		waitKey(100);

		result.copyTo(prev);
	}

	delete[] idx;
}