#pragma once
#include <iostream>
#include "Detector.h"
//#include <opencv2\video\tracking.hpp>
//#include <opencv2\calib3d\calib3d.hpp>
#include <time.h>

int main() {
	Mat frame;
	clock_t start, end;

	VideoCapture cap("test.mp4");
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	float m_sgm_threshold = 2;
	float m_detection_threshold = 0.5;
	float m_variance_threshold = 2500;
	float m_decaying_parameter = 0.001;
	int m_key_point_number = 10;
	float m_qlevel = 0.01;
	float m_min_dist = 1;
	Detector detector(width, height, m_sgm_threshold, m_detection_threshold, m_variance_threshold, m_decaying_parameter, m_key_point_number, m_qlevel, m_min_dist);
	cap >> frame;
	//GaussianBlur(frame, frame, Size(3, 3), 0, 0);
	//medianBlur(frame, frame, 3);
	start = clock();
	detector.init(frame);
	end = clock();
	cout << end - start << endl;

	while(cap.read(frame)) {
		start = clock();
		//GaussianBlur(frame, frame, Size(3, 3), 0, 0);
		//medianBlur(frame, frame, 3);
		detector.compute(frame);
		end = clock();
		cout << end - start << endl;
	}
}

/*
#define DIVIDE_WIDTH 32
#define DIVIDE_HEIGHT 24
int main() {
	Mat frame, prev_frame;
	Mat prev_frames[DIVIDE_HEIGHT][DIVIDE_WIDTH];
	vector<Point2f> prev_points[DIVIDE_HEIGHT][DIVIDE_WIDTH], next_points[DIVIDE_HEIGHT][DIVIDE_WIDTH];
	int max_count = 100;
	double qlevel = 0.01;
	double min_dist = 1;
	vector<uchar> status;
	vector<float> err;
	clock_t start, end;

	VideoCapture cap("man.mp4");
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int grid_width = width / DIVIDE_WIDTH;
	int grid_height = height / DIVIDE_HEIGHT;

	cap >> frame;
	GaussianBlur(frame, frame, Size(3, 3), 0, 0);
	medianBlur(frame, frame, 3);
	imshow("test", frame);
	waitKey(1);
	for(int y=0; y<DIVIDE_HEIGHT; y++)
		for(int x=0; x<DIVIDE_WIDTH; x++) {
			Mat sub_frame = frame(Rect(x * grid_width, y * grid_height, grid_width, grid_height));
			cvtColor(sub_frame, sub_frame, CV_RGB2GRAY);
			prev_frames[y][x] = sub_frame;
			goodFeaturesToTrack(sub_frame, prev_points[y][x], max_count, qlevel, min_dist);
		}
	prev_frame = frame.clone();

	while(cap.read(frame)) {
		start = clock();
		cap >> frame;
		GaussianBlur(frame, frame, Size(3, 3), 0, 0);
		medianBlur(frame, frame, 3);
		Mat result = frame.clone();
		Mat result2(result.rows, result.cols, CV_8UC3);
		vector<Point2f> center;
		center.push_back(Point2f(grid_width/2, grid_height/2));

		for(int y=0; y<DIVIDE_HEIGHT; y++)
			for(int x=0; x<DIVIDE_WIDTH; x++) {
				Mat sub_frame = frame(Rect(x * grid_width, y * grid_height, grid_width, grid_height));
				cvtColor(sub_frame, sub_frame, CV_RGB2GRAY);
				if(prev_points[y][x].size() > 0) {
						calcOpticalFlowPyrLK(prev_frames[y][x], sub_frame, prev_points[y][x], next_points[y][x], status, err);
					if(next_points[y][x].size() >= 4) {
						Mat H = findHomography(next_points[y][x],  prev_points[y][x], CV_RANSAC);	
						if(countNonZero(H) >= 1) {
							vector<Point2f> center2;
							perspectiveTransform(center, center2, H);
							Point2f yxCenter(x*grid_width + grid_width/2, y*grid_height + grid_height/2);
							Point2f yxCenter2(center2.at(0).x + x * grid_width, center2.at(0).y + y * grid_height);
							if(norm(yxCenter - yxCenter2) <= 10)
								line(result, yxCenter, yxCenter2, Scalar(255, 0, 0), 1, 8);
						}
						else {
							//cout << H << endl;
						}
					}
				}

				prev_frames[y][x] = sub_frame;
				prev_points[y][x] = next_points[y][x];
			}
		
		prev_frame = frame.clone();
		imshow("test", result);
		imshow("test2", result2);
		waitKey(1);
		end = clock();
		cout << end - start << endl;
	}
}
*/