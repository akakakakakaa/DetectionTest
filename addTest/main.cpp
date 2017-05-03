#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <iostream>
#include <time.h>
using namespace cv;

#define NUM 2
#define MAX_DIST 100

struct Identifier {
	Point2f direc_v;
	Point2f point;

	Identifier(Point2f x_, Point2f y_) :direc_v(x_), point(y_) {}
};

void getKeyPoint(Mat image_next, vector<Point2f>& features_next, int max_count, double qlevel, double min_dist) {
#if DEBUG 1
	DWORD startTime = GetTickCount();
#endif
	goodFeaturesToTrack(image_next, features_next, max_count, qlevel, min_dist);
#if DEBUG 1
	cout << "key point extraction interval : " << GetTickCount() - startTime << "ms" << endl;
#endif
}

vector<Rect> DetectMovingObjectRectangle(Mat frame) {
	Mat hierarchy;
	int minX, minY, maxX, maxY;
	vector<vector<Point>> contours;
	morphologyEx(frame, frame, MORPH_OPEN, Mat(7, 7, CV_8U, cv::Scalar(1)));
	imshow("morphology", frame);
	findContours(frame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	bool check = false;
	vector<Rect> rectList;
	for(int i=0; i<contours.size(); i++) {
		Rect rect = boundingRect(contours.at(i));
		for(int j=0; j<rectList.size(); j++) {
			Rect obj = rectList.at(j);
            Point center((int) (rect.tl().x + rect.br().x / 2), (int) (rect.tl().y + rect.br().y / 2));
            Point center2((int) (obj.tl().x + obj.br().x / 2), (int) (obj.tl().y + obj.br().y / 2));
			double distance = sqrt(pow((double)center.x - center2.x, 2) + pow((double)center.y - center2.y, 2));
            if (distance <= 200) {
                minX = min(obj.tl().x, rect.tl().x);
                minY = min(obj.tl().y, rect.tl().y);
                maxX = max(obj.br().x, rect.br().x);
                maxY = max(obj.br().y, rect.br().y);
                Rect obj2 (Point(minX, minY), Point(maxX, maxY));
                rectList.erase(rectList.begin() + j);
                rectList.push_back(obj2);
                check = true;
                break;
            }
		}
		if(check == false)
			rectList.push_back(rect);
		check = false;
	}

	return rectList;
}

void BackgroundSubtractionWithOpticalFlowTest() {
	VideoCapture cap("test3.mp4");	
	cv::Mat frame_prev, frame_next, frame_save, result, fgMaskMOG;
	vector<Point2f> features_prev, features_next;
	vector<vector<float>> lineLengths;
	vector<uchar> status;
    vector<float> err;
	int max_count = 1000;
	double qlevel = 0.01;
	double min_dist = 10;
	int fps = 1000;
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int stride = 10;
	bool isCalc = false;

	clock_t start, end;
	while(true) {
        cap >> frame_next;
		cvtColor(frame_next, frame_next, CV_RGB2GRAY);
		getKeyPoint(frame_next, features_next, max_count, qlevel, min_dist);

		int keyPointSize = features_next.size();
		double reKeyPointRatio = 0.666;
		while(true) {
			start = clock();
			//전 Gray 이미지를 저장 
			frame_prev = frame_next.clone();
			//RGB값으로 가지고 옴
			cap >> frame_next;
			cv::GaussianBlur(frame_next, frame_save, cv::Size(0, 0), 3);
			cv::addWeighted(frame_next, 1.5, frame_save, -0.5, 0, frame_next);
			//화면을 보여주기 위한 RGB 이미지를 Gray로 변환하기 전에 저장
			frame_next.copyTo(result);
			//RGB를 Gray로 변환
			cvtColor(frame_next, frame_next, CV_RGB2GRAY);
			//전 특징점을 저장.
			features_prev = features_next;

			//Optical Flow를 Lucas–Kanade method with Pyramid 방식으로 측정한다.
			calcOpticalFlowPyrLK(frame_prev, frame_next, features_prev, features_next, status, err);
			//전, 후 특징점을 네모로 표시하고, 전, 후 특징점을 빨간 줄로 이어준다,
			int zeroStatus = 0;

			//vector<float> lineLength;
			float minimum[1000][NUM];
			int minimumIdx[1000][NUM];

			if(!isCalc) {
				for(int i=0; i<features_next.size(); i++) {
					for(int j=0; j<NUM; j++) {
						minimum[i][j] = 100000;
						minimumIdx[i][j] = -1;
					}

					zeroStatus+=status[i];
					for(int j=i; j<features_next.size(); j++) {
						float x = features_next[i].x - features_next[j].x;
						float y = features_next[i].y - features_next[j].y;
						float dist = sqrt(x*x + y*y);

						if(dist < MAX_DIST) {
							for(int k=0; k<NUM; k++)
								if(minimum[i][k] > dist) {
									minimum[i][k] = dist;
									minimumIdx[i][k] = j;
									break;
								}
						}
					}
				}
			}

			for(int i=0; i<features_next.size(); i++)
				for(int j=0; j<NUM; j++)
					if(minimumIdx[i][j] != -1)
						line(result, features_next[i], features_next[minimumIdx[i][j]], Scalar(0, 0, 255));


			if(zeroStatus <= keyPointSize*reKeyPointRatio)
				break;

			imshow("result", result);
			waitKey(1000/fps);
			end = clock();
			std::cout << end - start << std::endl;
		}

	}
    cap.release();
}

void main() {
	BackgroundSubtractionWithOpticalFlowTest();
}