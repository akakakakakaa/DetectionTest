#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <iostream>
#include <time.h>
using namespace cv;

#define DIVIDE 16

struct Identifier {
	Point2f direc_v;
	Point2f point;

	Identifier(Point2f x_, Point2f y_) :direc_v(x_), point(y_) {}
};

struct MyPoint2f {
	int idx;
	Point2f point;

	MyPoint2f(int idx_, Point2f point_) : idx(idx_), point(point_) {}
};

struct MyDist {
	int idx;
	float dist;

	MyDist(int idx_, float dist_) : idx(idx_), dist(dist_) {}
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

//Draw voronoi diagram
void draw_voronoi( Mat& img, Subdiv2D& subdiv ) {
    vector<vector<Point2f> > facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
 
    vector<Point> ifacet;
    vector<vector<Point> > ifacets(1);
 
    for( size_t i = 0; i < facets.size(); i++ )
    {
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ )
            ifacet[j] = facets[i][j];
 
        Scalar color;
        color[0] = rand() & 255;
        color[1] = rand() & 255;
        color[2] = rand() & 255;
        fillConvexPoly(img, ifacet, color, 8, 0);
 
        ifacets[0] = ifacet;
        polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);
        circle(img, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);
    }
}

void draw_point( Mat& img, Point2f fp, Scalar color ) {
    circle( img, fp, 2, color, CV_FILLED, CV_AA, 0 );
}

void draw_delaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color ) { 
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
 
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
         
        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}

void BackgroundSubtractionWithOpticalFlowTest() {
	VideoCapture cap("test3.mp4");
	
	cv::Mat frame_prev, frame_next, frame_save, result, fgMaskMOG;
	vector<Point2f> initial_features, features_prev, features_next;
	vector<vector<MyPoint2f>> initial_points[DIVIDE];

	vector<vector<float>> lineLengths;
	vector<uchar> status;
    vector<float> err;
	int max_count = 10000;
	double qlevel = 0.01;
	double min_dist = 10;
	int fps = 5;
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	int stride = 10;

	clock_t start, end;
	while(true) {
        cap >> frame_next;
		//blur(frame_next, frame_next, Size(3, 3));
		//cv::GaussianBlur(frame_next, frame_save, cv::Size(0, 0), 3);
		//cv::addWeighted(frame_next, 1.5, frame_save, -0.5, 0, frame_next);
		
		cvtColor(frame_next, frame_next, CV_RGB2GRAY);
		getKeyPoint(frame_next, initial_features, max_count, qlevel, min_dist);
		/*
		for(int i=0; i<width; i+=stride)
			for(int j=0; j<height; j+=stride)
				features_next.push_back(Point2f(i, j));
		*/
		int keyPointSize = initial_features.size();
		features_prev = initial_features;
		double reKeyPointRatio = 0.5;

		for(int i=0; i<DIVIDE; i++) {
			initial_points[i].clear();
			for(int j=0; j<DIVIDE; j++)
				initial_points[i].push_back(vector<MyPoint2f>());
		}

		for(int i=0; i<initial_features.size(); i++) {
			int x = (int)initial_features[i].x / (width / DIVIDE);
			int y = (int)initial_features[i].y / (height / DIVIDE);
			initial_points[y][x].push_back(MyPoint2f(i, initial_features[i]));
		}

		//vector<Point2f> movedPoint;
		while(true) {
			start = clock();
			//전 Gray 이미지를 저장 
			frame_prev = frame_next.clone();
			//RGB값으로 가지고 옴
			cap >> frame_next;
			//blur(frame_next, frame_next, Size(3, 3));
			//cv::GaussianBlur(frame_next, frame_save, cv::Size(0, 0), 3);
			//cv::addWeighted(frame_next, 1.5, frame_save, -0.5, 0, frame_next);

			//화면을 보여주기 위한 RGB 이미지를 Gray로 변환하기 전에 저장
			frame_next.copyTo(result);
			//RGB를 Gray로 변환
			cvtColor(frame_next, frame_next, CV_RGB2GRAY);

			//Optical Flow를 Lucas–Kanade method with Pyramid 방식으로 측정한다.
			calcOpticalFlowPyrLK(frame_prev, frame_next, features_prev, features_next, status, err);
			//전, 후 특징점을 네모로 표시하고, 전, 후 특징점을 빨간 줄로 이어준다,
			int zeroStatus = 0;

			Subdiv2D subdiv(cv::Rect(0, 0, width, height));
			vector<vector<MyDist>> dists[DIVIDE];
			vector<Point2f> movedPoint;

			for(int i=0; i<DIVIDE; i++)
				for(int j=0; j<DIVIDE; j++)
					dists[i].push_back(vector<MyDist>());

			for(int y=0; y<DIVIDE; y++)
				for(int x=0; x<DIVIDE; x++) {
					vector<MyPoint2f>::iterator it = initial_points[y][x].begin();
					if(it != initial_points[y][x].end()) {
						float avgDist = 0;
						while(it != initial_points[y][x].end()) {
							int idx = it->idx;
							int xIdx = (int)features_next[idx].x / (width / DIVIDE);
							int yIdx = (int)features_next[idx].y / (height / DIVIDE);
						
							if(xIdx < 0 || xIdx >= DIVIDE || yIdx < 0 || yIdx >= DIVIDE)
								it = initial_points[y][x].erase(it);							
							else if(yIdx != y || xIdx != x) {
								it = initial_points[y][x].erase(it);
								initial_points[yIdx][xIdx].push_back(MyPoint2f(idx, features_next[idx]));
							}
							else {
								float dist = norm(features_next[idx] - it->point);
								if(dist >= 0.3) {
									dists[y][x].push_back(MyDist(idx, dist));
								}
								avgDist += dist;
								++it;
							}
						}
						
						int count = 0;
						if(dists[y][x].size() >= 10) {
							avgDist /= dists[y][x].size();
							for(int k=0; k<dists[y][x].size(); k++) {
								if(abs(avgDist - dists[y][x][k].dist) >= 10) {
									count++;
									//movedPoint.push_back(features_next[dists[y][x][k].idx]);
								}
							}

							if((float)count / dists[y][x].size() >= 1/2.0) {
								rectangle(result, Point2f(x * width / DIVIDE, y * height / DIVIDE), Point2f((x + 1) * width / DIVIDE,(y + 1) * height / DIVIDE), Scalar(255, 0, 0), CV_FILLED, 8, 0);
								it = initial_points[y][x].begin();
								while(it != initial_points[y][x].end()) {
									it->point = features_next[it->idx];
									it++;
								}
							}
						}
					}
				}

			for(int i=0; i<features_next.size(); i++) {
				zeroStatus+=status[i];
				Point2f point = initial_features.at(i);
				Point2f next_point = features_next.at(i);
				if(norm(point - next_point) >= 1000) {
					//movedPoint.push_back(initial_features.at(i));
					initial_features.at(i) = features_next.at(i);
					//movedPoint.push_back(features_next.at(i));
				}
				line(result, initial_features[i], features_next[i], Scalar(0, 255, 0));
				if(next_point.x <= width && next_point.y <= height && next_point.x >= 0 && next_point.y >= 0)
					subdiv.insert(features_next.at(i));
			}
			draw_delaunay(result, subdiv, Scalar(0, 0, 255));

			/*
			for(int i=0; i<features_next.size(); i++) {
				zeroStatus+=status[i];
				Point2f point = initial_features.at(i);
				Point2f next_point = features_next.at(i);
				if(norm(point - next_point) >= 1) {
					movedPoint.push_back(initial_features.at(i));
					initial_features.at(i) = features_next.at(i);
					movedPoint.push_back(features_next.at(i));
				}

				if(next_point.x <= width && next_point.y <= height && next_point.x >= 0 && next_point.y >= 0)
					subdiv.insert(features_next.at(i));
				// Show animation
				// Draw delaunay triangles
				//draw_delaunay(result, subdiv, Scalar(255, 0, 0));
				//line(result, initial_features[i], features_next[i], Scalar(0, 0, 255));
			}

			draw_delaunay(result, subdiv, Scalar(0, 0, 255));
			for( vector<Point2f>::iterator it = features_next.begin(); it != features_next.end(); it++)
				draw_point(result, *it, Scalar(0, 255, 0));
			*/
			//Mat img_voronoi = Mat::zeros(result.rows, result.cols, CV_8UC3);
			//draw_voronoi(img_voronoi, subdiv);
			//imshow("voronoi", img_voronoi);
			/*
			//vector<float> lineLength;
			vector<Point2f> movedPoint;
			float avgDist = 0;
			for(int i=0; i<features_next.size(); i++) {
				zeroStatus+=status[i];
				//Rect rect(Point(features_prev[i].x, features_prev[i].y), Point(features_prev[i].x + 3, features_prev[i].y + 3));
				//Rect rect2(Point(features_next[i].x, features_next[i].y), Point(features_next[i].x + 3, features_next[i].y + 3));
				//rectangle(result, rect.tl(), rect.br(), Scalar(255, 0, 0), 1, 8, 0);
				//rectangle(result, rect2.tl(), rect2.br(), Scalar(0, 255, 0), 1, 8, 0);
				float x = features_next[i].x - features_prev[i].x;
				float y = features_next[i].y - features_prev[i].y;
				avgDist += sqrt(x*x + y*y) / features_next.size();

				//lineLength.push_back(length);
				line(result, features_prev[i], features_next[i], Scalar(0, 0, 255));
			}
			*/
			/*
			for(int i=0; i<features_next.size(); i++) {
				float x = features_next[i].x - features_prev[i].x;
				float y = features_next[i].y - features_prev[i].y;
				float diff = sqrt(x*x + y*y) / features_next.size() - avgDist;
				if(avgDist >= 0.3 || avgDist <= -0.3)
					movedPoint.push_back(features_next[i]);
			}
			std::cout << "moved point size is " << movedPoint.size() << std::endl;
			*/
			/*
			int size = 5;
			if(movedPoint.size() >= size) {
				Mat labels, centers;
				TermCriteria tc;
				kmeans(movedPoint, size, labels, tc, 3, cv::KMEANS_PP_CENTERS, centers);
				vector<vector<Point2f>> clustered;
				for(int i=0; i<size; i++)
					clustered.push_back(vector<Point2f>());
				for(int i=0; i<movedPoint.size(); i++)
					clustered[labels.at<int>(i)].push_back(movedPoint[i]);

				vector<Rect> rects;
				for(int i=0; i<size; i++) {
					int minX = 10000;
					int minY = 10000;
					int maxX = -1;
					int maxY = -1;
					for(int j=0; j<clustered[i].size(); j++) {
						if(minX > clustered[i][j].x)
							minX = clustered[i][j].x;
						if(maxX < clustered[i][j].x)
							maxX = clustered[i][j].x;
						if(minY > clustered[i][j].y)
							minY = clustered[i][j].y;
						if(maxY < clustered[i][j].y)
							maxY = clustered[i][j].y;

					}
					if(minX != maxX && minY != maxY)
						rects.push_back(Rect(Point(minX, minY), Point(maxX, maxY)));
				}

				int minX;
				int minY;
				int maxX;
				int maxY;
				bool check = false;
				for(int i=0; i<rects.size(); i++) {
					Rect rect = rects.at(i);
					for(int j=0; j<rects.size(); j++) {
						Rect obj = rects.at(j);
						Point center((int) (rect.tl().x + rect.br().x / 2), (int) (rect.tl().y + rect.br().y / 2));
						Point center2((int) (obj.tl().x + obj.br().x / 2), (int) (obj.tl().y + obj.br().y / 2));
						double distance = sqrt(pow((double)center.x - center2.x, 2) + pow((double)center.y - center2.y, 2));
						if (distance <= 100) {
							minX = min(obj.tl().x, rect.tl().x);
							minY = min(obj.tl().y, rect.tl().y);
							maxX = max(obj.br().x, rect.br().x);
							maxY = max(obj.br().y, rect.br().y);
							Rect obj2 (Point(minX, minY), Point(maxX, maxY));
							rects.erase(rects.begin() + j);
							rects.push_back(obj2);
							check = true;
							break;
						}
					}
					if(check == false)
						rects.push_back(rect);
					check = false;
				}
				
				for(int i=0; i<rects.size(); i++)
					rectangle(result, rects[i].tl(), rects[i].br(), Scalar(255, 0, 0), 1, 8, 0);
				movedPoint.clear();
			}
			*/
			if(zeroStatus <= keyPointSize*reKeyPointRatio)
				break;

			imshow("result", result);
			waitKey(1000/fps);
			end = clock();
			std::cout << end - start << std::endl;

			//전 특징점을 저장.
			features_prev = features_next;
		}

	}
    cap.release();
}

void main() {
	BackgroundSubtractionWithOpticalFlowTest();
}