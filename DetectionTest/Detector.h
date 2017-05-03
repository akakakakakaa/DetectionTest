#pragma once
#include <iostream>
#include <math.h>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\calib3d\calib3d.hpp>
using namespace std;
using namespace cv;
#define GRID_SIZE 4

struct SGM {
	float mean;
	float variance;
	float age;
	float sgm_threshold;
	float detection_threshold;
	float variance_threshold;
	float decaying_parameter;
};

class Detector {
public:
	Detector(int m_width, int m_height, float m_sgm_threshold, float m_detection_threshold, float m_variance_threshold, float m_decaying_parameter, int m_key_point_number, float m_qlevel, float m_min_dist) {
		grid_width = m_width / GRID_SIZE;
		grid_height = m_height / GRID_SIZE;
		grid_pixels_number = grid_width * grid_height;
		sgm_threshold = m_sgm_threshold;
		detection_threshold = m_detection_threshold;
		variance_threshold = m_variance_threshold;
		decaying_parameter = m_decaying_parameter;
		key_point_number = m_key_point_number;
		qlevel = m_qlevel;
		min_dist = m_min_dist;
	}

	~Detector() {
	}

	void init(Mat frame) {
		Mat sub_frame;
		float variance, mean;

		for(int y=0; y<GRID_SIZE; y++)
			for(int x=0; x<GRID_SIZE; x++) {
				sub_frame = frame(Rect(x*grid_width, y*grid_height, grid_width, grid_height));
				cvtColor(sub_frame, sub_frame, CV_RGB2GRAY);

				//for SGM
				mean = (sum(sub_frame)[0]) / grid_pixels_number;
				variance = 0;
				for(int i=0; i<grid_pixels_number; i++)
					variance = max(variance, pow(mean - sub_frame.data[i], 2));
				//variance = calcVariance3(mean, sub_frame.data, 0, 0);
				avg_pixels = (sum(sub_frame)[0]) / grid_pixels_number;

				A_SGMs[y][x].age = 1;
				A_SGMs[y][x].mean = mean;
				A_SGMs[y][x].variance = variance;
				A_SGMs[y][x].decaying_parameter = decaying_parameter;
				A_SGMs[y][x].detection_threshold = detection_threshold;
				A_SGMs[y][x].sgm_threshold = sgm_threshold;
				A_SGMs[y][x].variance_threshold = variance_threshold;
				C_SGMs[y][x].age = 0;
				C_SGMs[y][x].decaying_parameter = decaying_parameter;
				C_SGMs[y][x].detection_threshold = detection_threshold;
				C_SGMs[y][x].sgm_threshold = sgm_threshold;
				C_SGMs[y][x].variance_threshold = variance_threshold;

				//for motion compensation
				goodFeaturesToTrack(sub_frame, prev_points[y][x], key_point_number, qlevel, min_dist);
				prev_frames[y][x] = sub_frame;
			}
		center.push_back(Point2f(grid_width/2, grid_height/2));
	}

	//function 1.
	float calcMean(float m_prev_age, float m_prev_mean, float m_avg_pixels) {
		return m_prev_age * m_prev_mean / (m_prev_age + 1) + m_avg_pixels / (m_prev_age + 1);
	}

	//function 2 using function 5
	float calcVariance(float m_mean, uchar* data, float m_prev_age, float m_prev_variance) {
		float max_variance = 0;
		for(int i=0; i<grid_pixels_number; i++)
			max_variance = max(max_variance, pow(m_mean - data[i], 2));

		return m_prev_age * m_prev_variance / (m_prev_age + 1) + max_variance / (m_prev_age + 1);
	}

	//function 2 using function 6
	float calcVariance2(float m_mean, float m_avg_pixels, float m_prev_age, float m_prev_variance) {
		float variance = pow(m_mean - m_avg_pixels, 2);

		return m_prev_age * m_prev_variance / (m_prev_age + 1) + variance / (m_prev_age + 1);
	}

	float calcVariance3(float m_mean, uchar* data, float m_prev_age, float m_prev_variance) {
		float variance = 0;
		for(int i=0; i<grid_pixels_number; i++)
			variance += abs(m_mean - data[i]);

		return m_prev_age * m_prev_variance / (m_prev_age + 1) + variance / (m_prev_age + 1);
	}

	//function 3.
	float calcAge(float m_age) {
		return m_age + 1;
	}

	//function 7,8.
	bool calcSquareDiff(float m_avg_pixels, float m_mean, float m_threshold, float m_variance) {
		return (pow(m_avg_pixels - m_mean, 2) < m_threshold * m_variance);
	}

	//function 16
	bool calcClassifyPixel(int pixel, float m_mean, float m_threshold, float m_variance) {
		return (pow(pixel - m_mean, 2) > m_threshold * m_variance);
	}
	
	void dualModeSGM(Mat m_sub_frame, SGM& A_SGM, SGM& C_SGM) {
		avg_pixels = (sum(m_sub_frame)[0]) / grid_pixels_number;

		//for SGM
		//for apparent background model
		A_prev_age = A_SGM.age;
		A_prev_mean = A_SGM.mean;
		A_prev_variance = A_SGM.variance;

		A_mean = calcMean(A_prev_age, A_prev_mean, avg_pixels);
		A_variance = calcVariance(A_mean, m_sub_frame.data, A_prev_age, A_prev_variance);
		//A_variance = calcVariance2(A_mean, avg_pixels, A_prev_age, A_prev_variance);
		//A_variance = calcVariance3(A_mean, m_sub_frame.data, A_prev_age, A_prev_variance);

		if(calcSquareDiff(avg_pixels, A_mean, A_SGM.sgm_threshold, A_variance)) {
			A_SGM.age++;
			A_SGM.mean = A_mean;
			A_SGM.variance = A_variance;
		}
		else {
			//for candidate background model
			//not initialized.
			if(C_SGM.age == 0) {
				//C_SGMs[y][x].age = A_prev_age + 1; ?
				//C_SGMs[y][x].age++; ?
				//or not touch age?
				C_SGM.age = 1;
				C_SGM.mean = A_mean;
				C_SGM.variance = A_variance;
			}
			else {
				C_prev_age = C_SGM.age;
				C_prev_mean = C_SGM.mean;
				C_prev_variance = C_SGM.variance;

				C_mean = calcMean(C_prev_age, C_prev_mean, avg_pixels);
				C_variance = calcVariance(C_mean, m_sub_frame.data, C_prev_age, C_prev_variance);
				//C_variance = calcVariance2(C_mean, avg_pixels, C_prev_age, C_prev_variance);
				//C_variance = calcVariance3(C_mean, m_sub_frame.data, C_prev_age, C_prev_variance);

				if(calcSquareDiff(avg_pixels, C_mean, C_SGM.sgm_threshold, C_variance)) {
					C_SGM.age++;
					C_SGM.mean = C_mean;
					C_SGM.variance = C_variance;
				}
				else {
					//If none of the conditions hold, we initialize the candidate background model with the current observation
					//만약 아무 조건도 맞지 않는다면 age를 변화시키지 말란 소리일까?
					//C_SGMs[y][x].age++;
					//C_SGM.age++;
					C_SGM.mean = A_mean;
					C_SGM.variance = A_variance;
				}
			}
		}

		if(C_SGM.age > A_SGM.age) {
			SGM tmp = C_SGM;
			C_SGM = A_SGM;
			A_SGM = tmp;
		}
	}

	void motionCompensation(Mat m_sub_frame, int m_x, int m_y, Mat& prev_frame, vector<Point2f>& prev_points, vector<Point2f>& next_points) {
		//for motion compensation
		int x_diff;
		int y_diff;
		float grid_width_v = grid_width;
		float grid_height_v = grid_height;

		tmp_SGMs[m_y][m_x] = A_SGMs[m_y][m_x];
		if(prev_points.size() > 0) {
			calcOpticalFlowPyrLK(prev_frame, m_sub_frame, prev_points, next_points, status, err);
			if(next_points.size() >= 4) {
				Mat H = findHomography(next_points,  prev_points, CV_RANSAC);
				if(countNonZero(H) >= 1) {
					vector<Point2f> next_center;
					perspectiveTransform(center, next_center, H);
					
					int prev_x = center[0].x;
					int prev_y = center[0].y;
					int next_x = next_center[0].x;
					int next_y = next_center[0].y;
					
					Point idx[4];
					Point2f weight[4];
					int block_num = 1;
					int x_idx = m_x + (next_x - grid_width / 2) / grid_width;
					int y_idx = m_y + (next_y - grid_height / 2) / grid_height;
					//cout << x_idx << " " << y_idx << endl;
					int x_remainder = next_x % grid_width;
					int y_remainder = next_y % grid_height;

					if(x_remainder >= 0) {
						if(x_remainder < grid_width_v/2) {
							idx[0].x = (x_idx - 1 < 0) ? 0 : x_idx - 1;
							weight[0].x = grid_width_v / 2 - x_remainder;
							idx[1].x = x_idx;
							weight[1].x = grid_width_v / 2 + x_remainder;
							idx[2].x = (x_idx - 1 < 0) ? 0 : x_idx - 1;
							weight[2].x = grid_width_v / 2 - x_remainder;
							idx[3].x = x_idx;
							weight[3].x = grid_width_v / 2 + x_remainder;
							block_num*=2;
						}
						else if(x_remainder == grid_width_v/2) {
							idx[0].x = x_idx;
							weight[0].x = grid_width_v;
							idx[1].x = x_idx;
							weight[1].x = grid_width_v;
							block_num*=1;
						}
						else {
							idx[0].x = x_idx;
							weight[0].x = 3 * grid_width_v / 2 - x_remainder;
							idx[1].x = (x_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : x_idx + 1;
							weight[1].x = x_remainder - grid_width_v / 2;
							idx[2].x = x_idx;
							weight[2].x = 3 * grid_width_v / 2 - x_remainder;
							idx[3].x = (x_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : x_idx + 1;
							weight[3].x = x_remainder - grid_width_v / 2;
							block_num*=2;
						}
					}
					else if(x_remainder < 0) {
						if(x_remainder > -grid_width_v/2) {
							idx[0].x = (x_idx - 1 < 0) ? 0 : x_idx - 1;
							weight[0].x = grid_width_v / 2 - x_remainder;
							idx[1].x = x_idx;
							weight[1].x = grid_width_v / 2 + x_remainder;
							idx[2].x = (x_idx - 1 < 0) ? 0 : x_idx - 1;
							weight[2].x = grid_width_v / 2 - x_remainder;
							idx[3].x = x_idx;
							weight[3].x = grid_width_v / 2 + x_remainder;
							block_num*=2;
						}
						else if(x_remainder == -grid_width_v/2) {
							idx[0].x = x_idx;
							weight[0].x = grid_width_v;
							idx[1].x = x_idx;
							weight[1].x = grid_width_v;
							block_num*=1;
						}
						else if(x_remainder < -grid_width_v/2) {
							idx[0].x = x_idx;
							weight[0].x = -x_remainder - grid_width_v / 2;
							idx[1].x = (x_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : x_idx + 1;
							weight[1].x = 3 * grid_width_v / 2 + x_remainder;
							idx[2].x = x_idx;
							weight[2].x = -x_remainder - grid_width_v / 2;
							idx[3].x = (x_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : x_idx + 1;
							weight[3].x = 3 * grid_width_v / 2 + x_remainder;
							block_num*=2;
						}
					}
					if(y_remainder >= 0) {
						if(abs(y_remainder) < grid_height_v/2) {
							idx[0].y = (y_idx - 1 < 0) ? 0 : y_idx - 1;;
							weight[0].y = grid_height_v / 2 - y_remainder;
							idx[1].y = y_idx;
							weight[1].y = grid_height_v / 2 - y_remainder;
							idx[2].y = (x_idx - 1 < 0) ? 0 : y_idx - 1;
							weight[2].y = grid_height_v / 2 + y_remainder;
							idx[3].y = y_idx;
							weight[3].y = grid_height_v / 2 + y_remainder;
							block_num*=2;
						}
						else if(abs(y_remainder) == grid_height_v/2) {
							idx[0].y = y_idx;
							weight[0].y = grid_height_v;
							idx[1].y = y_idx;
							weight[2].y = grid_height_v;
							block_num*=1;
						}
						else {
							idx[0].y = y_idx;
							weight[0].y = 3 * grid_height_v / 2 - y_remainder;
							idx[1].y = (y_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : y_idx + 1;
							weight[1].y = 3 * grid_height_v / 2 - y_remainder;
							idx[2].y = y_idx;
							weight[2].y = y_remainder - grid_height_v / 2;
							idx[3].y = (y_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : y_idx + 1;
							weight[3].y = y_remainder - grid_height_v / 2;
							block_num*=2;
						}
					}
					else if(y_remainder < 0) {
						if(y_remainder > -grid_height_v/2) {
							idx[0].y = (y_idx - 1 < 0) ? 0 : y_idx - 1;;
							weight[0].y = grid_height_v / 2 - y_remainder;
							idx[1].y = y_idx;
							weight[1].y = grid_height_v / 2 - y_remainder;
							idx[2].y = (x_idx - 1 < 0) ? 0 : y_idx - 1;
							weight[2].y = grid_height_v / 2 + y_remainder;
							idx[3].y = y_idx;
							weight[3].y = grid_height_v / 2 + y_remainder;
							block_num*=2;
						}
						else if(y_remainder == -grid_height_v/2) {
							idx[0].y = y_idx;
							weight[0].y = grid_height_v;
							idx[1].y = y_idx;
							weight[2].y = grid_height_v;
							block_num*=1;
						}
						else if(y_remainder < -grid_height_v/2) {
							idx[0].y = y_idx;
							weight[0].y = -y_remainder - grid_height_v / 2;
							idx[1].x = (y_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : y_idx + 1;
							weight[1].y = -y_remainder - grid_height_v / 2;
							idx[2].x = x_idx;
							weight[2].y = 3 * grid_height_v / 2 + y_remainder;
							idx[3].y = (y_idx + 1 > GRID_SIZE-1) ? GRID_SIZE - 1 : y_idx + 1;
							weight[3].y = 3 * grid_height_v / 2 + y_remainder;
							block_num*=2;
						}
					}

					if(idx[0].x >= 0 && idx[0].x < GRID_SIZE &&
						idx[1].x >= 0 && idx[1].x < GRID_SIZE &&
						idx[2].x >= 0 && idx[2].x < GRID_SIZE &&
						idx[3].x >= 0 && idx[3].x < GRID_SIZE &&
						idx[0].y >= 0 && idx[0].y < GRID_SIZE &&
						idx[1].y >= 0 && idx[1].y < GRID_SIZE &&
						idx[2].y >= 0 && idx[2].y < GRID_SIZE &&
						idx[3].y >= 0 && idx[3].y < GRID_SIZE) {

						float weight_rate[4] = {0, 0, 0, 0};
						if(block_num == 1) {
							weight_rate[0] = 1;
							tmp_SGMs[m_y][m_x] = A_SGMs[m_y][m_x];
						}
						else if(block_num == 2) {
							weight_rate[0] = weight[0].x * weight[0].y / grid_pixels_number;
							weight_rate[1] = weight[1].x * weight[2].y / grid_pixels_number;
							tmp_SGMs[m_y][m_x].mean = A_SGMs[idx[0].y][idx[0].x].mean * weight_rate[0]
													+ A_SGMs[idx[1].y][idx[1].x].mean * weight_rate[1];
							tmp_SGMs[m_y][m_x].variance = weight_rate[0]*(A_SGMs[idx[0].y][idx[0].x].variance + pow(A_SGMs[idx[0].y][idx[0].x].mean, 2) - pow(tmp_SGMs[m_y][m_x].mean, 2))
													+weight_rate[1]*(A_SGMs[idx[1].y][idx[1].x].variance + pow(A_SGMs[idx[1].y][idx[1].x].mean, 2) - pow(tmp_SGMs[m_y][m_x].mean, 2));
							tmp_SGMs[m_y][m_x].age = A_SGMs[idx[0].y][idx[0].x].age * weight_rate[0]
													+ A_SGMs[idx[1].y][idx[1].x].age * weight_rate[1];
						}
						else {
							weight_rate[0] = weight[0].x * weight[0].y / grid_pixels_number;
							weight_rate[1] = weight[1].x * weight[1].y / grid_pixels_number;
							weight_rate[2] = weight[2].x * weight[2].y / grid_pixels_number;
							weight_rate[3] = weight[3].x * weight[3].y / grid_pixels_number;
							tmp_SGMs[m_y][m_x].mean = A_SGMs[idx[0].y][idx[0].x].mean * weight_rate[0]
													+ A_SGMs[idx[1].y][idx[1].x].mean * weight_rate[1]
													+ A_SGMs[idx[2].y][idx[2].x].mean * weight_rate[2]
													+ A_SGMs[idx[3].y][idx[3].x].mean * weight_rate[3];
							tmp_SGMs[m_y][m_x].variance = weight_rate[0]*(A_SGMs[idx[0].y][idx[0].x].variance + pow(A_SGMs[idx[0].y][idx[0].x].mean, 2) - pow(tmp_SGMs[m_y][m_x].mean, 2))
													+weight_rate[1]*(A_SGMs[idx[1].y][idx[1].x].variance + pow(A_SGMs[idx[1].y][idx[1].x].mean, 2) - pow(tmp_SGMs[m_y][m_x].mean, 2))
													+weight_rate[2]*(A_SGMs[idx[2].y][idx[2].x].variance + pow(A_SGMs[idx[2].y][idx[2].x].mean, 2) - pow(tmp_SGMs[m_y][m_x].mean, 2))
													+weight_rate[3]*(A_SGMs[idx[3].y][idx[3].x].variance + pow(A_SGMs[idx[3].y][idx[3].x].mean, 2) - pow(tmp_SGMs[m_y][m_x].mean, 2));
							tmp_SGMs[m_y][m_x].age = A_SGMs[idx[0].y][idx[0].x].age * weight_rate[0]
													+ A_SGMs[idx[1].y][idx[1].x].age * weight_rate[1]
													+ A_SGMs[idx[2].y][idx[2].x].age * weight_rate[2]
													+ A_SGMs[idx[3].y][idx[3].x].age * weight_rate[3];
						}
						if(tmp_SGMs[m_y][m_x].mean <= 0 || tmp_SGMs[m_y][m_x].variance <= 0 || tmp_SGMs[m_y][m_x].age <= 0)
							cout << "aa" << endl;

						if(tmp_SGMs[m_y][m_x].variance > tmp_SGMs[m_y][m_x].variance_threshold) {
							tmp_SGMs[m_y][m_x].age = tmp_SGMs[m_y][m_x].age * exp(-decaying_parameter*(tmp_SGMs[m_y][m_x].variance / grid_pixels_number - tmp_SGMs[m_y][m_x].variance_threshold));
							//tmp_SGMs[m_y][m_x].variance_threshold += exp(-decaying_parameter*(tmp_SGMs[m_y][m_x].variance / grid_pixels_number - tmp_SGMs[m_y][m_x].variance_threshold));
						}
						if(weight_rate[0] + weight_rate[1] + weight_rate[2] + weight_rate[3] < 0.99)
							cout << "bb" << endl;
						//if(tmp_SGMs[m_y][m_x].age < 0.95)
						//	cout << "aa" << tmp_SGMs[m_y][m_x].age << endl;
						prev_frame = m_sub_frame;
						prev_points = next_points;
						return;
					}

						/*
					Point2f yxCenter(x*grid_width_v + grid_width_v/2, y*grid_height_v + grid_height_v/2);
					Point2f yxCenter2(center2.at(0).x + x * grid_width_v, center2.at(0).y + y * grid_height_v);
					if(norm(yxCenter - yxCenter2) <= 10) {
								
					}
					*/
					//	line(result, yxCenter, yxCenter2, Scalar(255, 0, 0), 1, 8);
				}
				else {
					//cout << H << endl;
				}
			}
		}

		tmp_SGMs[m_y][m_x] = A_SGMs[m_y][m_x];
		prev_frame = m_sub_frame;
		prev_points = next_points;
	}

	void detectionForgroundPixels(Mat m_sub_frame, Mat m_sub_show_frame, SGM A_SGMs) {
		int count = 0;
		for(int	i=0; i<grid_pixels_number; i++) {
			if(calcClassifyPixel(m_sub_frame.data[i], A_SGMs.mean, A_SGMs.detection_threshold, A_SGMs.variance)) {
				m_sub_show_frame.data[i] = 255;
				count++;
			}
		}
		if(count >= 4 * grid_width * grid_height / 5)
			cout << count << endl;
	}

	void compute(Mat frame) {
		imshow("original", frame);
		waitKey(1);

		Mat show_frame = Mat(grid_height * GRID_SIZE, grid_width * GRID_SIZE, CV_8UC1);
		Mat sub_frame[GRID_SIZE][GRID_SIZE];
		Mat sub_show_frame[GRID_SIZE][GRID_SIZE];
		for(int y=0; y<GRID_SIZE; y++)
			for(int x=0; x<GRID_SIZE; x++) {
				frame(Rect(x*grid_width, y*grid_height, grid_width, grid_height)).copyTo(sub_frame[y][x]);
				cvtColor(sub_frame[y][x], sub_frame[y][x], CV_RGB2GRAY);
				sub_frame[y][x].copyTo(sub_show_frame[y][x]);

				motionCompensation(sub_frame[y][x], x, y, prev_frames[y][x], prev_points[y][x], next_points[y][x]);
			}

		for(int y=0; y<GRID_SIZE; y++)
			for(int x=0; x<GRID_SIZE; x++)
				A_SGMs[y][x] = tmp_SGMs[y][x];

		for(int y=0; y<GRID_SIZE; y++)
			for(int x=0; x<GRID_SIZE; x++) {
				dualModeSGM(sub_frame[y][x], A_SGMs[y][x], C_SGMs[y][x]);
				detectionForgroundPixels(sub_frame[y][x], sub_show_frame[y][x], A_SGMs[y][x]);
				sub_show_frame[y][x].copyTo(show_frame(Rect(x*grid_width, y*grid_height, grid_width, grid_height)));
			}

		imshow("result", show_frame);
		waitKey(1);
	}

private:
	int grid_width;
	int grid_height;
	int grid_pixels_number;

	//for dualmode sgm
	SGM A_SGMs[GRID_SIZE][GRID_SIZE];
	SGM tmp_SGMs[GRID_SIZE][GRID_SIZE];
	SGM C_SGMs[GRID_SIZE][GRID_SIZE];
	float sgm_threshold; //for candidate SGM
	float avg_pixels;
	float A_prev_age, A_prev_mean, A_prev_variance, A_mean, A_variance, A_max_variance;
	float C_prev_age, C_prev_mean, C_prev_variance, C_mean, C_variance, C_max_variance;

	//for motion compensation
	vector<Point2f> prev_points[GRID_SIZE][GRID_SIZE], next_points[GRID_SIZE][GRID_SIZE];
	Mat prev_frames[GRID_SIZE][GRID_SIZE];
	vector<Point2f> center;
	int key_point_number;
	float qlevel;
	float min_dist;
	vector<uchar> status;
	vector<float> err;
	float mixture_weight[4];
	float variance_threshold;
	float decaying_parameter;

	//for detection
	float detection_threshold;
};
