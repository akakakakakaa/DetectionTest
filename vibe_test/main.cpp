#include "bgfg_vibe.hpp"
#include <time.h>
using namespace cv;
/*
int main(int argc, char ** argv)
{
	Mat rgb_frame, gray_frame, blur_gray_frame, lab_frame, grad_x, grad_y;
	double x_sum, y_sum;
	bgfg_vibe bgfg_rgb, bgfg_gray, bgfg_blur_gray, bgfg_lab;

	VideoCapture cap("test.mp4");
	cap >> rgb_frame;
	resize(rgb_frame, rgb_frame, Size(320, 240));
	cvtColor(rgb_frame, gray_frame, CV_RGB2YUV);
	GaussianBlur(gray_frame, blur_gray_frame, Size(3, 3), 0, 0);
	//cvtColor(rgb_frame, lab_frame, CV_RGB2Lab);

	//bgfg_rgb.init_model(rgb_frame);
	bgfg_gray.init_model(gray_frame);
	bgfg_blur_gray.init_model(blur_gray_frame);
	//bgfg_lab.init_model(lab_frame);

	clock_t start, end;
	for (;;)
	{
		start = clock();
		cap >> rgb_frame;
		resize(rgb_frame, rgb_frame, Size(320, 240));
		cvtColor(rgb_frame, gray_frame, CV_RGB2GRAY);
		//GaussianBlur(gray_frame, blur_gray_frame, Size(3, 3), 0, 0);
		//cvtColor(rgb_frame, lab_frame, CV_RGB2Lab);

		//Mat fg_rgb = *bgfg_rgb.fg(rgb_frame);
		Mat fg_gray = *bgfg_gray.fg(gray_frame);
		//Mat fg_blur_gray = *bgfg_blur_gray.fg(blur_gray_frame);
		//cv::Sobel(fg_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		//cv::convertScaleAbs(grad_x, grad_x);
		//cv::Sobel(fg_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
		//cv::convertScaleAbs(grad_y, grad_y);
		//x_sum = cv::sum(grad_x)[0] / (grad_x.cols * grad_x.rows);
		//y_sum = cv::sum(grad_y)[0] / (grad_y.cols * grad_y.rows);
		//printf("x_sum : %f, y_sum : %f \n", x_sum, y_sum);

		//imshow("grad_x", grad_x);
		//imshow("grad_y", grad_y);
		//Mat fg_lab = *bgfg_lab.fg(lab_frame);
		//imshow("fg_rgb", fg_rgb);
		imshow("fg_gray", fg_gray);
		//imshow("fg_blur_gray", fg_blur_gray);
		imshow("origin_gray", gray_frame);
		bgfg_gray.show_samples();
		waitKey(1);
		end = clock();
		printf("%dms\n", end - start);
	}
	return 0;
}
*/

int main() {
	Mat rgb_frame, gray_frame;
	bgfg_vibe bgfg_gray(3);//, bgfg_gray2(5), bgfg_gray3(7);
	VideoCapture cap("test.mp4");
	cap >> rgb_frame;
	resize(rgb_frame, rgb_frame, Size(320, 240));
	cvtColor(rgb_frame, gray_frame, CV_RGB2GRAY);

	bgfg_gray.init_model(gray_frame);
	//bgfg_gray2.init_model(gray_frame);
	//bgfg_gray3.init_model(gray_frame);

	clock_t start, end;
	for (;;)
	{
		start = clock();
		cap >> rgb_frame;
		resize(rgb_frame, rgb_frame, Size(320, 240));
		cvtColor(rgb_frame, gray_frame, CV_RGB2GRAY);

		Mat fg_gray = *bgfg_gray.fg(gray_frame);
		//Mat fg_gray2 = *bgfg_gray2.fg(gray_frame);
		//Mat fg_gray3 = *bgfg_gray3.fg(gray_frame);

		imshow("fg_gray", fg_gray);
		//imshow("fg_gray2", fg_gray2);
		//imshow("fg_gray3", fg_gray3);
		imshow("origin_gray", gray_frame);
		waitKey(1);
		end = clock();
		printf("%dms\n", end - start);
	}
	return 0;
}