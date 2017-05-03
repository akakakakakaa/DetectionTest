#ifndef bgfg_vibe_hpp
#define bgfg_vibe_hpp
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2\video\tracking.hpp>
#include <opencv2\calib3d\calib3d.hpp>

struct Model {
    cv::Mat*** samples;
    cv::Mat** fgch;
    cv::Mat* fg;
};

class bgfg_vibe
{
#define rndSize 256
    unsigned char ri;
#define rdx ri++
public:
    bgfg_vibe();
	bgfg_vibe(int kernel);
    int N,R,noMin,phi;
	int max_count;
	double accu_xdiff, accu_ydiff;
	double qlevel,min_dist;
	int grid, kernel;
    void init_model(cv::Mat& firstSample);
    void setphi(int phi);
    cv::Mat* fg(cv::Mat& frame);
	void motionCompensation(cv::Mat m_sub_frame, cv::Mat m_prev_sub_frame, std::vector<cv::Point2f>& prev_points);
	void calcMotionCompensationResult(int channel);
private:
    bool initDone;
    cv::RNG rnd;
    Model* model;
    void init();
    void fg1ch(cv::Mat& frame,cv::Mat** samples,cv::Mat* fg);
    int rndp[rndSize],rndn[rndSize],rnd8[rndSize];
	std::vector<cv::Point2f>** prevPoints;
	cv::Mat** prevSubFrames;
};

#endif