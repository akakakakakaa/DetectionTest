#include "bgfg_vibe.hpp"

bgfg_vibe::bgfg_vibe():R(10),N(10),noMin(2),phi(0)
{
    initDone=false;
    rnd=cv::theRNG();
    ri=0;
	max_count = 100;
	qlevel = 0.01;
	min_dist = 1;
	grid = 1;
	accu_xdiff = 0;
	accu_ydiff = 0;
}

bgfg_vibe::bgfg_vibe(int kernel):R(20),N(10),noMin(2),phi(0)
{
    initDone=false;
    rnd=cv::theRNG();
    ri=0;
	max_count = 100;
	qlevel = 0.01;
	min_dist = 1;
	grid = 1;
	this->kernel = kernel;
	accu_xdiff = 0;
	accu_ydiff = 0;
}

void bgfg_vibe::init()
{
    for(int i=0;i<rndSize;i++)
    {
        rndp[i]=rnd(phi);
        rndn[i]=rnd(N);
        rnd8[i]=rnd(kernel*kernel - 1);
    }
}
void bgfg_vibe::setphi(int phi)
{
    this->phi=phi;
    for(int i=0;i<rndSize;i++)
    {
        rndp[i]=rnd(phi);
    }
}
void bgfg_vibe::init_model(cv::Mat& firstSample)
{
    std::vector<cv::Mat> channels;
    split(firstSample,channels);
    if(!initDone)
    {
        init();
        initDone=true;
    }
    model=new Model;
    model->fgch= new cv::Mat*[channels.size()];
    model->samples=new cv::Mat**[N];
    model->fg=new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
    for(size_t s=0;s<channels.size();s++)
    {       
        model->fgch[s]=new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
        cv::Mat** samples= new cv::Mat*[N];
        for(int i=0;i<N;i++)
        {
            samples[i]= new cv::Mat(cv::Size(firstSample.cols,firstSample.rows), CV_8UC1);
        }
        for(int i=0;i<channels[s].rows;i++)
        {
            int ioff=channels[s].step.p[0]*i;
            for(int j=0;j<channels[0].cols;j++)
            {
                for(int k=0;k<1;k++)
                {
                    (samples[k]->data + ioff)[j]=channels[s].at<uchar>(i,j);
                }
                (model->fgch[s]->data + ioff)[j]=0;

                if(s==0)(model->fg->data + ioff)[j]=0;
            }
        }
        model->samples[s]=samples;
    }


	prevSubFrames = new cv::Mat*[grid];
	prevPoints = new std::vector<cv::Point2f>*[grid];
	for(int y=0; y<grid; y++) {
		prevSubFrames[y] = new cv::Mat[grid];
		prevPoints[y] = new std::vector<cv::Point2f>[grid];
		for(int x=0; x<grid; x++) {
			prevSubFrames[y][x] = cv::Mat(firstSample.rows, firstSample.cols, CV_8UC1);
			firstSample(cv::Rect(x*firstSample.cols / grid, y*firstSample.rows / grid, firstSample.cols / grid, firstSample.rows / grid)).copyTo(prevSubFrames[y][x]);
			//cvtColor(prevSubFrames[y][x], prevSubFrames[y][x], CV_RGB2GRAY);
			cv::goodFeaturesToTrack(prevSubFrames[y][x], prevPoints[y][x], max_count, qlevel, min_dist);
		}
	}

}
void bgfg_vibe::fg1ch(cv::Mat& frame,cv::Mat** samples,cv::Mat* fg)
{
    int step=frame.step.p[0];
    for(int i=kernel/2;i<frame.rows-kernel/2;i++)
    {
        int ioff= step*i;
        for(int j=kernel/2;j<frame.cols-kernel/2;j++)
        {
            int count =0,index=0;
            while((count<noMin) && (index<N))
            {
                int dist= (samples[index]->data + ioff)[j]-(frame.data + ioff)[j];
                if(dist<=R && dist>=-R)
                {
                    count++; 
                }
                index++;
            }
            if(count>=noMin)
            {
                ((fg->data + ioff))[j]=0;
                int rand= rndp[rdx];
                if(rand==0)
                {
                    rand= rndn[rdx];
                    (samples[rand]->data + ioff)[j]=(frame.data + ioff)[j];
                }
                rand= rndp[rdx];
                int nxoff=ioff;
                if(rand==0)
                {
                    int nx=i,ny=j;
                    int cases= rnd8[rdx];
					switch (cases)
					{
					case 0:
						//nx--;
						nxoff = ioff - step;
						ny--;
						break;
					case 1:
						//nx--;
						nxoff = ioff - step;
						ny;
						break;
					case 2:
						//nx--;
						nxoff = ioff - step;
						ny++;
						break;
					case 3:
						//nx++;
						nxoff = ioff + step;
						ny--;
						break;
					case 4:
						//nx++;
						nxoff = ioff + step;
						ny;
						break;
					case 5:
						//nx++;
						nxoff = ioff + step;
						ny++;
						break;
					case 6:
						//nx;
						ny--;
						break;
					case 7:
						//nx;
						ny++;
						break;
					case 8:
						nxoff = ioff - step * 2;
						ny -= 2;
						break;
					case 9:
						nxoff = ioff - step * 2;
						ny--;
						break;
					case 10:
						nxoff = ioff - step * 2;
						break;
					case 11:
						nxoff = ioff - step * 2;
						ny++;
						break;
					case 12:
						nxoff = ioff - step * 2;
						ny+=2;
						break;
					case 13:
						nxoff = ioff + step * 2;
						ny -= 2;
						break;
					case 14:
						nxoff = ioff + step * 2;
						ny--;
						break;
					case 15:
						nxoff = ioff + step * 2;
						break;
					case 16:
						nxoff = ioff + step * 2;
						ny++;
						break;
					case 17:
						nxoff = ioff + step * 2;
						ny += 2;
						break;
					case 18:
						nxoff = ioff - step;
						ny -= 2;
						break;
					case 19:
						nxoff = ioff;
						ny -= 2;
						break;
					case 20:
						nxoff = ioff + step;
						ny -= 2;
						break;
					case 21:
						nxoff = ioff - step;
						ny += 2;
						break;
					case 22:
						nxoff = ioff;
						ny += 2;
						break;
					case 23:
						nxoff = ioff + step;
						ny += 2;
						break;
					case 24:
						nxoff = ioff - step * 3;
						ny -= 3;
						break;
					case 25:
						nxoff = ioff - step * 3;
						ny -= 2;
						break;
					case 26:
						nxoff = ioff - step * 3;
						ny--;
						break;
					case 27:
						nxoff = ioff - step * 3;
						break;
					case 28:
						nxoff = ioff - step * 3;
						ny++;
						break;
					case 29:
						nxoff = ioff - step * 3;
						ny += 2;
						break;
					case 30:
						nxoff = ioff - step * 3;
						ny += 3;
						break;
					case 31:
						nxoff = ioff + step * 3;
						ny -= 3;
						break;
					case 32:
						nxoff = ioff + step * 3;
						ny -= 2;
						break;
					case 33:
						nxoff = ioff + step * 3;
						ny--;
						break;
					case 34:
						nxoff = ioff + step * 3;
						break;
					case 35:
						nxoff = ioff + step * 3;
						ny++;
						break;
					case 36:
						nxoff = ioff + step * 3;
						ny += 2;
						break;
					case 37:
						nxoff = ioff + step * 3;
						ny += 3;
						break;
					case 38:
						nxoff = ioff - step * 2;
						ny -= 3;
					case 39:
						nxoff = ioff - step;
						ny -= 3;
						break;
					case 40:
						nxoff = ioff;
						ny -= 3;
						break;
					case 41:
						nxoff = ioff + step;
						ny -= 3;
						break;
					case 42:
						nxoff = ioff + step * 2;
						ny -= 3;
						break;
					case 43:
						nxoff = ioff - step * 2;
						ny += 3;
					case 44:
						nxoff = ioff - step;
						ny += 3;
						break;
					case 45:
						nxoff = ioff;
						ny += 3;
						break;
					case 46:
						nxoff = ioff + step;
						ny += 3;
						break;
					case 47:
						nxoff = ioff + step * 2;
						ny += 3;
						break;
					}

                    rand= rndn[rdx];
                    (samples[rand]->data + nxoff)[ny]=(frame.data + ioff)[j];
                }
            }else
            {
                ((fg->data + ioff))[j]=255;
            }
        }
    }
}
cv::Mat* bgfg_vibe::fg(cv::Mat& frame)
{
    std::vector<cv::Mat> channels;
    split(frame,channels);
    for(size_t i=0;i<channels.size();i++)
    {
        fg1ch(channels[i],model->samples[i],model->fgch[i]);        
        if(i>0 && i<2)
        {
            bitwise_or(*model->fgch[i-1],*model->fgch[i],*model->fg);
        }
        if(i>=2)
        {
            bitwise_or(*model->fg,*model->fgch[i],*model->fg);
        }
    }

	cv::Mat subFrame;
	for(int y=0; y<grid; y++)
		for(int x=0; x<grid; x++) {
			frame(cv::Rect(x*frame.cols / grid, y*frame.rows / grid, frame.cols / grid, frame.rows / grid)).copyTo(subFrame);
			motionCompensation(subFrame, prevSubFrames[y][x], prevPoints[y][x]);
			subFrame.copyTo(prevSubFrames[y][x]);
		}
	calcMotionCompensationResult(frame.channels());
	if(channels.size()==1) return model->fgch[0];
	return model->fg;
}

void bgfg_vibe::motionCompensation(cv::Mat m_sub_frame, cv::Mat m_prev_sub_frame, std::vector<cv::Point2f>& prev_points) {
	//for motion compensation
	std::vector<cv::Point2f> next_points;
	std::vector<uchar> status;
	std::vector<float> err;
	
	double xdiff=0;
	double ydiff=0;
	if(prev_points.size() >= 4) {
		cv::calcOpticalFlowPyrLK(m_prev_sub_frame, m_sub_frame, prev_points, next_points, status, err);
		cv::Mat H = cv::findHomography(prev_points, next_points, CV_RANSAC);
		for(int i=0; i<prev_points.size(); i++) {
			xdiff += next_points.at(i).x - prev_points.at(i).x;
			ydiff += next_points.at(i).y - prev_points.at(i).y;
		}
		xdiff /= prev_points.size();
		ydiff /= prev_points.size();
	}
	accu_xdiff += xdiff;
	accu_ydiff += ydiff;

	printf("x : %f, y : %f\n", xdiff, ydiff);
	prev_points = next_points;
}

void bgfg_vibe::calcMotionCompensationResult(int channel) {
	int xdiff = (int)accu_xdiff;
	int ydiff = (int)accu_ydiff;

	if(xdiff != 0 || ydiff != 0) {
		for(int i=0; i<channel; i++)
			for(int j=0; j<N; j++)
				sample
		accu_xdiff -= xdiff;
		accu_ydiff -= ydiff;
	}
}