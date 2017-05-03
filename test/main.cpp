#include <opencv/cv.h>
//#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <time.h>
#include "cv_yuv_codebook.h"

//#include "cv_yuv_codebook.h"
/*
This program will take the video from a webcam and extract the foreground
by using an average model 
*/

//Variables used with the trackbars
int g_open=0;
int g_hist=1;
int g_metod=0;
int g_Scale=10;
int g_nFrame=30;
int g_perim=4;
int g_pause=0;

//Global images
//Float, 3 channel
IplImage *IavgF,*IdiffF, *IprevF, *IhiF, *IlowF;
IplImage *Iscratch,*Iscratch2;

//Float, 1 channel
IplImage *Igray1,*Igray2,*Igray3;
IplImage *Ilow1,*Ilow2,*Ilow3,*Ihi1,*Ihi2,*Ihi3;

//Byte, 1 channel
IplImage *Imaskt, *Imask1, *Imask2, *Imask3;

//Byte, 3 channel
IplImage *bigWindowImageB, *bigWindowImageB1;
IplImage* originalFrame, *finalImage;

//Other variables
float Icount;	//Counts the number of images learned for averaging
int width,height;

clock_t start,end;
double cpu_time_used;







void trackbarFunction( int) {}		//We will use this function when we only need the value of the trackbar

void trackbarHistoric(int) 
{
	if (g_hist == 1) 
	{
		Icount = 0.00001;
		cvZero(IavgF);
		cvZero(IdiffF);
		cvZero(IprevF);
		cvZero(IhiF );
		cvZero(IlowF  );
	}
}

void trackbarColor(int)
{
	cvSetTrackbarPos("Historic","Controls",1);
}




/*
This function creates the structure of images we need
I is a sample image for sizing
*/
void AllocateImages(IplImage* I)
{
	CvSize sz = cvGetSize(I);
	IavgF =		cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IdiffF =	cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IprevF =	cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IhiF =		cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	IlowF =		cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	Ilow1 =		cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ilow2 =		cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ilow3 =		cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ihi1 =		cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ihi2 =		cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Ihi3 =		cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	cvZero(IavgF  );
	cvZero(IdiffF  );
	cvZero(IprevF  );
	cvZero(IhiF );
	cvZero(IlowF  );		
	Icount = 0.00001; //Protect against divide by zero
	Iscratch =	cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	Iscratch2 = cvCreateImage( sz, IPL_DEPTH_32F, 3 );
	Igray1 =	cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Igray2 =	cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	Igray3 =	cvCreateImage( sz, IPL_DEPTH_32F, 1 );
	finalImage = cvCreateImage( sz, IPL_DEPTH_8U, 3 );
	Imaskt =	cvCreateImage( sz, IPL_DEPTH_8U, 3 );
	Imask1 =	cvCreateImage( sz, IPL_DEPTH_8U, 1 );
	Imask2 =	cvCreateImage( sz, IPL_DEPTH_8U, 1 );
	Imask3 =	cvCreateImage( sz, IPL_DEPTH_8U, 1 );

	bigWindowImageB =	cvCreateImage(cvSize(1.5*width,1.5*height),IPL_DEPTH_8U,3);
	bigWindowImageB1 =	cvCreateImage(cvSize(1.5*width,1.5*height),IPL_DEPTH_8U,1);
}

/*
This function releases the space used by the images
*/
void DeallocateImages()
{
	
	cvReleaseImage(&IavgF);
	cvReleaseImage(&IdiffF );
	cvReleaseImage(&IprevF );
	cvReleaseImage(&IhiF );
	cvReleaseImage(&IlowF );
	cvReleaseImage(&Ilow1  );
	cvReleaseImage(&Ilow2  );
	cvReleaseImage(&Ilow3  );
	cvReleaseImage(&Ihi1   );
	cvReleaseImage(&Ihi2   );
	cvReleaseImage(&Ihi3  );
	
	cvReleaseImage(&Iscratch);
	cvReleaseImage(&Iscratch2);

	cvReleaseImage(&Igray1  );
	cvReleaseImage(&Igray2 );
	cvReleaseImage(&Igray3 );

	cvReleaseImage(&Imaskt);
	cvReleaseImage(&Imask1);
	cvReleaseImage(&Imask2);		
	cvReleaseImage(&Imask3);

	cvReleaseImage(&bigWindowImageB);
	cvReleaseImage(&bigWindowImageB1);
	cvReleaseImage(&finalImage);
}

// Accumulate the background statistics for one more frame
// We accumulate the images, the image differences and the count of images for the 
//    the routine createModelsfromStats() to work on after we're done accumulating N frames.
// I		Background image, 3 channel, 8u
void accumulateBackground(IplImage *I)
{
	static int first = 1;
	cvCvtScale(I,Iscratch,1,0); //To float;
	switch(g_metod)
	{
		case 1:	cvCvtColor( Iscratch, Iscratch, CV_BGR2HSV ); break;		//We convert the image color from rgb to hsv 
		case 2:	cvCvtColor( Iscratch, Iscratch, CV_BGR2YCrCb ); break;	
		case 3:	cvCvtColor( Iscratch, Iscratch, CV_BGR2HLS ); break;	
		case 4:	cvCvtColor( Iscratch, Iscratch, CV_BGR2Lab ); break;	
		case 5:	cvCvtColor( Iscratch, Iscratch, CV_BGR2Luv ); break;	
		default: break;	
	}
	if (!first){
		cvAcc(Iscratch,IavgF);	//We add the image to the sum of all the images
		cvAbsDiff(Iscratch,IprevF,Iscratch2);	//We make the absolute diference
		cvAcc(Iscratch2,IdiffF);	//Then we add the diference to the sum of all the differences
		Icount += 1.0;
	}
	first = 0;
	cvCopy(Iscratch,IprevF);
}

// Scale the average difference from the average image high acceptance threshold
void scaleHigh(float scale)
{
	cvConvertScale(IdiffF,Iscratch,scale); //Converts with rounding and saturation
	cvAdd(Iscratch,IavgF,IhiF);
	cvSplit( IhiF, Ihi1,Ihi2,Ihi3, 0 );
}

// Scale the average difference from the average image low acceptance threshold
void scaleLow(float scale)
{
	cvConvertScale(IdiffF,Iscratch,scale); //Converts with rounding and saturation
	cvSub(IavgF,Iscratch,IlowF);
	cvSplit( IlowF, Ilow1,Ilow2,Ilow3, 0 );
}

//Once you've learned the background long enough, turn it into a background model
void createModelsfromStats()
{
	cvConvertScale(IavgF,IavgF,(double)(1.0/Icount));
	cvConvertScale(IdiffF,IdiffF,(double)(1.0/Icount));
	cvAddS(IdiffF,cvScalar(1.0,1.0,1.0),IdiffF);  //Make sure diff is always something
	scaleHigh(g_Scale);
	scaleLow(g_Scale);
}

// Create a binary: 0,255 mask where 255 means foreground pixel
// I		Input image, 3 channel, 8u
// Imask	mask image to be created, 1 channel 8u
//
void backgroundDiff(IplImage *I)  //Mask should be grayscale
{
	cvCvtScale(I,Iscratch,1,0); //To float;
	switch(g_metod)
	{
		case 1:	cvCvtColor( Iscratch, Iscratch, CV_BGR2HSV ); break;		//We convert the image color from rgb to hsv 
		case 2:	cvCvtColor( Iscratch, Iscratch, CV_BGR2YCrCb ); break;	
		case 3:	cvCvtColor( Iscratch, Iscratch, CV_BGR2HLS ); break;	
		case 4:	cvCvtColor( Iscratch, Iscratch, CV_BGR2Lab ); break;	
		case 5:	cvCvtColor( Iscratch, Iscratch, CV_BGR2Luv ); break;	
		default: break;	
	}
	
	cvSplit( Iscratch, Igray1,Igray2,Igray3, 0 );
	//Channel 1
	cvInRange(Igray1,Ilow1,Ihi1,Imask1);
	cvSubRS( Imask1, cvScalar(255), Imask1);
	//Channel 2
	cvInRange(Igray2,Ilow2,Ihi2,Imask2);
	cvSubRS( Imask2, cvScalar(255), Imask2);
	//Channel 3
	cvInRange(Igray3,Ilow3,Ihi3,Imask3);
	cvSubRS( Imask3, cvScalar(255), Imask3);	
	
	//Combine all the channels
	cvOr(Imask1,Imask2,Imask1);
	cvOr(Imask1,Imask3,Imask1);

	//Let's clean the image
	int num=100;
	CvRect rectangles[100];
	cvconnectedComponents(Imask1,1,g_perim,&num,rectangles,NULL);
	std::cout << "Contornos hallados " << num << std::endl;

	//Now we show the foreground
	cvResize(Imask1, bigWindowImageB1); cvShowImage( "Foreground Chanel" , bigWindowImageB1);

	//Now we make an AND 

	cvCopy(originalFrame,finalImage);
	for (int i=0; i<num; i++)
	{
		cvRectangle(finalImage,
			cvPoint(rectangles[i].x,rectangles[i].y),
			cvPoint((rectangles[i].x+rectangles[i].width),(rectangles[i].y+rectangles[i].height)),
			cvScalar(0,255,255),
			3
			);
	}

	cvResize(finalImage, bigWindowImageB); cvShowImage( "Processed Image" , bigWindowImageB);
}

void trackbarThreshold(int)
{
	scaleHigh(g_Scale);
	scaleLow(g_Scale);
}

int main( int argc, char** argv)
{
	//First we get the video
	CvCapture* capture = cvCreateFileCapture("test2.mp4");		//CvCapture is the structure used by openCv to handle with video
	
	/*
	if (argc == 1)		//If we have provided an argument we will load a video, if not we will take the image from a webcam
        capture = cvCreateCameraCapture(-1);
	else
		capture = cvCreateFileCapture(argv [1]);
	*/
	
	if (!capture) return -1;	//Check that everything is ok

	double fps = cvGetCaptureProperty (
        capture,
        CV_CAP_PROP_FPS
    );	
	if (fps == 0) fps=15; //Sometimes the webcam gives 0 fps, but the real value is 15 fps

	//Second we organize the windows in the desktop	
	originalFrame = cvQueryFrame (capture);

	width = originalFrame->width;
	height = originalFrame->height;
	int widthMax = 400;		//You cand adjust this values to fit better in your computer
	int heightMax = 400;		
	if (width>height)
	{
		height = height * widthMax / width ;
		width = widthMax; 
	}
	else
	{
		width = width * heightMax / height;
		height = heightMax;
	}

	cvNamedWindow(	"Foreground Chanel",CV_WINDOW_AUTOSIZE);

	cvNamedWindow("Processed Image",CV_WINDOW_AUTOSIZE);

	cvNamedWindow("Controls",0);
	cvCreateTrackbar( "Scale",  "Controls", &g_Scale, 50, trackbarThreshold);
	cvCreateTrackbar( "Open",  "Controls", &g_open, 5, trackbarFunction);
	cvCreateTrackbar( "Color", "Controls", &g_metod, 5, trackbarColor);
	cvCreateTrackbar( "N Frames", "Controls", &g_nFrame, 100, trackbarFunction);
	cvCreateTrackbar( "Historic", "Controls", &g_hist, 1, trackbarHistoric);
	cvCreateTrackbar( "Perim", "Controls", &g_perim, 10, trackbarFunction);
	cvCreateTrackbar( "Pause", "Controls", &g_pause, 1, trackbarFunction);
	cvResizeWindow("Controls", 1.5*width, 350);

	cvMoveWindow("Processed Image", 1300,20);
	cvMoveWindow("Foreground Chanel", 1320+1.5*width,20);
	cvMoveWindow("Controls", 1300, 80+1.5*height);

	//Now we have all the windows well organized and we can proceed with the program
	AllocateImages(originalFrame);

	while(1) 
	{
		start=clock();	//Start counting time
		if (g_pause == 0)
			originalFrame = cvQueryFrame (capture);
		if (!originalFrame) break;		//We get the next frame and exit if it is the end

		if (g_hist == 1)	//We are creating the model
		{
			cvResize(originalFrame, bigWindowImageB); cvShowImage( "Processed Image" , bigWindowImageB);
			accumulateBackground(originalFrame);
			if (Icount > g_nFrame)
			{
				createModelsfromStats();
				cvSetTrackbarPos("Historic","Controls",0);
			}
		}
		else	//We are evaluating the model
		{
			backgroundDiff(originalFrame);
		}

		end=clock();
		int tiempoEspera;
		if (fps != 15)
		{
			cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC*1000;
			int milis = int (cpu_time_used);
			printf("Tiempo invertido = %d  ms",milis);
			int tiempoFrame = int (1000/fps);
			printf("    Tiempo frame = %d  ms\n",tiempoFrame);
			if (tiempoFrame>=milis)	tiempoEspera = tiempoFrame-milis;
			else
			{
				originalFrame = cvQueryFrame (capture);
				tiempoEspera=milis-tiempoFrame;
				printf("\n");
				while(tiempoEspera>tiempoFrame)
				{
					originalFrame = cvQueryFrame (capture);
					tiempoEspera = tiempoEspera-tiempoFrame;
					printf("\n");
				}
			}
			if (tiempoEspera == 0) tiempoEspera = 1;
		}
		else tiempoEspera = 10;
		if (cvWaitKey(tiempoEspera) == 27) break;		//We wait 25ms and if we press ESC the program will exit
	}
	
	DeallocateImages();
	cvReleaseCapture (&capture);
	cvDestroyWindow("Original");
	cvDestroyWindow("Chanel 1");
	cvDestroyWindow("Chanel 2");
	cvDestroyWindow("Chanel 3");
	cvDestroyWindow("Foreground Chanel 1");
	cvDestroyWindow("Foreground Chanel 2");
	cvDestroyWindow("Foreground Chanel 3");
	cvDestroyWindow("Processed Image");	
	cvDestroyWindow("Controls");	
	
}