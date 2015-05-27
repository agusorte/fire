#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>

///OPENCV
#include "cxcore.h"
#include <cv.h>
#include <highgui.h>
#include "opencv2/contrib/contrib.hpp"
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace std;

using namespace cv;


int getdir (string dir, vector<string> &files)
{
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening " << dir << endl;
    return errno;
  }
  
  while ((dirp = readdir(dp)) != NULL) {
    files.push_back(string(dirp->d_name));
  }
  closedir(dp);
  
  return 0;
}


int main(int argc, char **argv) {
    
  
    string dir,file_im_gray1,base_image;
    vector<string> file_img = vector<string>();
    Mat frame; //(almost) raw frame
    Mat back; // background image
    Mat fore; // foreground mask
    Mat fore_contours; // foreground mask
    Mat image_binary;
    Mat im_bin;

    BackgroundSubtractorMOG2 bg;
    std::vector<std::vector<cv::Point> > contours;
    
    int n1,n2,n3;
    int per_t_1;
    int per_t;
    // bg.initialize();
  // bg.set('nmixtures', 3); 
   
    
//     bg.initialize();
//     initialize 	( 	Size  	frameSize,
// 		double  	alphaT,
// 		double  	sigma = 15,
// 		int  	nmixtures = 5,
// 		bool  	postFiltering = false,
// 		double  	minArea = 15,
// 		bool  	detectShadows = true,
// 		bool  	removeForeground = false,
// 		double  	Tb = 16,
// 		double  	Tg = 9,
// 		double  	TB = 0.9,
// 		double  	CT = 0.05,
// 		int  	nShadowDetection = 127,
// 		double  	tau = 0.5 
// 	)
   // bg.nmixtures = 3;
  //  bg.bShadowDetection = true;
    
    /// read arguments
  if (argc == 2) {
    dir = argv[1];
//     base_image = argv[2];
//     n1=atoi (argv[3]);
//     n2=atoi (argv[4]);
//     n3=atoi (argv[5]);
  }
  else{
    cerr<<"Missing argument try ./gmm <PATH_DATASET>"<<endl;
    cerr<<"or  try ./gmm /home/aortega/proyectos/fire/data/fire_sequences/Securite_civile/Exp1/visible/" <<endl;
    
    return(0);
  }
  
  
   getdir(dir,file_img);
   
     //random numbers
   srand ( time(NULL) );
   
  //sort data
   std::sort(file_img.begin(),file_img.end());
   
   int At=400;
   int St=1;
   int first_time=0;
  for (unsigned int i = 0;i < file_img.size();i++) 
  {
    if(file_img[i].substr(file_img[i].find_last_of(".") + 1) == "bmp" || 
      file_img[i].substr(file_img[i].find_last_of(".") + 1) == "ppm")//our version     
    {
      
    
      //read_images(dir,files_pcd,im_gray1,im_gray2, im_color1,im_color2,i);
      file_im_gray1=dir +"/"+ file_img[i];
      cout <<"Reading images..."<<file_im_gray1 <<endl<<endl;
      
      frame = imread(file_im_gray1,CV_LOAD_IMAGE_COLOR);
      
      frame.copyTo(fore);
      //back=Mat(frame.rows, frame.cols, CV_8UC1);
      back=Mat(frame.rows, frame.cols, frame.type());
      frame.copyTo(back);
      bg.operator ()(frame,fore);
      bg.getBackgroundImage(back);
      
      image_binary=Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      im_bin=Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));

      
      Mat Mat_perimtr_t=Mat(frame.rows,frame.cols,DataType<int>::type);
      Mat Mat_perimtr_t_1=Mat(frame.rows,frame.cols,DataType<int>::type);
      cv::erode(fore,fore,cv::Mat());
      
      
      cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
      
      
      vector<Moments> mu(contours.size());
      vector<Point2f> mc(contours.size());
      
      vector<int> area(contours.size());
      vector<int> perimeter(contours.size());//t
    
      vector<int> rotundity(contours.size());

      //draw regions
      
     for(unsigned int i=0;i<contours.size();i++)
     {
       int r_rand = rand() % 255 + 1;
       int g_rand = rand() % 255 + 1;
       int b_rand = rand() % 255 + 1;
      // cv::drawContours(frame,contours,i,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
       //cv::fillConvexPoly(frame,(Point *)contours[i],100,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
       //cout<<"area ->"<< cv::contourArea(contours[i]);
       area[i]=cv::contourArea(contours[i]);
       perimeter[i]=cv::arcLength(contours[i],true);
       mu[i]=moments(contours[i],false);
       mc[i]=Point2f(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
      // rotundity[i]=(4*3.14159265358979)*(area[i]/perimeter[i]);
       ////////////////////////////////////////////////////////
       //verify in all the image for each region
       //////////////////////////////////////////////////////
       if( area[i]>At)
       {
	 cv::drawContours(frame,contours,i,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
	 cv::drawContours(image_binary,contours,i,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
	 cv::circle(frame, mc[i], 1, cv::Scalar(0, 255,222), CV_FILLED, CV_AA);
	 for(unsigned int j=0;j<contours[i].size();j++)
	   cv::circle(frame, contours[i][j], 1, cv::Scalar(r_rand, g_rand,b_rand), CV_FILLED, CV_AA);
	
	   
	   for(unsigned int ii=0; ii<=frame.rows;ii++)
	     for(unsigned int jj=0; jj<=frame.cols;jj++)
	     {
	       int val= cv::pointPolygonTest(contours[i],Point2f(ii,jj),false);
	       //cout<<"val"<< val<<endl;
	       
	       if (val==1)
	       {
		 
		 Mat_perimtr_t.at<int>(ii,jj)=perimeter[i];
		 
		 if(first_time)
		 {
		   per_t= perimeter[i];
		   //cout<<"per_t "<<per_t<<endl;
		   per_t_1= Mat_perimtr_t_1.at<int>(ii,jj);
		   
		   
		   // cout<<"per_t_1 "<<per_t_1<<endl;

		   if ((abs(per_t-per_t_1) -abs(per_t-per_t_1)/per_t)>St )
		   {
		     cv::circle(im_bin, Point2f(ii,jj), 1, cv::Scalar(255, 255,255), CV_FILLED, CV_AA);
		   }
		 }
	       }
	     }
	 }
       //////////////////////////////////////////////////////////////////////
       //conditions area
       //////////////////////////////////////////////////////////////////////
//        if (area[i]>300)
//        {
// 	 cv::drawContours(frame,contours,i,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
// 	 cv::circle(frame, mc[i], 1, cv::Scalar(0, 255,222), CV_FILLED, CV_AA);
// 	 
// 	 for(unsigned int j=0;j<contours[i].size();j++)
// 	 {  
// 	   cv::circle(frame, contours[i][j], 1, cv::Scalar(r_rand, g_rand,b_rand), CV_FILLED, CV_AA);
// 
// 	 }
//        }
       
       
     }
     
     cvtColor(image_binary,image_binary,CV_RGB2GRAY);
     
     Mat bin_=image_binary>0;
     imshow("Frame",frame);
     imshow("Fore",fore);
     imshow("Image binary",bin_);
     imshow("Image binary algortihm",im_bin);
     
     if(cv::waitKey(900) >= 0) break;
     
     first_time=1;
    
     Mat_perimtr_t.copyTo(Mat_perimtr_t_1);
     
    }
  }
  
    
    return 0;
}
