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


    BackgroundSubtractorMOG2 bg;
    std::vector<std::vector<cv::Point> > contours;
    
    int n1,n2,n3;
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
    cerr<<"Missing argument try ./gmm <PATH_DATASET> <BASE_IMAGE> <N1> <N2> <N3>"<<endl;
    cerr<<"or  try ./media/1606C02006C0032B/Documents and Settings/Agustin/Experiments_fire/4dec2013/exp4_12_13/Exp1/Fire/Cam1 rgb_12.04.13/ System1_Cam1 rgb_11- 20 47 340" <<endl;
    
    return(0);
  }
  
  
   getdir(dir,file_img);
   
     //random numbers
   srand ( time(NULL) );
   
  //sort data
   std::sort(file_img.begin(),file_img.end());
  for (unsigned int i = 0;i < file_img.size();i++) 
  {
    if(file_img[i].substr(file_img[i].find_last_of(".") + 1) == "bmp")//our version     
    {
      
    
      //read_images(dir,files_pcd,im_gray1,im_gray2, im_color1,im_color2,i);
      file_im_gray1=dir +"/"+ file_img[i];
      cout <<"Reading images..."<<file_im_gray1 <<endl;
      
      frame = imread(file_im_gray1,CV_LOAD_IMAGE_COLOR);
      
      frame.copyTo(fore);
      //back=Mat(frame.rows, frame.cols, CV_8UC1);
      back=Mat(frame.rows, frame.cols, frame.type());
      frame.copyTo(back);
      bg.operator ()(frame,fore);
      bg.getBackgroundImage(back);
      
//        cv::dilate(fore,fore,cv::Mat());
//        cv::dilate(fore,fore,cv::Mat());
      
     cv::erode(fore,fore,cv::Mat());
     /*  cv::erode(fore,fore,cv::Mat());
      cv::dilate(fore,fore,cv::Mat());
       cv::dilate(fore,fore,cv::Mat())*/;
       
      cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
      
      
     
      //draw regions
      for(unsigned int i=0;i<contours.size();i++)
      {
	  int r_rand = rand() % 255 + 1;
      int g_rand = rand() % 255 + 1;
      int b_rand = rand() % 255 + 1;
        cv::drawContours(frame,contours,i,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
	//cv::fillConvexPoly(frame,(Point *)contours[i],100,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
	//cout<<"area ->"<< cv::contourArea(contours[i]);
	int area=cv::contourArea(contours[i]);
	int perimeter=cv::arcLength(contours[i],true);
	
	cout<<"perim ->"<< perimeter<<endl;
	if (area>100)
	{
	  for(unsigned int j=0;j<contours[i].size();j++)
	  {  
	    
	    cv::circle(frame, contours[i][j], 1, cv::Scalar(r_rand, g_rand,b_rand), CV_FILLED, CV_AA);
	    // cv::fillConvexPoly();
	    
	  }
	}
	
	
      }
      
      imshow("Frame",frame);
      imshow("Background",back);
      imshow("Fore",fore);
      
      if(cv::waitKey(30) >= 0) break;
    }
  }
    
    
    return 0;
}
