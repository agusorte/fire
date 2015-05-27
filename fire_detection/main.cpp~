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
  
  
  /
  
  ///GMMM
  Mat frame; //(almost) raw frame
  Mat back; // background image
  Mat fore; // foreground mask
  Mat fore_contours; // foreground mask
  Mat image_binary;
  Mat im_bin;
  
  //COLOR segmentation
  Mat image,image_seg_rgb,image_seg_ycbcr1,image_seg_ycbcr2,image_seg_HSV,image_seg_HLS; //(almost) raw frame
  Mat ycbcr;
  Mat HSV;
  Mat HLS;
  Mat Union_color_gmm;
  
  
  BackgroundSubtractorMOG2 bg;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<std::vector<cv::Point> > contours_rgb;
  std::vector<std::vector<cv::Point> > contours_ycbcr;
  
  int n1,n2,n3;
  int per_t_1;
  int per_t;
  
  ///EROTION
  
  Mat element=getStructuringElement(cv::MORPH_ELLIPSE,Size(2,2));
  
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
    
  }
  else{
    cerr<<"Missing argument try ./fire_detection <PATH_DATASET>"<<endl;
    cerr<<"or  try ./fire_detection /home/aortega/proyectos/fire/data/fire_sequences/Securite_civile/Exp1/visible/" <<endl;
    ///media/1606C02006C0032B/Documents and Settings/Agustin/Experiments_fire/4dec2013/exp4_12_13/Exp1/Fire/Cam2 rgb_12.04.13
    return(0);
  }
  
  
  getdir(dir,file_img);
  
  //random numbers
  srand ( time(NULL) );
  
  //sort data
  std::sort(file_img.begin(),file_img.end());
  
  int At=600;
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
      
      back=Mat(frame.rows, frame.cols, frame.type());
      frame.copyTo(back);
      bg.operator ()(frame,fore);
      bg.getBackgroundImage(back);
      
      image_binary=Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      im_bin=Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      
      Mat Mat_perimtr_t=Mat(frame.rows,frame.cols,DataType<int>::type);
      Mat Mat_perimtr_t_1=Mat(frame.rows,frame.cols,DataType<int>::type);
      
      cv::dilate(fore,fore,element);
      cv::dilate(fore,fore,element);
      
      cv::erode(fore,fore,element);
      cv::erode(fore,fore,element);
      
      cv::findContours(fore,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
      
      // cv::CV_FILLED
      vector<Moments> mu(contours.size());
      vector<Point2f> mc(contours.size());
      
      vector<int> area(contours.size());
      vector<int> perimeter(contours.size());//t
      
      vector<int> rotundity(contours.size());
      
      //COLOR segmentation
      cv::cvtColor(frame,ycbcr,CV_RGB2YCrCb,0);
      cv::cvtColor(frame,HSV,CV_RGB2HSV_FULL,0);
      cv::cvtColor(frame,HLS,CV_RGB2HLS_FULL,0);
      
      image_seg_rgb   = Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_ycbcr1= Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_ycbcr2= Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HSV   = Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HLS   = Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      
      Union_color_gmm=Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      
      Mat Gradient_density_gmm=Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      
      int val_y=0;
      int val_cr=0;
      int val_cb=0;
      
      for (unsigned int ii_=0; ii_<frame.rows;ii_++)
	for (unsigned int jj_=0; jj_<frame.cols;jj_++){
	  val_y=val_y+ycbcr.at<cv::Vec3b>(ii_,jj_)[0];
	  val_cb=val_cb+ycbcr.at<cv::Vec3b>(ii_,jj_)[1];
	  val_cr=val_cr+ycbcr.at<cv::Vec3b>(ii_,jj_)[2];
	}
	
      float Y_mean=val_y/(frame.rows*frame.cols);  
      float Cb_mean=val_cb/(frame.rows*frame.cols);
      float Cr_mean=val_cr/(frame.rows*frame.cols);
      
      //cout<<"Ymean "<<Y_mean<<" Cb_mean "<<Cb_mean<<" Cr mean "<<Cr_mean<<endl;
      
      for (unsigned int ii=0; ii<frame.rows;ii++)
	for (unsigned int jj=0; jj<frame.cols;jj++)
	{
	  int r = frame.at<cv::Vec3b>(ii,jj)[2];
	  int g = frame.at<cv::Vec3b>(ii,jj)[1];
	  int b = frame.at<cv::Vec3b>(ii,jj)[0];
	  
	  int y = ycbcr.at<cv::Vec3b>(ii,jj)[0];
	  int Cb= ycbcr.at<cv::Vec3b>(ii,jj)[1];
	  int Cr= ycbcr.at<cv::Vec3b>(ii,jj)[2];
	  
	  int H = HSV.at<cv::Vec3b>(ii,jj)[0];
	  int S = HSV.at<cv::Vec3b>(ii,jj)[1];
	  int V = HSV.at<cv::Vec3b>(ii,jj)[2];
	  
	  int H2 = HLS.at<cv::Vec3b>(ii,jj)[0];
	  int I2 = HLS.at<cv::Vec3b>(ii,jj)[1];
	  int S2 = HLS.at<cv::Vec3b>(ii,jj)[2];
	  
	  ///////////////////////////////////////////////
	  /// BGR
	  //////////////////////////////////////////////
	  if((r>g) && (g>b) && (r>160)){
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  ///////////////////////////////////////////////
	  ///YCRCB
	  //////////////////////////////////////////////
	  if((y>Cb) && (Cr>Cb)){
	    image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  if((y>Y_mean) && (Cr>Cr_mean)&& (Cb<Cb_mean)){ 
	    image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  
	  if(S>=((255-r)*65/135)){
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  if(S2>=((255-r)*65/135)){
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	}
	
	///////////////////////////////////////////////////////////////////// 
	///draw regions
	//////////////////////////////////////////////////////////////////////
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
	  
	  //////////////////////////////////////////////////
	  ///compute boundibg box
	  ////////////////////////////////////////////////
	  Rect rec=cv::boundingRect(contours[i]);
	  
	  ////////////////////////////////////////////////////////
	  ///verify in all the image for each region
	  //////////////////////////////////////////////////////
	  if( area[i]>At){// area criteria
	  mu[i]=moments(contours[i],false);
	  mc[i]=Point2f(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00);
	  //cout<<"perimeter "<<perimeter[i]<<endl;
	  
	  // rotundity[i]=(4*3.14159265358979)*(area[i]/perimeter[i]);
	  
	  // cv::drawContours(frame,contours,i,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
	  //cv::drawContours(image_binary,contours,i,cv::Scalar(r_rand, g_rand,b_rand),CV_FILLED);
	  /*	 
	   *	 for(unsigned int j=0;j<contours[i].size();j++)
	   *	  cv::circle(frame, contours[i][j], 1, cv::Scalar(r_rand, g_rand,b_rand), CV_FILLED, CV_AA);*/
	  
	  for(unsigned int ii=rec.x; ii<=rec.x+rec.width;ii++){
	    for(unsigned int jj=rec.y; jj<=rec.y+rec.height;jj++){
	      int val= cv::pointPolygonTest(contours[i],Point2f(ii,jj),false);
	      if (val==1){
		Mat_perimtr_t.at<int>(jj,ii)=perimeter[i];
		// cv::circle(im_bin, Point2f(ii,jj), 1, cv::Scalar(255, 255,255), CV_FILLED, CV_AA);
		//cout<<"ii "<<ii<<" jj "<<jj<<endl;
		if(first_time){
		  per_t= perimeter[i];//Perimeter t
		  per_t_1= Mat_perimtr_t_1.at<int>(jj,ii);//perimeter t-1
		  // cout<<"Per t "<<per_t<<"Per T-1 "<<per_t_1<<endl;
		  
		  if(per_t==0)
		    cout<<"zero "<<endl;
		  
		  if ((abs(per_t-per_t_1) -abs(per_t-per_t_1)/per_t)>St ){
		    //cv::circle(im_bin, Point2f(ii,jj), 1, cv::Scalar(255, 255,255), CV_FILLED, CV_AA);
		    im_bin.at<cv::Vec3b>(jj,ii)[2]=frame.at<cv::Vec3b>(jj,ii)[2];
		    im_bin.at<cv::Vec3b>(jj,ii)[1]=frame.at<cv::Vec3b>(jj,ii)[1];
		    im_bin.at<cv::Vec3b>(jj,ii)[0]=frame.at<cv::Vec3b>(jj,ii)[0];
		  }
		}
	      }
	    }
	  }
	  }
	  
	}
	
	////////////////////////////////////////////////////
	///Intersecction regions
	///////////////////////////////////////////////////
	//////////////////////////////////////////////// 
	///RGB
	////////////////////////////////////////////////
	Mat im_bin_rgb=image_seg_rgb>0;
	Mat canny_output;
	Canny( im_bin_rgb, canny_output, 0, 255, 3 );
	findContours(canny_output,contours_rgb,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	///YCbCr
	Mat im_bin_ycbcr=image_seg_ycbcr1>0;
	Canny( im_bin_ycbcr, canny_output, 0, 255, 3 );
	findContours(canny_output,contours_rgb,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	///GMM
	Canny( im_bin, canny_output, 0, 255, 3 );
	findContours(canny_output,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	//////////////////////////////////////////////////////////
	///union regions
	/////////////////////////////////////////////////////////
	
	for (unsigned int ii=0; ii<frame.rows;ii++)
	  for (unsigned int jj=0; jj<frame.cols;jj++){
	    
	    int r_gmm = im_bin.at<cv::Vec3b>(ii,jj)[2];
	    int g_gmm = im_bin.at<cv::Vec3b>(ii,jj)[1];
	    int b_gmm = im_bin.at<cv::Vec3b>(ii,jj)[0];
	    
	    int r_rgb = image_seg_rgb.at<cv::Vec3b>(ii,jj)[2];
	    int g_rgb = image_seg_rgb.at<cv::Vec3b>(ii,jj)[1];
	    int b_rgb = image_seg_rgb.at<cv::Vec3b>(ii,jj)[0];
	    
	    int r_ycbcr = image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[2];
	    int g_ycbcr = image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[1];
	    int b_ycbcr = image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[0];
	    
	    int r_ycbcr2 = image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[2];
	    int g_ycbcr2 = image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[1];
	    int b_ycbcr2 = image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[0];
	    
	    if((r_gmm!=0 && g_gmm!=0 && b_gmm!=0) && (r_rgb!=0 && g_rgb!=0 && b_rgb!=0)
	      && (r_ycbcr!=0 && g_ycbcr!=0 && b_ycbcr!=0) && (r_ycbcr2!=0 && g_ycbcr2!=0 && b_ycbcr2!=0)){
	      
	      Union_color_gmm.at<cv::Vec3b>(ii,jj)[2] = frame.at<cv::Vec3b>(ii,jj)[2];
	      Union_color_gmm.at<cv::Vec3b>(ii,jj)[1] = frame.at<cv::Vec3b>(ii,jj)[1];
	      Union_color_gmm.at<cv::Vec3b>(ii,jj)[0] = frame.at<cv::Vec3b>(ii,jj)[0];
	      }
	      
	  }
	  ////////////////////////////////////////////////////////////////////////////
	  ///Gradient_density
	  ///////////////////////////////////////////////////////////////////////////
	  Mat im_bin_2=Union_color_gmm>0;
	  Canny( im_bin_2, canny_output, 0, 255, 3 );
	  findContours(canny_output,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	  
	  for(unsigned int i=0;i<contours.size();i++){
	     Rect rec2=cv::boundingRect(contours[i]);
	     
	     int val_sat=0;
	   for(unsigned int ii=rec2.x; ii<=rec2.x+rec2.width;ii++){
	    for(unsigned int jj=rec2.y; jj<=rec2.y+rec2.height;jj++){
	      
	      int H = HLS.at<cv::Vec3b>(jj,ii)[0];
	      int I = HLS.at<cv::Vec3b>(jj,ii)[1];
	      int S = HLS.at<cv::Vec3b>(jj,ii)[2];
	      
	      val_sat=val_sat+S;
	    }
	   }
	    int per=cv::arcLength(contours[i],true);
	    
	    if(per==0)
	     per=-10;
	    
	    float Ds=val_sat/per;
	   
	    cout<<"DS "<<Ds<<endl;
	    if(0.8<=Ds){
	      for(unsigned int ii=rec2.x; ii<=rec2.x+rec2.width;ii++){
		for(unsigned int jj=rec2.y; jj<=rec2.y+rec2.height;jj++){
		  
		  int val= cv::pointPolygonTest(contours[i],Point2f(jj,ii),false);
		  if(val==1){
		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[2] = frame.at<cv::Vec3b>(jj,ii)[2];
		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[1] = frame.at<cv::Vec3b>(jj,ii)[1];
		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[0] = frame.at<cv::Vec3b>(jj,ii)[0];
		  }
		}
	      }
	      
	    }
	    
	  }

  
  //////////////////////////////////////
  ///Save images
  //////////////////////////////////////
  
  imshow("Frame",frame);
  imshow("Image GGM",im_bin);
  
  imshow("Segmentation rgb",image_seg_rgb);
  imshow("Segmentation YCrCb",image_seg_ycbcr1);
  
  imshow("Segmentation GMM and Color",Union_color_gmm);
  
  imshow("Segmentation YCrCb2",image_seg_ycbcr2);
  
  imshow("Segmentation Saturation",image_seg_HLS);
  
  imshow("Gradient Density",Gradient_density_gmm);
  
  Mat im_bin_segmentation=Union_color_gmm>0;
   imshow("Segmentation binary",im_bin_segmentation);
  
  //imshow("HSL", image_seg_HLS);
  ///////////////////////////////////////////////
  ///Now save images
  ////////////////////////////////////////////////
  /*
   *	  string path="/home/aortega/proyectos/fire/Experiments_gmm/";
   *	  stringstream ss;
   *	  ss << i;
   *	  ///save images
   *	  imwrite(  path+"gmm_" +ss.str()+".jpg", im_bin );
   *	  imwrite(  path+"rgb_" +ss.str()+".jpg", image_seg_rgb );
   *	  imwrite(  path+"ycbcr_" +ss.str()+".jpg", image_seg_ycbcr1 );
   *          imwrite(  path+"ycbcr2_" +ss.str()+".jpg", image_seg_ycbcr2 );
   *	  imwrite(  path+"segmentation_" +ss.str()+".jpg", Union_color_gmm );*/
  //imshow("Segmentation HSV",image_seg_HSV);
  //imshow("Segmentation HLS",image_seg_HLS);
  
  string path="/home/aortega/proyectos/fire/Experiments_gmm/";
  stringstream ss;
  ss << i;
  ///save images
  imwrite(  path+"seg_gmm" +ss.str()+".jpg", im_bin_segmentation);
  
  if(cv::waitKey(10) >= 0) break;
  
  first_time=1;
  
  //Mat_perimtr_t.copyTo(Mat_perimtr_t_1);
  
}
}
    
    
    return 0;
  }
