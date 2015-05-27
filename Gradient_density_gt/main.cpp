#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
///OPENCV
#include "cxcore.h"
#include <cv.h>
#include <highgui.h>
#include "opencv2/contrib/contrib.hpp"
#include<opencv2/opencv.hpp>
using namespace std;

using namespace cv;

#include "utilities.h"
#include "evaluation_criteria.h"



int main(int argc, char **argv) {
    
  
    string dir,dir_test,file_im_gray1,base_image,base_im,base_im_seg,im,im_seg;
    vector<string> file_img = vector<string>();
    Mat frame; //(almost) raw frame
    Mat back; // background image
    Mat fore; // foreground mask
    Mat fore_contours; // foreground mask
   
  Mat image,image_seg_rgb,image_seg_ycbcr1,image_seg_ycbcr2,image_seg_HSV,image_seg_HLS; //(almost) raw frame
  Mat ycbcr;
  Mat HSV;
  Mat HLS;
  Mat Gradient_density_gmm;
  std::vector<std::vector<cv::Point> > contours;
  Mat segmentation;
  Mat frame_gray;
  int n_im;
    BackgroundSubtractorMOG2 bg;

    
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
    ///Creating files
  ofstream myfile;
  myfile.open ("/home/aortega/proyectos/fire/data/Experiments_gradient/test.txt");
  
    /// read arguments
  if (argc == 5) {
      dir = argv[1];
     base_im = argv[2];
     base_im_seg = argv[3];
     n_im=atoi(argv[4]);;
  }
  else{
    cerr<<"Missing argument try ./gmm <PATH_DATASET> <BASE_IMAGE> <N1> <N2> <N3>"<<endl;
    cerr<<"or  try ./gradiente_gt /home/aortega/proyectos/fire/data/GroundTrouth/Labeled_sequences/Aerodrome1/ SC1_rgb.bmp SC1_gt.bmp 32" <<endl;
    
    return(0);
  }
  
  
   getdir(dir,file_img);
   
     //random numbers
   srand ( time(NULL) );
    Mat element=getStructuringElement(cv::MORPH_ELLIPSE,Size(2,2));
  //sort data
   std::sort(file_img.begin(),file_img.end());
   double mean1=0;
 double mean2=0;
 double mean3=0;
  for (unsigned int i = 1;i <=n_im;i++){
     stringstream ss;
     ss << i;
    if(i<10){
       im=dir+"00"+ss.str()+base_im;
       im_seg=dir+"00"+ss.str()+base_im_seg;
     }else if(10<=i && i<100){
       im=dir+"0"+ss.str()+base_im;
       im_seg=dir+"0"+ss.str()+base_im_seg;
     }else{
       im=dir+ss.str()+base_im;
       im_seg=dir+ss.str()+base_im_seg;
     }
     
//       if(i<10){
//        im=dir+"0"+ss.str()+base_im;
//        im_seg=dir+"0"+ss.str()+base_im_seg;
//      }else if(10<=i && i<100){
//        im=dir+ss.str()+base_im;
//        im_seg=dir+ss.str()+base_im_seg;
//      }else{
//        im=dir+ss.str()+base_im;
//        im_seg=dir+ss.str()+base_im_seg;
//      }
      
    
      //read_images(dir,files_pcd,im_gray1,im_gray2, im_color1,im_color2,i);
      cout <<"Reading images..."<<im <<endl<<endl;
      cout <<"Reading images..."<<im_seg <<endl<<endl;
     /// Read image base
      image = imread(im,CV_LOAD_IMAGE_COLOR);
      segmentation=imread(im_seg,CV_LOAD_IMAGE_GRAYSCALE);
    

      cv::cvtColor(image,HSV,CV_RGB2HSV_FULL,0);
      cv::cvtColor(image,HLS,CV_RGB2HLS_FULL,0);
      
      image_seg_rgb=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
       Gradient_density_gmm=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HSV=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HLS=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
    
      int st=55;
      int rt=115;
      for (unsigned int ii=0; ii<image.rows;ii++){
	for (unsigned int jj=0; jj<image.cols;jj++){
	  int r = image.at<cv::Vec3b>(ii,jj)[2];
	  int g=  image.at<cv::Vec3b>(ii,jj)[1];
	  int  b= image.at<cv::Vec3b>(ii,jj)[0];
	  
	 // cout<<"R "<<r<<" G "<<g<<" B "<<b<<endl;
	  int H = HSV.at<cv::Vec3b>(ii,jj)[0];
	  int S=  HSV.at<cv::Vec3b>(ii,jj)[1];
	  int V= HSV.at<cv::Vec3b>(ii,jj)[2];
	  
	  int H2 = HLS.at<cv::Vec3b>(ii,jj)[0];
	  int I2=  HLS.at<cv::Vec3b>(ii,jj)[1];
	  int S2= HLS.at<cv::Vec3b>(ii,jj)[2];
	  
	  
	  if((r>g) && (g>b) && (r>rt)){
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  
	  
	  
	  if((S>=(((255-r)*st)/rt)) && (r>g) && (g>b) && (r>rt))
	  {
	    
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	    if(S2>=(((255-r)*st)/rt) && (r>g) && (g>b) && (r>rt))
	  {
	    
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	}
	
      }
       //gradient 
	 Mat im_bin_2=image_seg_HSV>0;
	 Mat canny_output;
	 Canny( image_seg_HSV, canny_output, 100, 255, 3 );
	 
	 cv::dilate(canny_output,canny_output,element);
	 //cv::erode(canny_output,canny_output,element);
	 
	 findContours(canny_output,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	 
	  cout<<"Regions Number ->"<<contours.size()<<endl;
	  
	 for(unsigned int i_=0;i_<contours.size();i_++){
	     Rect rec2=cv::boundingRect(contours[i_]);     
	     double val_sat=0;
	      for(unsigned int j_=0;j_<contours[i_].size();j_++){
		Point2f po=contours[i_][j_];
		int H = image_seg_HSV.at<cv::Vec3b>(po.y,po.x)[0];
		double S = image_seg_HSV.at<cv::Vec3b>(po.y,po.x)[1];
		int V = image_seg_HSV.at<cv::Vec3b>(po.y,po.x)[2];
		val_sat=val_sat+S;
		 //cout<<"H "<<H<<" S "<<S<<" V "<<V<<endl;
		//int H = canny_output.at<cv::Vec3b>(p,ii)[0];
		 cv::circle(Gradient_density_gmm, contours[i_][j_], 1, cv::Scalar(255, 0,0), CV_FILLED, CV_AA);
	      }

	    double per=cv::arcLength(contours[i_],true);
	    
	    if(per==0)
	     per=-10;
	    
	    double Ds=(val_sat)/per;

	    cout<<"DS "<<Ds<<" Val "<<val_sat<<" per "<<per<<endl;
	    
	 if(0.8<=Ds){
	      for(unsigned int ii=rec2.x; ii<=rec2.x+rec2.width;ii++){
		for(unsigned int jj=rec2.y; jj<=rec2.y+rec2.height;jj++){
		  int val= cv::pointPolygonTest(contours[i_],Point2f(ii,jj),false);
		  if(val==1){
 		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[2] = image.at<cv::Vec3b>(jj,ii)[2];
 		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[1] = image.at<cv::Vec3b>(jj,ii)[1];
 		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[0] = image.at<cv::Vec3b>(jj,ii)[0];

		    
		  }
		}
	      }
	      
	   }
	    
	  }
	  
	  cvtColor(Gradient_density_gmm,frame_gray,CV_RGB2GRAY,0);
	  double f1score_result=F1score(segmentation,frame_gray);
	  double MCC_result=MCC(segmentation,frame_gray);  
	  double hafiane_result=Hafiane(segmentation,frame_gray);
	  stringstream ss_f1score,ss_Mcc,ss_hafiane;
	  
	  mean1=mean1+f1score_result;
	  mean2=mean2+MCC_result;
	  mean3=mean3+hafiane_result;
	  
	  ss_f1score << f1score_result;
	  ss_Mcc << MCC_result;
	  ss_hafiane  << hafiane_result;
	  // write data
	  
	  string input;
	  
	  input=ss_f1score.str()+" "+ss_Mcc.str()+" "+ss_hafiane.str()+"\n";
	  
	  myfile<<input;
	  
	  imshow("frame ",image);
	  imshow("Segmentation rgb",image_seg_rgb);
	  
	  imshow("Segmentation HSV",image_seg_HSV);
	  imshow("Segmentation HLS",image_seg_HLS);
	  
	  
	  imshow("Gradiente ",Gradient_density_gmm);
	  imshow("Canny ",canny_output);
	  
	  if(cv::waitKey(1000) >= 0) break;
	
	  string path="/home/aortega/proyectos/fire/data/Experiments_gradient/";
	stringstream sss;
	sss << i;
	///save images
	imwrite(  path+"seg_gradiente_" +sss.str()+".jpg", image_seg_rgb);  
	
  }
  
  
  string input2;
  stringstream ss_f1score2,ss_Mcc2,ss_hafiane2;
  ss_f1score2 << mean1/n_im;
  ss_Mcc2 << mean2/n_im;
  ss_hafiane2  << mean3/n_im;
  input2="Mean "+ss_f1score2.str()+" "+ss_Mcc2.str()+" "+ss_hafiane2.str()+"\n";
  
  myfile<<input2;
  
  myfile.close();
  
    
    return 0;
}
