#include "evaluation_criteria.h"

double MCC(Mat Im_ref,Mat Im_seg)
{
  Mat Im_ref_bin,Im_seg_bin;
  
  Im_ref_bin=Im_ref>0;
  Im_seg_bin=Im_seg>0;
  
 float TP,TN,FP,FN;
 
  TP=0;
  TN=0;
  FP=0;
  FN=0;
  
  float sum_ref=0;
  float sum_seg=0;
  
  float val_ref;
  float val_seg;
  
  for(unsigned int i=0; i<=Im_ref_bin.rows;i++){
    for(unsigned int j=0; j<=Im_ref_bin.cols;j++){
      
      if(Im_ref_bin.at<uchar>(i,j)==255)
	val_ref=1;
      else
	val_ref=0;
      
      if(Im_seg_bin.at<uchar>(i,j)==255)
	val_seg=1;
      else
	val_seg=0;
      
      TP=(val_ref*val_seg)+TP;
      TN= ((1-val_ref)*(1-val_seg))+TN;
      FP = ((1-val_ref)*(val_seg))+FP;
      FN =((val_ref)*(1-val_seg))+FN;
      
     sum_ref=val_ref+sum_ref;
     sum_seg=val_seg+sum_seg;
      
    }
    
  } 
  if((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP)==0){
    if(sum_ref==sum_seg)
      return 1;
    else
      return 0;
    
    
  }
  else{
    return (TP*TN-FP*FN)/(sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP)));
  }
  

  
}

double F1score(Mat Im_ref,Mat Im_seg)
{
  
  Mat Im_ref_bin,Im_seg_bin;
  
  Im_ref_bin=Im_ref>0;
  Im_seg_bin=Im_seg>0;
  
  float VP,FP,FN;

  VP=0;
  FP=0;
  FN=0;
  float sum_ref=0;
  float sum_seg=0;
  
  float val_ref;
  float val_seg;
  
  for(unsigned int i=0; i<=Im_ref_bin.rows;i++){
    for(unsigned int j=0; j<=Im_ref_bin.cols;j++){
      
      if(Im_ref_bin.at<uchar>(i,j)==255)
	val_ref=1;
      else
	val_ref=0;
      
      if(Im_seg_bin.at<uchar>(i,j)==255)
	val_seg=1;
      else
	val_seg=0;
      
      VP=(val_ref*val_seg)+VP;
     
      FP = (1-val_ref)*(val_seg)+FP;
      FN =(val_ref)*(1-val_seg)+FN;
      
     sum_ref=val_ref+sum_ref;
     sum_seg=val_seg+sum_seg;
      
    }
    
  } 
  
  
  float Pr = VP/(VP+FP);
  float Ra= VP/(VP+FN);

  if ((Pr+Ra)>0){
   return 2*Pr*Ra/(Pr+Ra);
  }
  else{
    if(sum_ref==sum_seg)
      return 1;
    else
      return 0;
    
  }
    
  
}

double Hafiane(Mat Im_ref,Mat Im_seg){
  
  
  Mat Im_ref_bin,Im_seg_bin;
  
  Im_ref_bin=Im_ref>128;
  Im_seg_bin=Im_seg>128;
  
  Mat canny_output;
  float sum_ref=0;
  float sum_seg=0;
  
  std::vector<std::vector<cv::Point> > contours_ref;
  std::vector<std::vector<cv::Point> > contours_seg;
  
  Canny( Im_ref_bin, canny_output, 128, 255, 3 );
  cv::dilate(canny_output, canny_output, cv::Mat(), cv::Point(-1,-1));
  findContours(canny_output,contours_ref,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//   imshow("canny ref",canny_output);

  
  Canny( Im_seg_bin, canny_output, 128, 255, 3 );
  cv::dilate(canny_output, canny_output, cv::Mat(), cv::Point(-1,-1));
  findContours(canny_output,contours_seg,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  
  
//   imshow("canny seg",canny_output);
//   
//   cv::waitKey(0);
  
  int Num1=contours_ref.size();
  int Num2 =contours_seg.size();
  float m = 0.5;
  float n;
  float tailletot=0;
  
  if (Num2>=Num1)
    float n = Num1/Num2;
  else
    float n = log(1+Num2/Num1);
  
  double M1 = 0;
  
  float val_ref;
  float val_seg;
  
  for(unsigned int i=0; i<=Im_ref_bin.rows;i++){
    for(unsigned int j=0; j<=Im_ref_bin.cols;j++){
      
      if(Im_ref_bin.at<uchar>(i,j)==255)
	val_ref=1;
      else
	val_ref=0;
      
      if(Im_seg_bin.at<uchar>(i,j)==255)
	val_seg=1;
      else
	val_seg=0;
      
      tailletot=tailletot+val_seg;
      sum_ref=val_ref+sum_ref;
      sum_seg=val_seg+sum_seg;
      
    }
  }
  
  int taille1=0;
  
  int sum_region_seg=0;
  int sum_region_ref_intect=0;
  float sum_max_region=0;
  float sum_max_region_aux=0;
  
  for(unsigned int i=0;i<contours_seg.size();i++){
    
    Rect rec=cv::boundingRect(contours_seg[i]);
    
    double area_seg = contourArea(contours_seg[i]);
   
  //  cout<<" area "<<area_seg<<endl;
    
    for(unsigned int ii=rec.x; ii<=rec.x+rec.width;ii++){
      for(unsigned int jj=rec.y; jj<=rec.y+rec.height;jj++){
	int val= cv::pointPolygonTest(contours_seg[i],Point2f(ii,jj),false);
	
	if (val==1){
	  
	  if( Im_ref_bin.at<uchar>(jj,ii)==255){
	    taille1=taille1+1;
	  }
	  if( Im_seg_bin.at<uchar>(jj,ii)==255){
	    sum_region_seg=sum_region_seg+1;
	  }
	  
	}
      }
    }
    
    
    double p=taille1/tailletot;  
    
    M1=M1+ ((p* taille1)/(sum_ref+taille1));
    
  }
  
  float HAF = (M1+n*m)/(1+m);
  if (isnan(HAF)){
    if (sum_ref == sum_seg)
      HAF = 1;
    else
      HAF = 0;
    
  }
  
  return HAF;
	  
}
