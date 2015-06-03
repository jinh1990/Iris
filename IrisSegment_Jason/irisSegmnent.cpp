#include "stdio.h"
#include "iostream"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>

#include <ctime>

#include "my_image_utils.h"
#include "my_iris_segmentation.h"

#include "fit_curves.h"



using namespace cv;
using namespace std;

void removeSmallBlobs(cv::Mat& im, double size);
void cal_canny_high_thresd(Mat src, float ratio, int& high_thresd);

void main()
{
	Mat img = imread("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\010.jpg",1);
	//Mat img = imread("C:\\Users\\pebble\\Desktop\\66_Jun_Liu_iris_15_04_22_04_49_49.jpg",1);

	int rows = img.rows;
	int cols = img.cols;

	int function_volue = 0;

	//Mat img = img1(Rect(Point(cols/4,rows/4),Point(cols*3/4,rows*3/4))).clone();

 	if (img.type() != CV_8UC1)
 	{
 		cvtColor(img,img,CV_BGR2GRAY);
 	}
	
	time_t cur_time, last_time;
	last_time = clock();

	//resize(img,img,Size(200,150));

	Mat img_enhanced;
	//equalizeHist(img,img);

 	medianBlur(img,img_enhanced,7);

	//int high_threshold = 0;
	//int low_threshold = 0;

 //	calc_threshold(img_enhanced,0,0.9,low_threshold,high_threshold);

	//Mat bw_img, bw_big;
	//threshold(img,bw_img,high_threshold,255,0);

	//bw_big = bw_img.clone();

	//removeSmallBlobs(bw_big,1000);

	//for(int i = 0; i < bw_img.rows; i++)
	//	for(int j = 0; j < bw_img.cols; j++)
	//	{
	//		if (bw_big.at<unsigned char>(i,j) != 0)
	//		{
	//			bw_img.at<unsigned char>(i,j) = 0;
	//		}
	//	}

	//int dilation_type = MORPH_ELLIPSE;
	//int dilation_size = 15;

	//Mat element = getStructuringElement( MORPH_ELLIPSE, Size(dilation_size,dilation_size));

 // /// Apply the dilation operation
	//Mat dil_img;
	//dilate( bw_img, dil_img, element );

	GaussianBlur(img_enhanced,img_enhanced,Size(7,7),0);

	//Mat img_smooth, edgemap;
	//get_rtv_l1_contour(img_enhanced,edgemap,img_smooth);

	//cvNamedWindow( "better_result", 1 );
	//cvNamedWindow( "img", 1 );
	//imshow("better_result", img_smooth);
	//imshow("img", edgemap);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );
	//cvDestroyWindow( "img" );

	//return;

	int canny_h_thresd = 0;
	cal_canny_high_thresd(img_enhanced,0.01,canny_h_thresd);

	int canny_l_threshold = canny_h_thresd*0.5;

	Mat canny_ret_img;
	Canny(img_enhanced, canny_ret_img, canny_l_threshold, canny_h_thresd);
//	Canny( img_enhanced, canny_ret_img, 50, 70);

	//for(int i = 0; i < img_enhanced.rows; i++)
	//	for(int j = 0; j < img_enhanced.cols; j++)
	//	{
	//		if (dil_img.ptr<unsigned char>(i)[j] != 0)
	//		{
	//			canny_ret_img.ptr<unsigned char>(i)[j] = 0;
	//		}
	//	}

	Mat canny_ret_copy = canny_ret_img.clone();

	//int points_gap = 10;
	//int radiusRange[2] = {1, 20};
	//int searchRange = 10;

	int points_gap = 5;
	int radiusRange[2] = {10, 33};
	int searchRange = 3;

	cur_time = clock();
	double time_cost = cur_time - last_time;
	cout<<"  Image preprocessing time_cost = "<<time_cost<<endl;
	last_time = clock();

	Point center_point(0,0);

	function_volue = findCenter(canny_ret_copy,radiusRange,points_gap,center_point);

	if (function_volue != 0)
	{
		cout<<"Find pupil center failed!"<<endl;
		getchar();
		return;
	}

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  find Center time_cost = "<<time_cost<<endl;
	last_time = clock();

	cout<<"center_point = "<<center_point<<endl;

	Point iris_center(0,0);
	Point pupil_center(0,0);
	int pupil_rds = 0;
	int iris_rds = 0;
	 
	function_volue = find_pupil_radius(canny_ret_copy,center_point,radiusRange, searchRange,pupil_center,pupil_rds);

	if (function_volue != 0)
	{
		cout<<"Find pupil radius failed!"<<endl;
		getchar();
		return;
	}

	cout<<"pupil_center = "<<pupil_center<<endl;
	cout<<"pupil_rds = "<<pupil_rds<<endl;

	//circle(canny_ret_img,pupil_center,pupil_rds,cv::Scalar(255));
	//canny_ret_img.ptr<unsigned char>(pupil_center.y)[pupil_center.x] = 255;
	//canny_ret_img.ptr<unsigned char>(pupil_center.y+1)[pupil_center.x] = 255;
	//canny_ret_img.ptr<unsigned char>(pupil_center.y-1)[pupil_center.x] = 255;
	//canny_ret_img.ptr<unsigned char>(pupil_center.y)[pupil_center.x+1] = 255;
	//canny_ret_img.ptr<unsigned char>(pupil_center.y)[pupil_center.x-1] = 255;

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", canny_ret_img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	//imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\014_pupil_location_on_canny.jpg",canny_ret_img);

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  find radius time_cost = "<<time_cost<<endl;
	last_time = clock();

	Point left_top(((pupil_center.x - 8*pupil_rds) > 5?(pupil_center.x - 8*pupil_rds):5),((pupil_center.y - 8*pupil_rds) > 5?(pupil_center.y - 8*pupil_rds):5));
	Point right_bottom(((pupil_center.x + 8*pupil_rds) < (img_enhanced.cols -5)?(pupil_center.x + 8*pupil_rds):(img_enhanced.cols -5)),((pupil_center.y + 8*pupil_rds) < (img_enhanced.rows - 5)?(pupil_center.y + 8*pupil_rds):(img_enhanced.rows - 5)));
	
	Rect iris_area(left_top,right_bottom);
	Mat iris_img = img_enhanced(iris_area).clone();
	Mat iris_mask;

	GaussianBlur(iris_img,iris_img,Size(7,7),0);
	GaussianBlur(iris_img,iris_img,Size(7,7),0);

	iris_preprecessing(iris_img, Point(iris_area.width/2,iris_area.height/2), pupil_rds,iris_mask);

	double thresh = 15;
	Mat new_canny_result;
	soble_double_direction(iris_img,iris_mask,thresh,new_canny_result);

	removeSmallBlobs(new_canny_result,10);	

	Point new_pupil_center(pupil_center.x-iris_area.x,pupil_center.y-iris_area.y);

	int irisRange[2] = {(int)pupil_rds*1.5, (int)pupil_rds*7};
	function_volue = find_iris_radius(new_canny_result,new_pupil_center, irisRange, searchRange,iris_center,iris_rds);

	//circle(new_canny_result,iris_center,iris_rds,cv::Scalar(255));

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", new_canny_result);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	if (function_volue != 0)
	{
		cout<<"Find iris radius failed!"<<endl;
		getchar();
		return;
	}

	cout<<"iris_center = "<<iris_center<<endl;
	cout<<"iris_rds = "<<iris_rds<<endl;

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  find iris radius time_cost = "<<time_cost<<endl;
	last_time = clock();

	//circle(new_canny_result,iris_center,iris_rds,cv::Scalar(255));
	//circle(iris_img,iris_center,iris_rds,cv::Scalar(255));

	//cvNamedWindow( "new_canny_result", 1 );
	//cvNamedWindow( "iris_img", 1 );
	//imshow("better_result", new_canny_result);
	//imshow("iris_img", iris_img);
	//cvWaitKey(0);
	//cvDestroyWindow( "new_canny_result" );
	//cvDestroyWindow( "iris_img" );

	//imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\001_iris_location_on_canny.jpg",new_canny_result);
	//imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\001_iris_location.jpg",iris_img);

	//cout<<"iris_center = "<<iris_center<<endl;
	//cout<<"iris_rds = "<<iris_rds<<endl;
 
	//cur_time = clock();
	//time_cost = cur_time - last_time;
	//cout<<"  pupil location time_cost = "<<time_cost<<endl;
	//last_time = clock();


	//cur_time = clock();
	//time_cost = cur_time - last_time;

	Point rst_iris_center(iris_center.x+iris_area.x,iris_center.y+iris_area.y), rst_pupil_center((pupil_center.x),(pupil_center.y));
	int rst_iris_rds = iris_rds;
	int rst_pupil_rds = pupil_rds;

	circle(img,rst_iris_center,rst_iris_rds,cv::Scalar(255));
	circle(img,rst_pupil_center,rst_pupil_rds,cv::Scalar(255));

	

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	//Mat coffs_lower(3,1,CV_32FC1), coffs_upper(3,1,CV_32FC1);
	Mat coffs_lower, coffs_upper;

	//Mat struct_map;
	//Canny(img_enhanced, struct_map, canny_l_threshold, canny_h_thresd);

	double range[2] = {0.6,1.2};
	double thresh_canny[2] = {canny_l_threshold,canny_h_thresd};
	function_volue = fit_lower_eyelid(img_enhanced,rst_iris_center,rst_iris_rds,rst_pupil_center,rst_pupil_rds,range,thresh_canny,coffs_lower);

	//if (function_volue != 0)
	//{
	//	cout<<"Fitting lower eyelid failed!"<<endl;
	//	getchar();
	//	return;
	//}

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  lower eyelid location time_cost = "<<time_cost<<endl;
	last_time = clock();

	//cout<<"rst = "<<rst<<endl;
	//drawPolynomial(coffs_lower,1,img);

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	fit_upper_eyelid(img_enhanced,rst_iris_center,rst_iris_rds,rst_pupil_center,rst_pupil_rds,range,5,false,coffs_upper);

	//if (function_volue != 0)
	//{
	//	cout<<"Fitting upper eyelid failed!"<<endl;
	//	getchar();
	//	return;
	//}

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  Upper eyelid location time_cost = "<<time_cost<<endl;
	last_time = clock();

	if (!coffs_lower.empty())
	{
		drawPolynomial(coffs_lower,1,img);
	}
	
	if (!coffs_upper.empty())
	{
		drawPolynomial(coffs_upper,2,img);
	}
	//
	//cout<<"After draw Poly"<<endl;

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

//	imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\014_eyelid_fitting.jpg",img);

	Mat mask(img_enhanced.size(),CV_8UC1);
	//mask.setTo(255);

	//double thresh[2] = {0.1,0.8};
	//process_ES_region(img_enhanced, mask, rst_iris_center, rst_iris_rds, coffs_upper, thresh);

	Vector<cv::Point> eyelash_points;
	eyelash_pixels_location(img_enhanced,rst_iris_center,rst_iris_rds,pupil_center,pupil_rds,eyelash_points);
	mask.setTo(0);

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  Mask pixels location time_cost = "<<time_cost<<endl;
	last_time = clock();

	for (int i = 0; i < eyelash_points.size(); i++)
	{
		img.ptr<unsigned char>(eyelash_points[i].y)[eyelash_points[i].x] = 255;
		mask.ptr<unsigned char>(eyelash_points[i].y)[eyelash_points[i].x] = 255;
	}

//	imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\014_eyelash.jpg",img);

	if (!coffs_lower.empty())
	{
		drawPolynomial(coffs_lower,1,mask);
	}
	
	if (!coffs_upper.empty())
	{
		drawPolynomial(coffs_upper,2,mask);
	}

	//mask_lower_region(img_enhanced,iris_center,iris_rds,extend,mask,thresh_high,thresh_low,cir_correct);

	//imwrite("mask.bmp",mask);
	//imwrite("F:\\nir_face\\Fine images\\eyeImages\\final_result.bmp",img);

	Mat dst,mask_dst;
	iris_normalization(img,dst,mask,mask_dst,rst_pupil_center,rst_pupil_rds,rst_iris_center,rst_iris_rds,80,512);

	Mat final_dst = dst.clone();
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (mask_dst.ptr<unsigned char>(i)[j] != 0)
			{
				final_dst.ptr<unsigned char>(i)[j] = 255;
			}
		}
	}

	cvNamedWindow( "better_result", 1 );
	cvNamedWindow( "img", 1 );
	cvNamedWindow( "final_dst", 1 );
	imshow("better_result", img);
	imshow("final_dst", img_enhanced);
	imshow("img", final_dst);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );
	cvDestroyWindow( "img" );
	cvDestroyWindow( "final_dst" );


	//imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\007_nom_dst.jpg",dst);
	//imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\007_mask_dst.jpg",mask_dst);
	//imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\007_final_dst.jpg",final_dst);

	//imwrite("C:\\Users\\hkpuadmin\\Desktop\\eyeImages\\014_final_result.jpg",img);

//	getchar();
}



void cal_canny_high_thresd(Mat src, float ratio, int& high_thresd)
{
	Mat sobel_result;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_result);

	const int channels[1]={0};
    const int histSize[1]={256};
    float hranges[2]={0,255};
    const float* ranges[1]={hranges};
	Mat hist = Mat::zeros(256,1,CV_32FC1);
    calcHist(&sobel_result,1,channels,Mat(),hist,1,histSize,ranges);

	//double max_v = 0;
	//minMaxLoc(hist,0,&max_v);
	//Mat hist_show = Mat::zeros(257,256,CV_8UC1);
	//for (int i = 0; i < 256; i++)
	//{
	//	cout<<"hist.ptr<unsigned char>(i)[0] = "<<hist.ptr<float>(i)[0]<<endl;
	//	hist_show.ptr<unsigned char>((int)256-hist.ptr<float>(i)[0]*256/max_v)[i] = 255;
	//	//hist_show.ptr<unsigned char>((hist.ptr<float>(i)[0]*256/max_v-1)<0?0:(hist.ptr<float>(i)[0]*256/max_v-1))[i] = 255;
	//	//hist_show.ptr<unsigned char>((hist.ptr<float>(i)[0]*256/max_v+1)>255?255:(hist.ptr<float>(i)[0]*256/max_v+1))[i] = 255;
	//}
	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", hist_show);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	float num_thresd = src.cols*src.rows*ratio;
	
	int num_of_pixel = 0;
	for (int i = 255; i >= 0; i--)
	{
		num_of_pixel += hist.ptr<float>(i)[0];
		if (num_of_pixel > num_thresd)
		{
			high_thresd = i;
			break;
		}
		
	}
}