// �㷨ʵ��.cpp :��������˾�
// jsxyhelu.cnblogs.com(2017/07)

#include "stdafx.h"
#include <opencv2/photo.hpp>
#include "HSL.hpp"
using namespace std;
using namespace cv;

static string window_name = "photo";
static Mat src;

static HSL hsl;
static int color = 0;
static int hue = 180;
static int saturation = 100;
static int brightness = 100;

#define  VP  vector<Point>  //��VP���Ŵ��� vector<point>

//Ѱ����������
VP FindBigestContour(Mat src){    
	int imax = 0; //����������������
	int imaxcontour = -1; //������������Ĵ�С
	std::vector<std::vector<Point>>contours;    
	findContours(src,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
	for (int i=0;i<contours.size();i++){
		int itmp =  contourArea(contours[i]);//������õ���������С
		if (imaxcontour < itmp ){
			imax = i;
			imaxcontour = itmp;
		}
	}
	return contours[imax];
}
 
// Remove black dots (upto 4x4 in size) of noise from a pure black & white image.  
// ie: The input image should be mostly white (255) and just contains some black (0) noise  
// in addition to the black (0) edges.  
void removePepperNoise(Mat &mask)  
{  
	// For simplicity, ignore the top & bottom row border.  
	for (int y=2; y<mask.rows-2; y++) {  
		// Get access to each of the 5 rows near this pixel.  
		uchar *pThis = mask.ptr(y);  
		uchar *pUp1 = mask.ptr(y-1);  
		uchar *pUp2 = mask.ptr(y-2);  
		uchar *pDown1 = mask.ptr(y+1);  
		uchar *pDown2 = mask.ptr(y+2);  

		// For simplicity, ignore the left & right row border.  
		pThis += 2;  
		pUp1 += 2;  
		pUp2 += 2;  
		pDown1 += 2;  
		pDown2 += 2;  
		for (int x=2; x<mask.cols-2; x++) {  
			uchar v = *pThis;   // Get the current pixel value (either 0 or 255).  
			// If the current pixel is black, but all the pixels on the 2-pixel-radius-border are white  
			// (ie: it is a small island of black pixels, surrounded by white), then delete that island.  
			if (v == 0) {  
				bool allAbove = *(pUp2 - 2) && *(pUp2 - 1) && *(pUp2) && *(pUp2 + 1) && *(pUp2 + 2);  
				bool allLeft = *(pUp1 - 2) && *(pThis - 2) && *(pDown1 - 2);  
				bool allBelow = *(pDown2 - 2) && *(pDown2 - 1) && *(pDown2) && *(pDown2 + 1) && *(pDown2 + 2);  
				bool allRight = *(pUp1 + 2) && *(pThis + 2) && *(pDown1 + 2);  
				bool surroundings = allAbove && allLeft && allBelow && allRight;  
				if (surroundings == true) {  
					// Fill the whole 5x5 block as white. Since we know the 5x5 borders  
					// are already white, just need to fill the 3x3 inner region.  
					*(pUp1 - 1) = 255;  
					*(pUp1 + 0) = 255;  
					*(pUp1 + 1) = 255;  
					*(pThis - 1) = 255;  
					*(pThis + 0) = 255;  
					*(pThis + 1) = 255;  
					*(pDown1 - 1) = 255;  
					*(pDown1 + 0) = 255;  
					*(pDown1 + 1) = 255;  
				}  
				// Since we just covered the whole 5x5 block with white, we know the next 2 pixels  
				// won't be black, so skip the next 2 pixels on the right.  
				pThis += 2;  
				pUp1 += 2;  
				pUp2 += 2;  
				pDown1 += 2;  
				pDown2 += 2;  
			}  
			// Move to the next pixel.  
			pThis++;  
			pUp1++;  
			pUp2++;  
			pDown1++;  
			pDown2++;  
		}  
	}  
}  
 

void cartoonifyImage(Mat srcColor, Mat dst)  
{  
	// Convert from BGR color to Grayscale  
	Mat srcGray;  
	cvtColor(srcColor, srcGray, CV_BGR2GRAY);  
	// Remove the pixel noise with a good Median filter, before we start detecting edges.  
	medianBlur(srcGray, srcGray, 7);  
	Size size = srcColor.size();  
	Mat mask = Mat(size, CV_8U);  
	Mat edges = Mat(size, CV_8U);   
	// Generate a nice edge mask, similar to a pencil line drawing.  
	Laplacian(srcGray, edges, CV_8U, 5);  
	threshold(edges, mask, 80, 255, THRESH_BINARY_INV);  
	// Mobile cameras usually have lots of noise, so remove small  
	// dots of black noise from the black & white edge mask.  
	removePepperNoise(mask);  
	// Do the bilateral filtering at a shrunken scale, since it  
	// runs so slowly but doesn't need full resolution for a good effect.  
	Size smallSize;  
	smallSize.width = size.width/2;  
	smallSize.height = size.height/2;  
	Mat smallImg = Mat(smallSize, CV_8UC3);  
	resize(srcColor, smallImg, smallSize, 0,0, INTER_LINEAR);  

	// Perform many iterations of weak bilateral filtering, to enhance the edges  
	// while blurring the flat regions, like a cartoon.  
	Mat tmp = Mat(smallSize, CV_8UC3);  
	int repetitions = 7;        // Repetitions for strong cartoon effect.  
	for (int i=0; i<repetitions; i++) {  
		int size = 9;           // Filter size. Has a large effect on speed.  
		double sigmaColor = 9;  // Filter color strength.  
		double sigmaSpace = 7;  // Positional strength. Effects speed.  
		bilateralFilter(smallImg, tmp, size, sigmaColor, sigmaSpace);  
		bilateralFilter(tmp, smallImg, size, sigmaColor, sigmaSpace);  
	}  
	// Go back to the original scale.  
	resize(smallImg, srcColor, size, 0,0, INTER_LINEAR);  
	// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).  
	// Use the blurry cartoon image, except for the strong edges that we will leave black.  
	srcColor.copyTo(dst);  
}  


int _tmain(int argc, _TCHAR* argv[])
{
	Mat matSrc = imread("src3.jpg");
	Mat matCloud=imread("stares.jpg");
	Mat temp;Mat matDst;Mat mask;
	vector<Mat> planes;
	/************************************************************************/
	/* 1.��������գ��ָ�                                                                  */
	/************************************************************************/
	cvtColor(matSrc,temp,COLOR_BGR2HSV);
	split(temp,planes);
	equalizeHist(planes[2],planes[2]);//��vͨ������equalizeHist
	merge(planes,temp);
	inRange(temp,Scalar(100,43,46),Scalar(124,255,255),temp);
	erode(temp,temp,Mat());//��̬ѧ�任����ڲ��ն�
	dilate(temp,temp,Mat());
	mask = temp.clone();//���������mask
	/************************************************************************/
	/* 2.���ں�,��1�Ľ��mask��ֱ�ӽ���ͼ����������֮ǰ��Ҫ�����߶ȱ任��                                                                     */
	/************************************************************************/
	//2017��7��25�� ���Ѱ�Ұ�ɫ���������Ӿ��εĴ���
	VP maxCountour = FindBigestContour(mask);
	Rect maxRect = boundingRect(maxCountour);
	if (maxRect.height == 0 || maxRect.width == 0)
	   maxRect =  Rect(0,0,mask.cols,mask.rows);//�������
	
	//��������,ע��������ͽ�ǰ����ͨ����
	matDst = matSrc.clone();
	//cartoonifyImage(matSrc,matDst);
	resize(matCloud,matCloud,matDst.size());
	matCloud.copyTo(matDst,mask);
	//Ϊseamless׼������

	//ע�������mask ��Ҫ��matCloudͬ���ߴ� 
	//A rough mask around the object you want to clone.
	//This should be the size of the source image. Set it to an all white image if you are lazy!
	mask = mask(maxRect);
	resize(matCloud,matCloud,maxRect.size());
	//seamless clone
	Point center =  Point ((maxRect.x+maxRect.width)/2,(maxRect.y+maxRect.height)/2);//�м�λ��Ϊ����ı���λ��
	Mat normal_clone;
	Mat mixed_clone;
	Mat monochrome_clone;
	
	seamlessClone(matCloud, matSrc, mask, center, normal_clone, NORMAL_CLONE);
	seamlessClone(matCloud, matSrc, mask, center, mixed_clone, MIXED_CLONE);
	seamlessClone(matCloud, matSrc, mask, center, monochrome_clone, MONOCHROME_TRANSFER);
	/************************************************************************/
	/* 3.��ͨ������                                                            */
	/************************************************************************/
	//˫���˲�
	bilateralFilter(normal_clone,temp,5,10.0,2.0);
	//��ɫֱ��ͼ���⣬��RGBͼ��ת��YCbCr������Ȼ���Y�����ϵ�ͼ�����ֱ��ͼ���⻯
	cvtColor(temp,temp,COLOR_BGR2YCrCb);
	split(temp,planes);
	equalizeHist(planes[0],planes[0]);
	merge(planes,temp);
	cvtColor(temp,temp,COLOR_YCrCb2BGR);
	//��߱��Ͷ�
	Mat Img_out(temp.size(), CV_32FC3);  
	temp.convertTo(Img_out, CV_32FC3);  
	Mat Img_in(temp.size(), CV_32FC3);  
	temp.convertTo(Img_in, CV_32FC3);  
	// define the iterator of the input image  
	MatIterator_<Vec3f> inp_begin, inp_end;  
	inp_begin=Img_in.begin<Vec3f>();  
	inp_end =Img_in.end<Vec3f>();  
	// define the iterator of the output image  
	MatIterator_<Vec3f> out_begin, out_end;  
	out_begin=Img_out.begin<Vec3f>();  
	out_end =Img_out.end<Vec3f>();  
	// increment (-100.0, 100.0)  
	float Increment=50.0/100.0;   //���ͶȲ�������
	float delta=0;  
	float minVal, maxVal;  
	float t1, t2, t3;  
	float L,S;  
	float alpha;  

	for(; inp_begin!=inp_end; inp_begin++, out_begin++)  
	{  
		t1=(*inp_begin)[0];  
		t2=(*inp_begin)[1];  
		t3=(*inp_begin)[2];  

		minVal=std::min(std::min(t1,t2),t3);  
		maxVal=std::max(std::max(t1,t2),t3);  
		delta=(maxVal-minVal)/255.0;  
		L=0.5*(maxVal+minVal)/255.0;  
		S=std::max(0.5*delta/L, 0.5*delta/(1-L));  

		if (Increment>0)  
		{  
			alpha=max(S, 1-Increment);  
			alpha=1.0/alpha-1;  
			(*out_begin)[0]=(*inp_begin)[0]+((*inp_begin)[0]-L*255.0)*alpha;  
			(*out_begin)[1]=(*inp_begin)[1]+((*inp_begin)[1]-L*255.0)*alpha;  
			(*out_begin)[2]=(*inp_begin)[2]+((*inp_begin)[2]-L*255.0)*alpha;  
		}  
		else  
		{  
			alpha=Increment;  
			(*out_begin)[0]=L*255.0+((*inp_begin)[0]-L*255.0)*(1+alpha);  
			(*out_begin)[1]=L*255.0+((*inp_begin)[1]-L*255.0)*(1+alpha);  
			(*out_begin)[2]=L*255.0+((*inp_begin)[2]-L*255.0)*(1+alpha);  

		}  
	}  
	Img_out /=255;
	Img_out.convertTo(matDst,CV_8UC3,255);

	
	imshow("ԭʼͼ",matSrc);
	imshow("���ͼ",matDst);
	cv::waitKey();
	getchar();
	return 0;
}

