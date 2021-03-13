#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;




int main(int argc, char** argv)
{
	Mat image3 = imread("C:/hotel/hotel-02.png", IMREAD_GRAYSCALE);
	Mat image4 = imread("C:/hotel/hotel-03.png", IMREAD_GRAYSCALE); //LEFT-MOST IMAGE
	Mat image2 = imread("C:/hotel/hotel-01.png", IMREAD_GRAYSCALE);
	Mat image = imread("C:/hotel/hotel-00.png", IMREAD_GRAYSCALE); //RIGHT-MOST IMAGE

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvNamedWindow("Display window", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.

	cvNamedWindow("Display window 2", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window 2", image2); // Show our image inside it.


	//*****************************STITCHING OF IMAGES 1 AND 2 (THE TWO IMAGES ON THE RIGHT SIDE)**********************************

	SiftFeatureDetector detector = SiftFeatureDetector();
	//SurfFeatureDetector detector = SurfFeatureDetector(); 
	vector<KeyPoint> keypoints1, keypoints2,keypoints3,keypoints4,keypoints5,keypoints6;
	detector.detect(image, keypoints1);
	detector.detect(image2, keypoints2);
	

	SiftDescriptorExtractor descriptor = SiftDescriptorExtractor();
	//SurfDescriptorExtractor descriptor = SurfDescriptorExtractor();
	Mat descriptor1, descriptor2,descriptor3,descriptor4,descriptor5,descriptor6;
	descriptor.compute(image, keypoints1, descriptor1);
	descriptor.compute(image2, keypoints2, descriptor2);
	
	vector<DMatch> matches;
	double distance;
	for (int i = 0; i < descriptor2.rows; i++) {
		double mindistance = numeric_limits<double>().max();
		double mindistance2 = mindistance;
		int minj = 0;
		int minj2 = 0;
		for (int j = 0; j < descriptor1.rows; j++) {
			distance = sum(abs(descriptor2.row(i) - descriptor1.row(j)))[0];
			if (distance < mindistance) {
				minj2 = minj;
				mindistance2 = mindistance;
				mindistance = distance;
				minj = j;
			}
			else if (distance < mindistance2) {
				mindistance2 = distance;
				minj2 = j;
			}
		}
		if (mindistance / mindistance2 <= 0.5)
			matches.push_back(DMatch(i, minj, mindistance));
	}

	Mat image_matches;
	drawMatches(image2, keypoints2, image, keypoints1, matches, image_matches);
	cvNamedWindow("Display window 3", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window 3", image_matches);

	//Homography
	vector <Point2f> img1_pts, img2_pts;
	for (int i = 0; i < matches.size(); i++) {
		img2_pts.push_back(keypoints2[matches[i].queryIdx].pt);
		img1_pts.push_back(keypoints1[matches[i].trainIdx].pt);
	}

	Mat H = findHomography(img1_pts, img2_pts, CV_RANSAC);

	Mat firstResult;
	warpPerspective(image, firstResult, H, Size(1531,834));//Size is adjusted properly for our image set. Should change if image-set changes

	cv::Rect roi(cv::Point(0, 0), image2.size());
	image2.copyTo(firstResult(roi));

	cvNamedWindow("Stitch");
	imshow("Stitch", firstResult);
	


	//*********************STITCHING IMAGE3 (middle) WITH THE ONE CALCULATED ABOVE*************************************



	detector.detect(firstResult, keypoints5);
	detector.detect(image3, keypoints3);

	descriptor.compute(firstResult, keypoints5, descriptor5);
	descriptor.compute(image3, keypoints3, descriptor3);
	

	vector<DMatch> matches1;
	double distance1;
	for (int i = 0; i < descriptor3.rows; i++) {
		double mindistance = numeric_limits<double>().max();
		double mindistance2 = mindistance;
		int minj = 0;
		int minj2 = 0;
		for (int j = 0; j < descriptor5.rows; j++) {
			distance1 = sum(abs(descriptor3.row(i) - descriptor5.row(j)))[0];
			if (distance1 < mindistance) {
				minj2 = minj;
				mindistance2 = mindistance;
				mindistance = distance1;
				minj = j;
			}
			else if (distance1 < mindistance2) {
				mindistance2 = distance1;
				minj2 = j;
			}
		}
		if (mindistance / mindistance2 <= 0.5)
			matches1.push_back(DMatch(i, minj, mindistance));
	}

	Mat image_matches1;
	drawMatches(image3, keypoints3, firstResult, keypoints5, matches1, image_matches1);
	cvNamedWindow("Display window 4", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window 4", image_matches1);

	//Homography
	vector <Point2f> img3_pts, img5_pts;
	for (int i = 0; i < matches1.size(); i++) {
		img3_pts.push_back(keypoints3[matches1[i].queryIdx].pt);
		img5_pts.push_back(keypoints5[matches1[i].trainIdx].pt);
	}

	Mat H1 = findHomography(img5_pts, img3_pts, CV_RANSAC);

	Mat secondResult;
	warpPerspective(firstResult, secondResult, H1, Size(2521, 1045));//Size is adjusted properly for our image set. Should change if image-set changes
	cv::Rect roi1(cv::Point(0, 0), image3.size());
	image3.copyTo(secondResult(roi1));


	
	
	
	
	cvNamedWindow("Stitch2");
	imshow("Stitch2", secondResult);
	
	
	//***********STITCHING LEFT-MOST IMAGE WITH THE ONE CALCULATED ABOVE*************************

	detector.detect(image4, keypoints4);
	detector.detect(secondResult, keypoints6);

	descriptor.compute(image4, keypoints4, descriptor4);
	descriptor.compute(secondResult, keypoints6, descriptor6);


	vector<DMatch> matches2;
	double distance2;
	for (int i = 0; i < descriptor4.rows; i++) {
		double mindistance = numeric_limits<double>().max();
		double mindistance2 = mindistance;
		int minj = 0;
		int minj2 = 0;
		for (int j = 0; j < descriptor6.rows; j++) {
			distance2 = sum(abs(descriptor4.row(i) - descriptor6.row(j)))[0];
			if (distance2 < mindistance) {
				minj2 = minj;
				mindistance2 = mindistance;
				mindistance = distance2;
				minj = j;
			}
			else if (distance2 < mindistance2) {
				mindistance2 = distance2;
				minj2 = j;
			}
		}
		if (mindistance / mindistance2 <= 0.5)
			matches2.push_back(DMatch(i, minj, mindistance));
	}

	Mat image_matches2;
	drawMatches(image4, keypoints4, secondResult, keypoints6, matches2, image_matches2);
	cvNamedWindow("Display window 5", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window 5", image_matches2);

	//Homography
	vector <Point2f> img4_pts, img6_pts;
	for (int i = 0; i < matches2.size(); i++) {
		img4_pts.push_back(keypoints4[matches2[i].queryIdx].pt);
		img6_pts.push_back(keypoints6[matches2[i].trainIdx].pt);
	}

	Mat H2 = findHomography(img6_pts, img4_pts, CV_RANSAC);

	Mat finalResult;
	warpPerspective(secondResult, finalResult, H2, Size(4630, 1528));//Size is adjusted properly for our image set. Should change if image-set changes

	cv::Rect roi2(cv::Point(0, 0), image4.size());
	image4.copyTo(finalResult(roi2));
	


	cvNamedWindow("Stitch3Final");
	imshow("Stitch3Final", finalResult);

	

	while (cvWaitKey(33) != 27);
	return 0;
}