// ObjectTrackingCPP.cpp

#include<opencv2/core/core.hpp>
#include "/usr/local/include/opencv2/objdetect.hpp"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <ctime>
#include<iostream>

#include "Blob.h"

#define SHOW_STEPS            // un-comment or comment this line to show steps or not

using namespace std;
using namespace cv;
// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool draw);
void detectAndDraw( Mat& img, CascadeClassifier& cascade, 
	CascadeClassifier& nestedCascade, double scale);
std::string cascadeName, nestedCascadeName;
///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;
    // VideoCapture class for playing video for which faces to be detected
    cv::Mat frame, image;

    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade, nestedCascade; 
    double scale=1;
    int i, j;
    double actionCoefficient;
    double ACTION_THRESH = 20;
    // Load classifiers from "opencv/data/haarcascades" directory 
    nestedCascade.load( "/Users/devinkim/Desktop/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml" ) ;

    // Change path before execution 
    cascade.load( "/Users/devinkim/Desktop/opencv/data/haarcascades/haarcascade_frontalcatface.xml" ) ; 
    std::vector<Blob> blobs;

    capVideo.open(0);
    capVideo.set(CV_CAP_PROP_BUFFERSIZE, 3);
    if (!capVideo.isOpened()) {                                                 // if unable to open video file
	std::cout << "error reading video file" << std::endl << std::endl;      // show error message
	return(0);                                                              // and exit program
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);
    //	capVideo >> imgFrame1;
    //	capVideo >> imgFrame2;
    bool blnFirstFrame = true;
    int chance = 0;
    while (1) {
	chance++;
	std::vector<Blob> currentFrameBlobs;

	cv::Mat imgFrame1Copy = imgFrame1.clone();
	cv::Mat imgFrame2Copy = imgFrame2.clone();

	cv::Mat imgDifference;
	cv::Mat imgThresh;

	if(!imgFrame1Copy.empty()){
	    cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
	    cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
	}

	cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(27, 27), 0);
	cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(27, 27), 0);

	cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

	cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

	cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

	cv::dilate(imgThresh, imgThresh, structuringElement5x5);
	cv::dilate(imgThresh, imgThresh, structuringElement5x5);
	cv::erode(imgThresh, imgThresh, structuringElement5x5);


	cv::Mat imgThreshCopy = imgThresh.clone();

	std::vector<std::vector<cv::Point> > contours;

	cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	drawAndShowContours(imgThresh.size(), contours, "imgContours");

	std::vector<std::vector<cv::Point> > convexHulls(contours.size());

	for (unsigned int i = 0; i < contours.size(); i++) {
	    cv::convexHull(contours[i], convexHulls[i]);
	}

	drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

	for (auto &convexHull : convexHulls) {
	    Blob possibleBlob(convexHull);

	    if (possibleBlob.currentBoundingRect.area() > 1000 &&
		    possibleBlob.dblCurrentAspectRatio >= 0.2 &&
		    possibleBlob.dblCurrentAspectRatio <= 1.25 &&
		    possibleBlob.currentBoundingRect.width > 100 &&
		    possibleBlob.currentBoundingRect.height > 200 &&
		    possibleBlob.dblCurrentDiagonalSize > 30.0 &&
		    (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.40) {
		currentFrameBlobs.push_back(possibleBlob);
	    }
	}

	drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

	if (blnFirstFrame == true) {
	    for (auto &currentFrameBlob : currentFrameBlobs) {
		blobs.push_back(currentFrameBlob);
	    }
	}
	else {
	    matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
	}

	drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");
	//////////////////////////////////////////////////////////////////////////////
	if(!blnFirstFrame && imgFrame1.cols == imgFrame2.cols && imgFrame1.rows == imgFrame2.rows){
	  Mat diff;
	  absdiff(imgFrame1,imgFrame2,diff);
	  double totalDist = 0;
	  for(j=0;j<diff.rows;j++){
	  for(i=0;i<diff.cols;i++){
	  Vec3b pix = diff.at<Vec3b>(j,i);
	  totalDist += sqrt(pix[0]*pix[0]+pix[1]*pix[1]+pix[2]*pix[2]);
	  }
	  }
	  actionCoefficient = totalDist / (diff.rows * diff.cols); 
	//   cout << actionCoefficient << endl;
	}
	/////////////////////////////////////////////////////////////////////////////
	imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

	if(chance == 1 ){
	    drawBlobInfoOnImage(blobs, imgFrame2Copy, cascade, nestedCascade,scale, actionCoefficient > ACTION_THRESH);
	    chance = 0;
	}
	else{
	    resize(imgFrame2Copy, imgFrame2Copy, Size(705, 397), 0, 0, INTER_CUBIC);
	    imshow("CAM 1", imgFrame2Copy);
	}
	//cv::imshow("imgFrame2Copy", imgFrame2Copy);
	//cv::waitKey(0);                 // uncomment this line to go frame by frame for debugging
	char c = (char)cv::waitKey(10);

	// now we prepare for the next iteration

	currentFrameBlobs.clear();

	imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is
	blnFirstFrame = false;
	capVideo.read(imgFrame2);
    }

    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

    return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

    for (auto &existingBlob : existingBlobs) {

	existingBlob.blnCurrentMatchFoundOrNewBlob = false;

	existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {

	int intIndexOfLeastDistance = 0;
	double dblLeastDistance = 100000.0;

	for (unsigned int i = 0; i < existingBlobs.size(); i++) {
	    if (existingBlobs[i].blnStillBeingTracked == true) {
		double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

		if (dblDistance < dblLeastDistance) {
		    dblLeastDistance = dblDistance;
		    intIndexOfLeastDistance = i;
		}
	    }
	}

	if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 1.15) {
	    addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
	}
	else {
	    addNewBlob(currentFrameBlob, existingBlobs);
	}

    }

    for (auto &existingBlob : existingBlobs) {

	if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
	    existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
	}

	if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
	    existingBlob.blnStillBeingTracked = false;
	}

    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    // cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto &blob : blobs) {
	if (blob.blnStillBeingTracked == true) {
	    contours.push_back(blob.currentContour);
	}
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    // cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob> &blobs, Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool draw) {

    vector<Rect> faces, faces2;
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    //GaussianBlur( img, gray, Size(21, 21),0, 0); // Apply Gaussian Blur
    double fx = 1 / scale;

    // Resize the Grayscale Image 
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR ); 
    equalizeHist( smallImg, smallImg );

    // Detect faces of different sizes using cascade classifier 
    cascade.detectMultiScale( smallImg, faces, 1.1, 
	    2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    // Draw circles around the faces
    for ( size_t i = 0; i < faces.size(); i++ )
    {
	Rect r = faces[i];
	Mat smallImgROI;
	vector<Rect> nestedObjects;
	Point center;
	int radius;

	double aspect_ratio = (double)r.width/r.height;
	if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
	{
	    center.x = cvRound((r.x + r.width*0.5)*scale);
	    center.y = cvRound((r.y + r.height*0.5)*scale);
	    radius = cvRound((r.width + r.height)*0.25*scale);
	    circle( img, center, radius, SCALAR_GREEN, 2, 8, 0 );
	}
	else
	    rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
		    cvPoint(cvRound((r.x + r.width-1)*scale), 
			cvRound((r.y + r.height-1)*scale)), SCALAR_GREEN, 2, 8, 0);
	if( nestedCascade.empty() )
	    continue;
	smallImgROI = smallImg( r );

	// Detection of eyes int the input image
	nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2,
		0|CASCADE_SCALE_IMAGE, Size(30, 30) ); 

	// Draw circles around eyes
	for ( size_t j = 0; j < nestedObjects.size(); j++ ) 
	{
	    Rect nr = nestedObjects[j];
	    center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
	    center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
	    radius = cvRound((nr.width + nr.height)*0.25*scale);
	    circle( img, center, radius, SCALAR_GREEN, 2, 8, 0 );
	}
    }
    for (unsigned int i = 0; i < blobs.size(); i++) {

	if (blobs[i].blnStillBeingTracked == true) {
	    cv::rectangle(img, blobs[i].currentBoundingRect, SCALAR_GREEN, 2);
	}
    }
    /////////////////////// ELIS CODE XD ///////////////////////////////
        if(draw){
	  rectangle(img,cvPoint(0,0),cvPoint(cvRound(img.cols*scale),cvRound(img.rows*scale)),SCALAR_RED,20,8,0);
	  }
	  Mat roi = img(Rect(5,5,320,35));
	  Mat color(roi.size(), CV_8UC3, SCALAR_BLACK);
	  double alpha = .7;
	  addWeighted(color, alpha, roi, 1.0-alpha, 0.0, roi);

	  time_t time = chrono::system_clock::to_time_t(chrono::system_clock::now());
	  String timestamp = ctime(&time);
	  timestamp = timestamp.substr(0,timestamp.size()-1);
	  putText(img,timestamp,cvPoint(10*scale,30*scale),FONT_HERSHEY_SIMPLEX,.7,SCALAR_WHITE,2,LINE_AA);
    resize(img, img, Size(705, 397), 0, 0, INTER_CUBIC);
    imshow( "CAM 1", img ); 
}
