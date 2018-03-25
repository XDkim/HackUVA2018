// Include required header files from OpenCV directory
#include "/usr/local/include/opencv2/objdetect.hpp"
#include "/usr/local/include/opencv2/highgui.hpp"
#include "/usr/local/include/opencv2/imgproc.hpp"
#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;
using namespace cv;

void draw( Mat& img, double scale, bool drawRect);

int main( int argc, const char** argv ){
    // VideoCapture class for playing video for which faces to be detected
    VideoCapture capture; 
    Mat frame, lastFrame;
    int i,j;
    double actionCoefficient,dist;
    double scale = 1;

    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    bool webcam = true;
    double ACTION_THRESH = 20;
    if(!webcam)
        capture.open("test_vids/videoplayback.mp4"); 
    else
        capture.open(0);
   // capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
   // capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
    if( capture.isOpened() ){
        cout << "Action Coefficient Started...." << endl;
        bool first = true;
        while(1){
            capture >> frame;
            if( frame.empty() )
                break;

            //cout << "f " << frame.at<Vec3b>(200,200) << endl;

            if(!first && frame.cols == lastFrame.cols && frame.rows == lastFrame.rows){
                Mat diff;
                absdiff(frame,lastFrame,diff);
                double totalDist = 0;
                for(j=0;j<diff.rows;j++){
                    for(i=0;i<diff.cols;i++){
                        Vec3b pix = diff.at<Vec3b>(j,i);
                        totalDist += sqrt(pix[0]*pix[0]+pix[1]*pix[1]+pix[2]*pix[2]);
                    }
                }
                actionCoefficient = totalDist / (diff.rows * diff.cols); 
                cout << actionCoefficient << endl;
            }

            lastFrame = frame.clone();
            //cout << "lf " << lastFrame.at<Vec3b>(0,0) << endl;

            if(webcam){
                draw(frame,scale,actionCoefficient > ACTION_THRESH);
            }
            
            char c = (char)waitKey(10);

            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ) 
                break;
            first = false;
        }
    }
    else
        cout<<"Could not Open Camera";
    return 0;
}

void draw( Mat& img,
        double scale, bool drawRect){
    Scalar red = Scalar(0,0,255);
    Scalar black = Scalar(0,0,0);
    if(drawRect){
        rectangle(img,cvPoint(0,0),cvPoint(cvRound(img.cols*scale),cvRound(img.rows*scale)),red,20,8,0);
    }
    Mat roi = img(Rect(5,5,320,35));
    Mat color(roi.size(), lsCV_8UC3, Scalar(0,0,0));
    double alpha = .7;
    addWeighted(color, alpha, roi, 1.0-alpha, 0.0, roi);
    
    time_t time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    String timestamp = ctime(&time);
    timestamp = timestamp.substr(0,timestamp.size()-1);
    putText(img,timestamp,cvPoint(10*scale,30*scale),FONT_HERSHEY_SIMPLEX,.7,Scalar(255,255,255),2,LINE_AA);
    //resize(img, img, Size(640, 360), 0, 0, INTER_CUBIC);
    


    imshow( "Webcam", img ); 
}
