// Include required header files from OpenCV directory
#include "/usr/local/include/opencv2/objdetect.hpp"
#include "/usr/local/include/opencv2/highgui.hpp"
#include "/usr/local/include/opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, const char** argv ){
    // VideoCapture class for playing video for which faces to be detected
    VideoCapture capture; 
    Mat frame, image, lastFrame, diff;
    int i,j;
    double actionCoefficient,dist;

    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    //capture.open("test_vids/videoplayback.mp4"); 
    capture.open(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
    if( capture.isOpened() ){
        // Capture frames from video and detect faces
        cout << "Action Coefficient Started...." << endl;
        bool first = true;
        while(1){
            capture >> frame;
            if( frame.empty() )
                break;

            if(!first){
                absdiff(lastFrame,frame,diff);
                actionCoefficient = 0;
                for(i=0;i<diff.rows;i++){
                    for(j=0;j<diff.cols;j++){
                        Vec3b pix = diff.at<Vec3b>(i,j);

                        dist = sqrt(pix[0]*pix[0] + pix[1]*pix[1] + pix[2]*pix[2]);
                        actionCoefficient += dist;
                    }
                }
                cout << actionCoefficient << endl;
            }

            char c = (char)waitKey(10);

            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ) 
                break;
            first = false;
            lastFrame = frame;
        }
    }
    else
        cout<<"Could not Open Camera";
    return 0;
}
