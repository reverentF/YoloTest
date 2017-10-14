#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;

const size_t network_width = 416;
const size_t network_height = 416;

int main(int argc, char** argv)
{
    String modelConfiguration = "yolo.cfg"; // parser.get<string>("cfg");
    String modelBinary = "yolo.weights"; //parser.get<string>("model");
    
    //! [Initialize network]
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    //! [Initialize network]
    
    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }
    
    // initialize tag names
    vector<string> tag_names;
    ifstream fin;
    fin.open("names.txt");
    
    string buf;
    while(fin && getline(fin, buf)) {
        tag_names.push_back(buf);
    }
    
    // init camera
    cv::VideoCapture cap(0);
    if(!cap.isOpened()){
        cout << "cannnot open camera.";
        return -1;
    }
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 800);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
    
    // for calc FPS
    int cnt = 0;
    int oldcnt = 0;
    int64 nowTime = 0;
    int64 diffTime = 0;
    
    int fps = 0;
    const double f = (1000 /cv::getTickFrequency());
    
    cv::Mat frame;
    int64 startTime = cv::getTickCount();
    while(true){
        cap >> frame;
        
        //! [Resizing without keeping aspect ratio]
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(network_width, network_height));
        //! [Resizing without keeping aspect ratio]
        
        //! [Prepare blob]
        Mat inputBlob = blobFromImage(resized, 1 / 255.F); //Convert Mat to batch of images
        //! [Prepare blob]
        
        //! [Set input blob]
        net.setInput(inputBlob, "data");                //set the network input
        //! [Set input blob]
        
        //! [Make forward pass]
        cv::Mat detectionMat = net.forward("detection_out");    //compute output
        //! [Make forward pass]
        
        
        float confidenceThreshold = 0.24; //parser.get<float>("min_confidence");
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            
            size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
            
            if (confidence > confidenceThreshold)
            {
                float x = detectionMat.at<float>(i, 0);
                float y = detectionMat.at<float>(i, 1);
                float width = detectionMat.at<float>(i, 2);
                float height = detectionMat.at<float>(i, 3);
                float xLeftBottom = (x - width / 2) * frame.cols;
                float yLeftBottom = (y - height / 2) * frame.rows;
                float xRightTop = (x + width / 2) * frame.cols;
                float yRightTop = (y + height / 2) * frame.rows;
                
                std::cout << "Class: " << tag_names[objectClass] << std::endl;
                std::cout << "Confidence: " << confidence << std::endl;
                
                std::cout << " " << xLeftBottom
                << " " << yLeftBottom
                << " " << xRightTop
                << " " << yRightTop << std::endl;
                
                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));
                
                rectangle(frame, object, Scalar(0, 255, 0));
                cv::putText(frame, tag_names[objectClass], Point((int)xLeftBottom, (int)yLeftBottom), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2, CV_AA);
            }
        }
        
        // calc & display FPS
        nowTime = cv::getTickCount();
        diffTime = (int)((nowTime- startTime)*f);
        
        if (diffTime >= 1000) {
            startTime = nowTime;
            fps = cnt - oldcnt;
            oldcnt = cnt;
        }
        cnt++;
        
        std::ostringstream os;
        os << fps;
        std::string number = os.str();
        
        cv::putText(frame, "FPS : " + number, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,200), 2, CV_AA);

        
        imshow("detections", frame);
        int k = waitKey(1);
        if(k == 'q'){
            break;
        }
    }
    return 0;
} // main

