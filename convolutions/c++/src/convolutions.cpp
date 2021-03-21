#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>

int main(int argc, const char **argv)
{
    // Define capture object and frame
    cv::VideoCapture cap(atoi(argv[1]));//, NULL, 10));
    cv::Mat frame;
    cv::Mat temp;
    cv::Mat out;

    // Create image window
    cv::namedWindow("Image");
    cv::namedWindow("Convolved Image");

    // Check if stream is correct
    if (!cap.isOpened())
    {
        std::cout << "Could Not open Video Stream. Check if device is connected.\n";
        return -1;
    }

    float sobelx_data[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};    
    float sobely_data[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};    

    cv::Mat kernel1(3, 3, CV_32F, sobelx_data);
    cv::Mat kernel2(3, 3, CV_32F, sobely_data);


    for (;;)
    {
        cap >> frame;
        cv::imshow("Image", frame);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        
        cv::filter2D(frame, temp, -1, kernel1);
        cv::filter2D(frame, out, -1, kernel2);
        cv::add(out, temp, out);

        cv::imshow("Convolved Image", out);

        char c = (char)cv::waitKey(1);
        if (c == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
