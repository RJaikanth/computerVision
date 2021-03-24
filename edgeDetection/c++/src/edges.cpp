#include <opencv2/opencv.hpp>
#include <iostream>

void sobel(cv::InputArray image, cv::OutputArray dest);
void prewitt(cv::InputArray image, cv::OutputArray dest);
void roger(cv::InputArray image, cv::OutputArray dest);
void sobelRoger(cv::InputArray image, cv::OutputArray dest);
void sobelPrewitt(cv::InputArray image, cv::OutputArray dest);

int main(int argc, char const *argv[])
{
    cv::VideoCapture cap(atoi(argv[1]));
    cv::Mat frame, sobelOut, prewittOut, rogerOut, cannyOut;

    cv::namedWindow("Image");
    cv::namedWindow("Sobel");
    cv::namedWindow("Prewitt");
    cv::namedWindow("Canny");

    for (;;) 
    {
        cap >> frame;
        cv::imshow("Image", frame);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        sobel(frame, sobelOut);
        prewitt(frame, prewittOut);
        cv::Canny(frame, cannyOut, 0, 200);

        cv::imshow("Sobel", sobelOut);
        cv::imshow("Prewitt", prewittOut);
        cv::imshow("Canny", cannyOut);

        char c = (char)cv::waitKey(1);
        if (c == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}


void sobel(cv::InputArray image, cv::OutputArray dest)
{
    cv::Mat grad_x, grad_y;

    cv::Sobel(image, grad_x, CV_16S, 1, 0, 3, 1, 0, 4);
    cv::Sobel(image, grad_y, CV_16S, 0, 1, 3, 1, 0, 4);

    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);

    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dest);
}

void prewitt(cv::InputArray image, cv::OutputArray dest)
{
    float dx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    float dy[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};

    cv::Mat kernel1(3, 3, CV_32F, dx);
    cv::Mat kernel2(3, 3, CV_32F, dy);
    cv::Mat grad_x, grad_y, blurredImage;

    cv::GaussianBlur(image, blurredImage, cv::Size(3, 3), 1, 1);
    cv::filter2D(blurredImage, grad_x, -1, kernel1);
    cv::filter2D(blurredImage, grad_y, -1, kernel2);

    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);

    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dest);
}

void roger(cv::InputArray image, cv::OutputArray dest)
{
    int dx_data[4] = {0, 1, -1, 0};
    int dy_data[4] = {1, 0, 0, -1};

    cv::Mat kernel1(2, 2, CV_32F, dx_data);
    cv::Mat kernel2(2, 2, CV_32F, dy_data);

    cv::Mat grad_x, grad_y, blurred_image;

    cv::GaussianBlur(image, blurred_image, cv::Size(3, 3), 1);
    cv::filter2D(blurred_image, grad_x, -1, kernel1);
    cv::filter2D(blurred_image, grad_y, -1, kernel2);

    cv::imshow("Roger", grad_x);

    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);

    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dest);

}

