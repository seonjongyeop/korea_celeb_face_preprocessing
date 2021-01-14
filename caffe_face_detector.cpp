#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
const String config = "deploy.prototxt";
//const String model = "opencv_face_detector_uint8.pb"; 
//const String config = "opencv_face_detector.pbtxt";

int main(void)
{
	int imgcount = 0;
	Net net = readNet(model, config);

	if (net.empty()) {
		cerr << "Net open failed!" << endl;
		return -1;
	}

	for (int i = 2; i < 7; i++) {

		for (int k = 1; k < 2; k++) {

			Mat image; 
			Mat subImage;
			string strfdpath = "./" + to_string(i) + ".jpg";
			image = imread(strfdpath, IMREAD_COLOR);
			Mat image2 = image;
			Mat resize;
			if (image.empty())
			{
				strfdpath = "./"  + to_string(i) + ".png";
				image = imread(strfdpath, IMREAD_COLOR);
				image2 = image;
				if (image.empty()) {
					cout << "Could not open or find the image" << endl;
					continue;
				}
			}


			Mat blob = blobFromImage(image, 1, Size(300, 300), Scalar(104, 177, 123));
			net.setInput(blob);
			Mat res = net.forward();

			Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>());

			for (int j = 0; j < detect.rows; j++) {
				float confidence = detect.at<float>(j, 2);

				if (confidence < 0.8) break;

				int x1 = cvRound(detect.at<float>(j, 3) * image.cols);
				int y1 = cvRound(detect.at<float>(j, 4) * image.rows);
				int x2 = cvRound(detect.at<float>(j, 5) * image.cols);
				int y2 = cvRound(detect.at<float>(j, 6) * image.rows);
				/*Rect crop2(Point(x1, y1), Point(x2, y2));
				subImage = image2(crop2);
				imshow("ffff", subImage);*/
				//rectangle(image, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 255, 0));

				//String label = format("Face: %4.3f", confidence);
				//putText(image, label, Point(x1, y1 - 1), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0));
				//imshow("frame", image);

				
				int scale = (x2 - x1) * 0.5;
				x1 = x1 - scale;
				y1 = y1 - scale;
				x2 = x2 + scale;
				y2 = y2 + scale;
				
				int h = y2 - y1; 
				int w = x2 - x1;
				int sub = h - w;
				h = h - sub;
				y2 = y1 + h*1.0;
				imgcount++;
				
				cout << to_string(i) << "-" << to_string(k) << "( " << to_string(imgcount) << " pic)" << endl;
				cout << "x1: " << x1 << " y1: " << y1 << endl;
				cout << "x2: " << x2 << " y2: " << y2 << endl;
				
				cout << " image.cols: " << image.cols;
				cout << "image.rows: " << image.rows << endl;
				cout << "-----------" << endl;
				
				Rect bounds(0, 0, image.cols, image.rows);
				Rect crop(Point(x1, y1), Point(x2, y2));
				subImage = image2(bounds & crop);
				//imshow("frame", subImage);
				cv::resize(subImage, resize, Size(500, 500), 0, 0, INTER_AREA);
				imshow("frame2", resize);
				string savefdpath = "./test/" + to_string(i) + ".jpg";
				
				imwrite(savefdpath, resize);


				break;


				//waitKey(0);



				//rectangle(image, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 255, 0)); // �簢������ �� ǥ���� �ִ� ���

				//String label = format("Face: %4.3f", confidence);
				//putText(image, label, Point(x1, y1 - 1), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0));

				//imshow("frame", image);
			}


			waitKey(0);
		}
	}

}
