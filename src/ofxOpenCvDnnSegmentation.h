#pragma once

#include "ofMain.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "ofxCv.h"

using namespace ofxCv;
using namespace cv;
using namespace cv::dnn;
using namespace std;



class ofxOpenCvDnnSegmentation{
public:
    ofxOpenCvDnnSegmentation();
    ~ofxOpenCvDnnSegmentation();
    void setup(string _path_to_weights,
               string _path_to_classlist);
    void setup(string _path_to_weights,
               string _path_to_config,
               string _path_to_classlist);

    void update(ofPixels &op);
    void setNetworkImageSize(int _w, int _h);
    vector<ofPolyline>getBlobs(int _class_number, float _threshold);
    void draw(int _x, int _y, int _w, int _h);
    void drawClass(int _class_id, int _alpha, int _x, int _y, int _w, int _h);
    void drawColorPalette(int _x, int _y, int _w, int _h);
    
    cv::Mat toCV(ofPixels &pix);
    dnn::Net net;
    int network_width = 512;
    int network_height = 256;
    int input_width;
    int input_height;
    float threshold;
    float scale;
    vector<ofPoint>p;
    Mat color;
    
    void colorizeSegmentation(const Mat &score, Mat &segm);
    std::vector<std::string> classes;
    std::vector<Vec3b> cv_colors;
    vector<ofColor> colors;
    ofImage image_segmented;
    ofxCv::ContourFinder contourFinder;
    ofColor targetColor;
};
