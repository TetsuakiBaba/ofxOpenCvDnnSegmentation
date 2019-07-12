#include "ofxOpenCvDnnSegmentation.h"

vector<string> split(string& input, char delimiter)
{
    istringstream stream(input);
    string field;
    vector<string> result;
    while (getline(stream, field, delimiter)) {
        result.push_back(field);
    }
    return result;
}


ofxOpenCvDnnSegmentation::ofxOpenCvDnnSegmentation()
{
    
}



ofxOpenCvDnnSegmentation::~ofxOpenCvDnnSegmentation()
{
    
}

template <typename T>
static cv::Mat toCvMat(ofPixels_<T>& pix)
{
    int depth;
    switch(pix.getBytesPerChannel())
    {
        case 4: depth = CV_32F; break;
        case 2: depth = CV_16U; break;
        case 1: default: depth = CV_8U; break;
    }
    return cv::Mat(pix.getHeight(), pix.getWidth(), CV_MAKETYPE(depth, pix.getNumChannels()), pix.getData(), 0);
}

cv::Mat ofxOpenCvDnnSegmentation::toCV(ofPixels &pix)
{
    return cv::Mat(pix.getHeight(), pix.getWidth(), CV_MAKETYPE(CV_8U, pix.getNumChannels()), pix.getData(), 0);
}

void ofxOpenCvDnnSegmentation::update(ofPixels &op)
{
    if( image_segmented.isAllocated() ){
        image_segmented.clear();
    }
    
    Mat img = toCvMat(op);
    input_width = (int)op.getWidth();
    input_height = (int)op.getHeight();
    
    Mat blob;
  
    Scalar *mean = new Scalar(0,0,0);
    bool swapRB = false;
    blobFromImage(img, blob, scale, cv::Size(network_width, network_height), *mean, swapRB, false);
    net.setInput(blob);
    Mat score = net.forward();
    Mat segm;
    colorizeSegmentation(score, segm);

    image_segmented.allocate(network_width, network_height,
                             OF_IMAGE_COLOR);
    unsigned char *pixels = new unsigned char[network_width*network_height*3];
    int pos = 0;
    for( int y = 0; y < segm.rows; y++ ) {
        cv::Vec3b* ptr = segm.ptr<cv::Vec3b>( y );
        for( int x = 0; x < segm.cols; x++ ) {
            cv::Vec3b bgr = ptr[x];
            pixels[pos] = bgr[2];pos++;
            pixels[pos] = bgr[1];pos++;
            pixels[pos] = bgr[0];pos++;
        }
    }
    
    image_segmented.setFromPixels(pixels, network_width, network_height, OF_IMAGE_COLOR);
    
    
    
    delete mean;
    delete[] pixels;
}

void ofxOpenCvDnnSegmentation::colorizeSegmentation(const Mat &score, Mat &segm)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];
    
    if (chns != (int)cv_colors.size())
    {
        CV_Error(Error::StsError, format("Number of output classes does not match "
                                         "number of colors (%d != %d)", chns, cv_colors.size()));
    }
    Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }
    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = cv_colors[ptrMaxCl[col]];
        }
    }
}


void ofxOpenCvDnnSegmentation::setup(string _path_to_weights,
                                     string _path_to_classlist)
{
  
    setup(_path_to_weights, "", _path_to_classlist);
}

void ofxOpenCvDnnSegmentation::setup(string _path_to_weights,
                                     string _path_to_config,
                                     string _path_to_classlist)
{
    // read the network model
    String modelBinary = _path_to_weights;
    net = readNet(modelBinary,_path_to_config);
    
    std::vector<String> lname = net.getLayerNames();
    for (int i = 0; i < lname.size();i++) {
        std::cout << i+1 << " " << lname[i] << std::endl;
    }
    
    ocl::setUseOpenCL( true );
    net.setPreferableTarget(DNN_TARGET_CPU);

    if (net.empty())
    {
        cout << "Can't load network by using the following files: " << endl;
        cout << "model-file: " << modelBinary << endl;
    }
    

    // open class list file which includes rgb infomation
    {
        ifstream ifs(_path_to_classlist);
        string line;
        while (getline(ifs, line)) {
            
            vector<string> strvec = split(line, ' ');
            
            Vec3b color;
            for (int i=1; i<strvec.size();i++){
                color[strvec.size()-i-1] = ofToInt(strvec.at(i));
            }
            classes.push_back(strvec.at(0));
            cv_colors.push_back(color);
            colors.push_back(ofColor(color[2],color[1],color[0]));
            
        }
    }
    
    contourFinder.setMinAreaRadius(10);
    //contourFinder.setMaxAreaRadius(150);
    targetColor = colors[1];
    contourFinder.setTargetColor(targetColor, true ? TRACK_COLOR_HS : TRACK_COLOR_RGB);
    contourFinder.setThreshold(0.5);
   
    
}

void ofxOpenCvDnnSegmentation::setNetworkImageSize(int _w, int _h)
{
    network_width = _w;
    network_height = _h;
}

void ofxOpenCvDnnSegmentation::drawColorPalette(int _x, int _y, int _w, int _h)
{
    float w = _w;
    float h = _h/(float)colors.size();
    for( int i = 0; i < colors.size(); i++){
        ofSetColor(colors[i]);
        ofDrawRectangle(_x, _y+h*i, w, h);
        ofSetColor(255);
        ofDrawBitmapString(classes[i], _x, _y+h*i+14);       
    }
}

vector<ofPolyline> ofxOpenCvDnnSegmentation::getBlobs(int _class_number, float _threshold)
{
    targetColor = colors[_class_number];
    contourFinder.setTargetColor(targetColor, true ? TRACK_COLOR_HS : TRACK_COLOR_RGB);
    contourFinder.setThreshold(_threshold);
    contourFinder.findContours(image_segmented);

    vector<ofPolyline> p;
    cout << contourFinder.size() << endl;
    if( contourFinder.size() > 0){
        p = contourFinder.getPolylines();
        
    }
    return p;
    
}

void ofxOpenCvDnnSegmentation::drawClass(int _class_id, int _alpha,
                                         int _x, int _y, int _w, int _h)
{
    ofSetLineWidth(3);
    int i = _class_id;
    vector<ofPolyline>p = getBlobs(i,0);
    for( int j = 0; j < p.size(); j++ ){
        ofPushMatrix();
        ofTranslate(_x, _y);
        ofScale(_w/(float)network_width,_h/(float)network_height);
        ofSetColor(ofColor::white);
        p[j].draw();
        
        ofSetColor(colors[i],_alpha);
        ofBeginShape();
        for( int k = 0; k < p[j].getVertices().size() ; k++){
            ofVertex(p[j].getVertices()[k].x,p[j].getVertices()[k].y);
        }
        ofEndShape();
        
        ofSetColor(ofColor::white);
        ofDrawBitmapString(classes[i], p[j].getBoundingBox().getCenter());
        ofPopMatrix();
    }

}

void ofxOpenCvDnnSegmentation::draw(int _x, int _y, int _w, int _h)
{
    image_segmented.draw(_x, _y, _w, _h);
}
                                
