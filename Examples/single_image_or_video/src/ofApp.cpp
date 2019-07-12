#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    img.load(ofToDataPath("sample.jpg"));
    ofSetWindowShape(img.getWidth(), img.getHeight());
    segmentation.setNetworkImageSize(512,256);
    
    segmentation.setup(ofToDataPath("dnn/Enet-model-best.net"),
                       ofToDataPath("dnn/classlist.txt"));
     segmentation.scale = 0.00392;
/*
    segmentation.setup(ofToDataPath("dnn/fcn8s-heavy-pascal.caffemodel"),
                       ofToDataPath("dnn/fcn8s-heavy-pascal.prototxt"),
                       ofToDataPath("dnn/pascal-classes.txt"));
    segmentation.scale = 1.0;
 */
    segmentation.update(img.getPixels());
    
   
}

//--------------------------------------------------------------
void ofApp::update(){
    if( video.isLoaded() ){
        video.update();
        if( video.isFrameNew() ){
            segmentation.update(video.getPixels());
           
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(0);
    ofSetColor(255);
    if( video.isLoaded() ){
        video.draw(0,0);
    }
    else{
        img.draw(0,0);
    }
    
    // set alpha to get background image(video/img.draw).
    // ofSetColor(ofColor::white, 200);
    // Simple and Easiest way to get the result.
    //segmentation.draw(0,0,ofGetWidth(), ofGetHeight());

    // you can draw a class separately ( i == 0: unlabeled class )
    for( int i = 1; i < segmentation.classes.size(); i++ ){
        segmentation.drawClass(i,150, // i: class id, 150: alpha value
                               0,0, ofGetWidth(), ofGetHeight());
    }
    
    // Here is to show a color palette. x, y, width, height
    segmentation.drawColorPalette(10, 10, 100, ofGetHeight()-20);
   
    // Image and Video file are available on this example
    ofDrawBitmapStringHighlight("Drag and Drop a image/video file",
                                 ofGetWidth()/2-100,
                                 ofGetHeight()-20);
   
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
}


//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

    if(dragInfo.files[0].find(".mov") != std::string::npos ||
       dragInfo.files[0].find(".mp4") != std::string::npos
       ){
        if( video.isLoaded() )video.closeMovie();
        video.load(dragInfo.files[0]);
        ofSetWindowShape(video.getWidth(), video.getHeight());
        video.play();
    }
    else if( dragInfo.files[0].find(".jpg") != std::string::npos ){
        if( video.isLoaded() )video.closeMovie();
        if( img.isAllocated() )img.clear();
        img.load(dragInfo.files[0]);
        img.setImageType(OF_IMAGE_COLOR);
        ofSetWindowShape(img.getWidth(), img.getHeight());
        segmentation.update(img.getPixels());
    }
    else{
        if( video.isLoaded() )video.closeMovie();
        ofSystemAlertDialog("Error: Invalid Extention File.");
    }
}
