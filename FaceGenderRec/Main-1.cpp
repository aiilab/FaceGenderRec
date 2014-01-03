// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

// Include the rest of our code!
#include "detectObject.h"       // Easily detect faces or eyes (using LBP or Haar Cascades).
#include "preprocessFace.h"     // Easily preprocess face images, for face recognition.
#include "recognition.h"     // Train the face recognition system and recognize a person from an image.

#include <io.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <string>

using namespace std;
using namespace cv;
using std::vector;
void getTrainingSet(string datapath, map<string, int>labelmap);
//void getTrainingSet(string datapath);
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2);
void initCamera(VideoCapture &videoCapture, int cameraNumber);
// The Face Recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).
const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";
//const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";


// Sets how confident the Face Verification algorithm should be to decide if it is an unknown person or a known person.
// A value roughly around 0.5 seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
// conditions, and if you use a different Face Recognition algorithm.
// Note that a higher threshold value means accepting more faces as known people,
// whereas lower values mean more faces will be classified as "unknown".
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;


// Cascade Classifier file, used for Face Detection.
//const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
//const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.
const bool preprocessLeftAndRightSeparately = true;   // Preprocess left & right sides of the face separately, in case there is stronger light on one side.

// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
const int faceWidth = 100;
const int faceHeight = faceWidth;

vector<Mat> images;
vector<int> labels;
Mat gray;

//ofstream flabel;	
 Rect faceRect;  // Position of detected face.
 Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
 Point leftEye, rightEye;    // Position of the detected eyes.
void main()
{
	
	CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);
	VideoCapture captureDevice; 
	initCamera(captureDevice,0);
	namedWindow("Video",1);
	Mat faceimg;
	//labeledLFW 包含：文件名+标记  eg: Aaron_Eckhart_0001.jpg 1 
	/*map<string, int> labelSet;			//文件名和标记一 一映射
//	flabel.open("labelfile.txt");
		//labeledLFW 包含：文件名+标记  eg: Aaron_Eckhart_0001.jpg 1 
	string labelSetFile = "labelfile.txt";
	ifstream fLFW;
	fLFW.open(labelSetFile.c_str());
	string name;
	int label;
	while (fLFW>>name>>label)
	{
		labelSet[name] = label;
	}	*/

	//getTrainingSet(DataPath,labelSet);
	//getTrainingSet(DataPath);
	
	/*Mat faceimg=images[2];
	imshow("faceimg",faceimg);
	cv::waitKey(10);*/
  	
	/*for (unsigned i = 0; i< images.size(); i++)
	{	   
		 Mat faceimg = images[i];
		 Rect faceRect;  // Position of detected face.
		 Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
		 Point leftEye, rightEye;    // Position of the detected eyes.
		 Mat preprocessedFace = getPreprocessedFace(faceimg, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye,		&searchedRightEye);  	   
	
		/* if (preprocessedFace.data)
		 {
			countnum++;
			cout<<"the num is "<<countnum<<endl;
		 }	*/ 		   
	 

		

	
			/*以下程序用来测试算法列表
				vector<String> algorithms;
				Algorithm::getList(algorithms);
				cout << "Algorithms: " << algorithms.size() << endl;
				for (size_t i=0; i < algorithms.size(); i++)
				 cout << algorithms[i] << endl;
			*/
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).	
	    /***********************************构造并训练分类器************************************************************/
		//cout<<"the label size is"<<labels.size()<<endl;

   	//	Ptr<FaceRecognizer> model = learnCollectedFaces(images,labels,"FaceRecognizer.Fisherfaces");
	  //  Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
		Ptr<FaceRecognizer> model=createFisherFaceRecognizer();
	//	model->train(images,labels);
	//	cout<<"Trainning is completed!"<<endl;
	//	model->save("Fisherfaces.yml");
		model->load("D:\\GitHub\\FaceGenderRec\\FaceGenderRec\\Fisherfaces2nd.yml");
		int male =0;
		int female=0;
		int frame = 0;
		int divide = 5;  //每多少帧显示一次
	//	string testpath="C:\\Users\\Eric\\Desktop\\FaceDataBase\\GenderTest\\img359.jpg";
		while(true)
		{
			
			captureDevice>>faceimg;
			frame++;
			Mat preprocessedFace = getPreprocessedFace(faceimg, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye,		&searchedRightEye);  
		
			if (preprocessedFace.rows==faceWidth)
			{
	
			//	cout<<preprocessedFace.channels()<<endl;
				rectangle(faceimg,faceRect,cvScalar(0, 255, 0, 0), 1, 8, 0);	
			//	cv::cvtColor(preprocessedFace,gray,CV_BGRA2GRAY);
			//	namedWindow("img");
			//	imshow("img",testSample);
			//	waitKey();
				int predict =1;	
				predict = model->predict(preprocessedFace);
				
				if (predict==0)
				{
					male++;
					//putText(faceimg,"Male",cvPoint(faceRect.x+faceRect.width,faceRect.y),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,0,255),2,8,false);
				}
				else
				{
					female++;
					//putText(faceimg,"Female",cvPoint(faceRect.x+faceRect.width,faceRect.y),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2,8,false);
				}
				if (frame%divide==0)
				{
					frame=0;
					if(female>male)
					{
						putText(faceimg,"Female",cvPoint(faceRect.x+faceRect.width,faceRect.y),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2,8,false);
					}
					else
						putText(faceimg,"Male",cvPoint(faceRect.x+faceRect.width,faceRect.y),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,0,255),2,8,false);
				}

			}
				imshow("video",faceimg);
				if(waitKey(20)==27)		//Waits for a pressed key:ESC
				{
					break;
				}
		}				
}
/****************************************************************************************/
/*功能：获取训练集，得到 Vector<Mat>images 和 Vector<int>labels
  输入：数据库地址，标签map
  返回值：无                                                                              */
/****************************************************************************************/
void getTrainingSet(string datapath, map<string, int>labelmap)
{

	struct _finddata_t FileInfo;
	string path=datapath+ "\\*.jpg";
	long Handle = _findfirst(path.c_str(),&FileInfo);

	if (Handle == -1L)    
	{        
		cerr << "can not match the folder path" << endl;        
		exit(-1);    
	}    
	do
	{   
		string name=FileInfo.name;			//获取文件名
	
		images.push_back(imread(datapath+"\\"+name,CV_LOAD_IMAGE_GRAYSCALE));		
		labels.push_back(labelmap[name]);
		cout<< "already pushed "<< name <<" label: " <<labelmap[name]<<endl;		
	}while (_findnext(Handle, &FileInfo) == 0);
	_findclose(Handle);  
	cout<< "the images size is "<<images.size() <<"\n the labels size is "<< labels.size()<<endl;
}

/****************************************************************************************/
/*功能：获取训练集，得到 Vector<Mat>images 和 Vector<int>labels
  输入：数据库地址，标签map
  返回值：无                                                                              */
/****************************************************************************************/
/*void getTrainingSet(string datapath)
{
	
	int num=0;
	struct _finddata_t FileInfo;
	string path=datapath+ "\\*.jpg";
	long Handle = _findfirst(path.c_str(),&FileInfo);
//	cout<<Handle<<endl;
	if (Handle == -1L)    
	{        
		cerr << "can not match the folder path" << endl;        
		exit(-1);    
	}    
	do
	{   
		string name=FileInfo.name;			//获取文件名
	//	cout<<name<<endl;
		images.push_back(imread(datapath+"\\"+name));	
		string a=datapath+"\\"+name;
		cout<<a;
		namedWindow("face");
		imshow("face",imread(datapath+"\\"+name));
		waitKey();
		cout << "Please input class: 男:0 / 女：1： ";
		cin>>num;
		labels.push_back(num);
		flabel <<FileInfo.name<<" "<< num<<endl;
	}while (_findnext(Handle, &FileInfo) == 0);
	_findclose(Handle);  
	cout<< "the imges size is "<<images.size() <<"\n the labels size is "<< labels.size()<<endl;
}*/
/****************************************************************************************/
/*功能：初始化检测器，加载分类器
  输入：CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2
  返回值：无                                                                              */
/****************************************************************************************/
// Load the face and 1 or 2 eye detection XML classifiers.

void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // Load the Face Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\haarcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
        // Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
        //exit(1);
    }
    else
        cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}
// Get access to the webcam.
void initCamera(VideoCapture &videoCapture, int cameraNumber)
{
    // Get access to the default camera.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        videoCapture.open(cameraNumber);
	} catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) 
	{
        cerr << "ERROR: Could not access the camera!" << endl;
		system("pause");
		exit(1);
	 }	
    cout << "Loaded camera " << cameraNumber << "." << endl;
}

