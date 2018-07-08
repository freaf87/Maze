/*
 * DisplayImage.cpp
 *
 *  Created on: Jul 6, 2018
 *      Author: freaf87
 */
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include "DisplayImage.hpp"


using namespace cv;

int main( int argc, char** argv )
{
  Mat image;
  image = imread( argv[1], 1 );


  if( argc != 2 || !image.data )
    {
      printf( "No image data \n" );
      return -1;
    }

  namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
  imshow( "Display Image", image );

  /* Start solving here */





  waitKey(0);

  return 0;
}
