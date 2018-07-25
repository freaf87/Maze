/**
 * Code for thinning a binary image using Zhang-Suen algorithm.
 *
 * Author:  Nash (nash [at] opencv-code [dot] com)
 * Website: http://opencv-code.com
 */
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "types_c.h"
#include "highgui_c.h"
#include <cmath>


#define MEAS_TIME 1
#define DEBUG 0

using namespace cv;
using namespace std;

void thinning(const cv::Mat& src, cv::Mat& dst);
void thinningIteration(cv::Mat& img, int iter);
bool isVertexConnected(Mat& m, Point vertex1, Point vertex2, int *distance);
void dijkstra(int *graph, int src, int size);
void printSolution(int dist[], int n);
int minDistance(int dist[], bool sptSet[], int size);

/* Global Variables */
Mat src;

int main()
{
#ifdef MEAS_TIME
   const int64 start = getTickCount();
#endif
   //namedWindow("sourceWindow", CV_WINDOW_AUTOSIZE );
   src = cv::imread("/home/freaf87/Workspaces/eclipse-workspace/DisplayImage/image/maze.png");
   if (!src.data)
      return -1;

   Mat grayscale, skel;
   cvtColor(src, grayscale, CV_BGR2GRAY);
   thinning(grayscale, skel);

   /* Corner detection */
   std::vector< cv::Point2f > corners;
   goodFeaturesToTrack(skel, corners, 500 , 0.01, 50, Mat(), 3 , false, 0.04);
#ifdef DEBUG
   cout << "** Number of corners detected: "<<corners.size()<<endl;
#endif
   for( size_t i = 0; i < corners.size(); i++ )
   {
#if !DEBUG
      char label[12];
      sprintf(label, "%d", (int)i);
      putText(src, label, corners[i], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
      printf("Vertex %d: P(%d, %d)\n", (int)i, (int)corners[i].x, (int)corners[i].y);

#else
      circle( src, corners[i], 3, cv::Scalar( 255. ), -1 );
#endif
   }

   imshow ("Original Image with labelled Corners",src);
/* construct way Graph */
   int  start_vertex;

   int graph[(int)corners.size()][(int)corners.size()];
   memset(graph, 0, sizeof graph);

   start_vertex = 5;

   for( int i = 0; i < (int)corners.size()-1; i++ )
   {
      for(int j = i+1; j < (int)corners.size(); j++)
      {
         int dist = 0;
         isVertexConnected(skel,corners[i],corners[j],&dist);
         graph[i][j] = (int)dist;
         graph[j][i] = (int)dist;
      }
   }

   dijkstra( (int*)graph, start_vertex, (int)corners.size());

#if DEBUG
   printf("max count = %d \n", (int)corners.size());
#endif

#if DEBUG
   imshow("Skeleton Image", skel );
#endif

#ifdef MEAS_TIME
   const double timeSec = (getTickCount() - start) / getTickFrequency();
   cout << "CPU Time : " << timeSec * 1000 << " ms" << endl;
#endif

   printf("Done !!");
   cv::waitKey();
   return 0;
}

bool isVertexConnected(Mat& m, Point vertex1, Point vertex2, int *distance)
{
   static int i = 0;
   // top left point, width, height

   Rect croppedRect = Rect(min(vertex1.x, vertex2.x)-3, min(vertex1.y, vertex2.y)-3, (int)(abs(vertex1.x - vertex2.x))+6, (int)(abs(vertex1.y-vertex2.y))+6);
   Mat  croppedImage = m(croppedRect);
   rectangle(src, croppedRect, Scalar(0,0,255), 1);

   char label[12];
   sprintf(label, "%d", (int)i++);

#if 0
   cout << "(" << vertex1.x << "," << vertex1.y << ")" << "   " << "(" << vertex2.x << "," << vertex2.y << ")" << endl;
   cout << croppedRect.x << " " << croppedRect.y <<" " << croppedRect.width << " " << croppedRect.height << endl;
#endif

   Mat dst, cdst;
   Canny(croppedImage, dst, 50, 200, 3);
   cvtColor(dst, cdst, CV_GRAY2BGR);
   vector<Vec4i> lines;
   HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );

   double maxDistance = 0;
   for( size_t i = 0; i < lines.size(); i++ )
   {
      Vec4i l = lines[i];

      line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
      maxDistance = max(maxDistance, sqrt((l[2] - l[0]) * (l[2] - l[0]) + (l[3] - l[1]) * (l[3] - l[1])));
   }

   /* compute distance between original Vertex */
   double Vertexdist = sqrt((vertex2.x - vertex1.x) * (vertex2.x - vertex1.x) + (vertex2.y - vertex1.y) * (vertex2.y - vertex1.y));

   if (abs(maxDistance - Vertexdist)/ Vertexdist < 0.08)
   {
      *distance = (int)(Vertexdist);
      return true;
   }
   else
   {
      *distance = (int)0;
      return false;
   }
}

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * Parameters:
 *      im    Binary image with range = [0,1]
 *      iter  0=even, 1=odd
 */
void thinningIteration(cv::Mat& img, int iter)
{
   CV_Assert(img.channels() == 1);
   CV_Assert(img.depth() != sizeof(uchar));
   CV_Assert(img.rows > 3 && img.cols > 3);

   cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

   int nRows = img.rows;
   int nCols = img.cols;

   if (img.isContinuous()) {
      nCols *= nRows;
      nRows = 1;
   }

   int x, y;
   uchar *pAbove;
   uchar *pCurr;
   uchar *pBelow;
   uchar *nw, *no, *ne;    // north (pAbove)
   uchar *we, *me, *ea;
   uchar *sw, *so, *se;    // south (pBelow)

   uchar *pDst;

   // initialize row pointers
   pAbove = NULL;
   pCurr  = img.ptr<uchar>(0);
   pBelow = img.ptr<uchar>(1);

   for (y = 1; y < img.rows-1; ++y) {
      // shift the rows up by one
      pAbove = pCurr;
      pCurr  = pBelow;
      pBelow = img.ptr<uchar>(y+1);

      pDst = marker.ptr<uchar>(y);

      // initialize col pointers
      no = &(pAbove[0]);
      ne = &(pAbove[1]);
      me = &(pCurr[0]);
      ea = &(pCurr[1]);
      so = &(pBelow[0]);
      se = &(pBelow[1]);

      for (x = 1; x < img.cols-1; ++x) {
         // shift col pointers left by one (scan left to right)
         nw = no;
         no = ne;
         ne = &(pAbove[x+1]);
         we = me;
         me = ea;
         ea = &(pCurr[x+1]);
         sw = so;
         so = se;
         se = &(pBelow[x+1]);

         int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
               (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
               (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
               (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
         int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
         int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
         int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

         if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
            pDst[x] = 1;
      }
   }

   img &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 *      src  The source image, binary with range = [0,255]
 *      dst  The destination image
 */
void thinning(const cv::Mat& src, cv::Mat& dst)
{
   dst = src.clone();
   bitwise_not ( dst, dst );
   dst /= 255;         // convert to binary image

   cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
   cv::Mat diff;

   do {
      thinningIteration(dst, 0);
      thinningIteration(dst, 1);
      cv::absdiff(dst, prev, diff);
      dst.copyTo(prev);
   }
   while (cv::countNonZero(diff) > 0);

   dst *= 255;
}

// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(int dist[], bool sptSet[], int size)
{
   // Initialize min value
   int min = INT_MAX, min_index;

   for (int v = 0; v < size; v++)
     if (sptSet[v] == false && dist[v] <= min)
         min = dist[v], min_index = v;

   return min_index;
}

// A utility function to print the constructed distance array
void printSolution(int dist[], int n)
{
   printf("Vertex   Distance from Source\n");
   for (int i = 0; i < n; i++)
      printf("%d tt %d\n", i, dist[i]);
}

// Function that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
void dijkstra(int *graph, int src, int size)
{
   int dist[size];     // The output array.  dist[i] will hold the shortest
                     // distance from src to i

    bool sptSet[size]; // sptSet[i] will true if vertex i is included in shortest
                    // path tree or shortest distance from src to i is finalized

    // Initialize all distances as INFINITE and stpSet[] as false
    for (int i = 0; i < size; i++)
       dist[i] = INT_MAX, sptSet[i] = false;

    // Distance of source vertex from itself is always 0
    dist[src] = 0;

    // Find shortest path for all vertices
    for (int count = 0; count < size-1; count++)
    {
      // Pick the minimum distance vertex from the set of vertices not
      // yet processed. u is always equal to src in the first iteration.
      int u = minDistance(dist, sptSet, size);

      // Mark the picked vertex as processed
      sptSet[u] = true;

      // Update dist value of the adjacent vertices of the picked vertex.
      for (int v = 0; v < size; v++)

        // Update dist[v] only if is not in sptSet, there is an edge from
        // u to v, and total weight of path from src to  v through u is
        // smaller than current value of dist[v]
        if (!sptSet[v] && graph[u*size+v] && dist[u] != INT_MAX
                                      && dist[u]+graph[u*size+v] < dist[v])
           dist[v] = dist[u] + graph[u*size+v];
    }
    // print the constructed distance array
    printSolution(dist, size);
}
