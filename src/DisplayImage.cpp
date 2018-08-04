
/*
 * DisplayImage.c
 *
 *  Created on: Aug 2, 2018
 *      Author: freaf87
 */
#include "DisplayImage.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include "types_c.h"
#include "highgui_c.h"
#include <unordered_map>
#include <vector>
#include <limits>
#include <algorithm>
#include <math.h>

#define MEAS_TIME 1
#define DEBUG 0
#define PI 3.14159265

using namespace cv;
using namespace std;

/* Sort Vectors helper class*/
struct SortXClass {
   bool operator() (Point2f pt1, Point2f pt2) { return (pt1.x < pt2.x);}
} sortXCoord;

struct SortYClass {
   bool operator() (Point2f pt1, Point2f pt2) { return (pt1.y < pt2.y);}
} sortYCoord;

/* Sort Maps helper class */
// Declaring the type of Predicate that accepts 2 pairs and return a bool
typedef std::function<bool(std::pair<int, Point2f>, std::pair<int, Point2f>)> ComparatorX;
// Defining a lambda function to compare two pairs. It will compare two pairs using second field
ComparatorX compFunctorX = [](std::pair<int, Point2f> elem1 ,std::pair<int, Point2f> elem2)
{
   return elem1.second.x < elem2.second.x;
};

// Declaring the type of Predicate that accepts 2 pairs and return a bool
typedef std::function<bool(std::pair<int, Point2f>, std::pair<int, Point2f>)> ComparatorY;
// Defining a lambda function to compare two pairs. It will compare two pairs using second field
ComparatorY compFunctorY = [](std::pair<int, Point2f> elem1 ,std::pair<int, Point2f> elem2)
{
   return elem1.second.y < elem2.second.y;
};


void thinning(const cv::Mat& src, cv::Mat& dst);
void thinningIteration(cv::Mat& img, int iter);
bool isVertexConnected(Mat& m, Point vertex1, Point vertex2, int *distance, bool debug);

class Graph
{
   unordered_map<int, unordered_map<int, int>> vertices;

public:
   void add_vertex(int name, const unordered_map<int, int>& edges)
   {
      // Insert the connected nodes in unordered map
      vertices.insert(unordered_map<int, const unordered_map<int, int>>::value_type(name, edges));
   }

   unordered_map<int, unordered_map<int, int>> getVertices(void)
   {
      return this->vertices;
   }

   void setVertices(unordered_map<int, unordered_map<int, int>> vertices)
   {
      this->vertices = vertices;
   }

   vector<int> shortest_path(int start, int finish)
   {
      // Second arguments -> distances
      // Find the smallest distance in the already in closed list and push it in -> previous
      unordered_map<int, int> distances;
      unordered_map<int, int> previous;
      vector<int> nodes; // Open list
      vector<int> path; // Closed list

      auto comparator = [&] (int left, int right) { return distances[left] > distances[right]; };

      for (auto& vertex : vertices)
      {
         if (vertex.first == start)
         {
            distances[vertex.first] = 0;
         }
         else
         {
            distances[vertex.first] = numeric_limits<int>::max();
         }

         nodes.push_back(vertex.first);
         push_heap(begin(nodes), end(nodes), comparator);
      }

      while (!nodes.empty())
      {
         pop_heap(begin(nodes), end(nodes), comparator);
         int smallest = nodes.back();
         nodes.pop_back();
#if 0
         std::cout << "Open list: ";
         for( std::vector<int>::const_iterator i = nodes.begin(); i != nodes.end(); ++i) std::cout << *i << ' ';
         std::cout << std::endl;
#endif
         if (smallest == finish)
         {
            while (previous.find(smallest) != end(previous))
            {
               path.push_back(smallest);
               smallest = previous[smallest];
#if 0
               std::cout << "Closed list: ";
               for( std::vector<int>::const_iterator i = path.begin(); i != path.end(); ++i) std::cout << *i << ' ';
               std::cout << std::endl;
#endif
            }

            break;
         }

         if (distances[smallest] == numeric_limits<int>::max())
         {
            break;
         }

         for (auto& neighbor : vertices[smallest])
         {
            int alt = distances[smallest] + neighbor.second;
            if (alt < distances[neighbor.first])
            {
               distances[neighbor.first] = alt;
               previous[neighbor.first] = smallest;
               make_heap(begin(nodes), end(nodes), comparator);
            }
         }
      }

      return path;
   }

};
/* Global Variables */
Mat src;

int main()
{
   /* Initializing and Solving the Maze*/
   int init_node = 9;
   int dest_node = 7;
   Graph g;
   int seq = 0;

#ifdef MEAS_TIME
   const int64 start = getTickCount();
#endif
   //namedWindow("sourceWindow", WINDOW_NORMAL );
   src = cv::imread("/home/freaf87/Workspaces/eclipse-workspace/DisplayImage/image/maze1.png");
   if (!src.data)
   {
      cout << "Input file not found !!!" << endl;
      return -1;
   }

   Mat grayscale, skel;
   cvtColor(src, grayscale, CV_BGR2GRAY);
   thinning(grayscale, skel);

   /* Corner detection */
   std::vector< cv::Point2f > corners;
   goodFeaturesToTrack(skel, corners, 500 , 0.01, 10, Mat(), 3 , false, 0.04);
#ifdef DEBUG
   cout << "** Number of corners detected: "<<corners.size()<<endl;
#endif
   for( size_t i = 0; i < corners.size(); i++ )
   {
#if 1
      char label[12];
      sprintf(label, "%d", (int)i);
      putText(src, label, corners[i], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
      printf("Vertex %d: P(%d, %d)\n", (int)i, (int)corners[i].x, (int)corners[i].y);

#else
      circle( src, corners[i], 2, cv::Scalar( 255. ), -1 );
#endif
   }

   imshow ("Original Image with labeled Corners",src);

   resizeWindow("sourceWindow", 500, 500);
#if DEBUG
   printf("max count = %d \n", (int)corners.size());
   imshow("Skeleton Image", skel );
#endif

   for( int i = 0; i < (int)corners.size(); i++ )
   {
      unordered_map<int, int> map;
      for(int j = 0; j < (int)corners.size(); j++)
      {
         int dist = 0;
         bool debug = false;
#if 0
         if(i==0 && j==4) debug = true;
         else debug = false;
#endif

         bool isCon = isVertexConnected(skel,corners[i],corners[j],&dist, debug);

#if DEBUG
         if(isCon == true) cout << i << " <--> " << j << endl;
         else cout << i << " --- " << j << endl;
#endif
         if (dist > 0) map.insert({j,dist});
      }
#if DEBUG
      cout << "g.add_vertex(" << i << ", {" ;
      for(auto it = map.begin(); it !=  map.end(); it++) cout << "{" << it->first << "," << it->second << "}";
      cout << "}" << endl;
#endif
      g.add_vertex(i,map);
   }

   /* Search PATH */
   vector<int> FoundPATH = g.shortest_path(init_node, dest_node);
   FoundPATH.push_back(init_node); /* Append last element to List */
   reverse(FoundPATH.begin(),FoundPATH.end()); /* improve readibility*/

   vector<int> FoundPATHResult = FoundPATH;
   if(FoundPATH.empty())
   {
      cout << "Path not found between vertex " << init_node << " and vertex " << dest_node << endl;
      return -1;
   }

   /* CLEAN UP GRAPH AND FOUNDPATH */
   seq = 0;
   int offset = 0;
   unordered_map<int, unordered_map<int, int>> ConnectionGraph;
   ConnectionGraph = g.getVertices();

   for (int vertex : FoundPATH)
   {
      try
      {

         int nextVertex = FoundPATH.at(seq+1);

         unordered_map<int, int>& map1 = ConnectionGraph[vertex];
         unordered_map<int, int>& map2 = ConnectionGraph[nextVertex];

#if 0
         std::cout << "map1 contains:";
         for ( auto it = map1.begin(); it != map1.end(); ++it )
            std::cout << " " << it->first << ":" << it->second;
         std::cout << std::endl;

         std::cout << "map2 contains:";
         for ( auto it = map2.begin(); it != map2.end(); ++it )
            std::cout << " " << it->first << ":" << it->second;
         std::cout << std::endl;
         cout << "**********************************************" << endl;
#endif

         for ( auto it = map1.begin(); it != map1.end(); ++it ) /* For all maps1 ' s elements */
         {
            if (map2.find(it->first) != map2.end()) /* Common Connection found */
            {
               //int direction;
               int CommonNode = it->first;
               int middlePoint;
               map<int, Point2f> VertCoord;
               VertCoord.insert(make_pair(vertex    , corners[vertex]    ));
               VertCoord.insert(make_pair(CommonNode, corners[CommonNode]));
               VertCoord.insert(make_pair(nextVertex, corners[nextVertex]));

#if DEBUG
                  cout << "Unsorted points in vertical direction: "<< endl;
                  for (std::pair<int, Point2f> element : VertCoord)
                     cout << element.first << " :: " << element.second << endl;
#endif

               /* "3 way connection" found. First find middle Point and decide if node is redundant or T-intersection */
               float XVals[] = {corners[vertex].x, corners[CommonNode].x, corners[nextVertex].x};
               float YVals[] = {corners[vertex].y, corners[CommonNode].y, corners[nextVertex].y};


               /* Get direction & sort to get middle point*/
               if(   ((*std::max_element(XVals,XVals+3)) - ((*std::min_element(XVals,XVals+3))))
                   > ((*std::max_element(YVals,YVals+3)) - ((*std::min_element(YVals,YVals+3))))
               )
               {
                  //direction = horizontal;
                  std::set<std::pair<int, Point2f>, ComparatorX> VertexXSort(VertCoord.begin(), VertCoord.end(), compFunctorX);
                  //TODO: Find a better way to get middle element
                  int count= 0;
                  for (std::pair<int, Point2f> element : VertexXSort)
                  {
                     if (count == 1)
                     {
                        middlePoint = (int)element.first;
                        break;
                     }
                     count++;
                  }
#if DEBUG
                  cout << "Sorted point in horizontal direction: "<< endl;
                  for (std::pair<int, Point2f> element : VertexXSort)
                     cout << element.first << " :: " << element.second << endl;

                  cout << "Middle point [" << middlePoint << "] will be used !!" << endl;
#endif

               }
               else
               {
                  //direction = vertical;
                  std::set<std::pair<int, Point2f>, ComparatorY> VertexYSort(
                        VertCoord.begin(), VertCoord.end(), compFunctorY);
                  //TODO: Find a better way to get middle element
                  int count= 0;
                  for (std::pair<int, Point2f> element : VertexYSort)
                  {
                     if (count == 1)
                     {
                        middlePoint = (int)element.first;
                        break;
                     }
                     count++;
                  }
#if DEBUG
                  cout << "Sorted point in vertical direction: "<< endl;
                  for (std::pair<int, Point2f> element : VertexYSort)
                     cout << element.first << " :: " << element.second << endl;

                  cout << "Middle point [" << middlePoint << "] will be used !!" << endl;
#endif
               }


               if (middlePoint == CommonNode) /* Continue processing only when both nodes are equal  to avoid double inclusion */
               {
                  unordered_map<int, int>& mapMiddlePoint = ConnectionGraph[middlePoint];

                  if(mapMiddlePoint.size() <= 2)
                  {
                     /* 2 Neighbors:
                      *    (1) delete in ConnectionGraph the key middlePoint and his Neighbors map
                      *    (2) delete in connected Points (map1, map2) the element where middlePoint appears
                      */

#if 0
                     std::cout << "map1 contains:";
                     for ( auto it = map1.begin(); it != map1.end(); ++it )
                        std::cout << " " << it->first << ":" << it->second;
                     std::cout << std::endl;
#endif
                     ConnectionGraph.erase(middlePoint);
                     map1.erase(middlePoint);
                     map2.erase(middlePoint);
                     FoundPATHResult.erase(std::remove(FoundPATHResult.begin(), FoundPATHResult.end(), middlePoint), FoundPATHResult.end());
                     cout << "Erase " << middlePoint << "from FoundPATHResult" << endl;
#if 0
                     std::cout << "map1 now contains:";
                     for ( auto it = map1.begin(); it != map1.end(); ++it )
                        std::cout << " " << it->first << ":" << it->second;
                     std::cout << std::endl;
#endif
                  }
                  else
                  {
                     /* greather than 2 Neighbors:
                      *    (1) Remove unwanted ConnectionGraph over map1 and map2
                      *    (2) insert middlePoint between vertex and nextVertex in FoundPath since this is a valid intersection
                      */
                     map1.erase(nextVertex);
                     map2.erase(vertex);
                     cout << middlePoint << " inserted at " << seq << endl;
                     FoundPATHResult.insert(FoundPATHResult.begin()+ (seq+1)+ offset, middlePoint);
                     cout << "Inserting " << middlePoint << " in FoundPATHResult" << endl;
                     offset++;
                  }
               }
            }
            else
            {
               /* Not found */
            }

         }

         seq++;
      }
      catch (std::out_of_range &)
      {
         cout << "Last index reached..." << endl;
      }

   }


   if(!FoundPATHResult.empty())
   {
      seq = 0;
      cout << "RESULT: " << endl;
      cout << "As initial node: " << init_node << endl;
      cout << "As goal node: " << dest_node << endl;


      for (int vertex : FoundPATHResult)
      {
         cout << "Solution path from goal sequence : " << seq << " Node : " << vertex << endl;
         seq++;
      }
   }



#ifdef MEAS_TIME
   const double timeSec = (getTickCount() - start) / getTickFrequency();
   cout << "CPU Time : " << timeSec * 1000 << " ms" << endl;
#endif


   printf("Done !!");


   cv::waitKey();
   return 0;
}

bool isVertexConnected(Mat& m, Point vertex1, Point vertex2, int *distance, bool debug)
{

   vector<cv::Point> Houghpts(2);
   vector<cv::Point> Inputpts(2);
   Inputpts[0] = vertex1;
   Inputpts[1] = vertex2;

   Point HoughP1, HoughP2;
   float angleInputPts, angleHoughPts;

   /* compute distance between original Vertex */
   double Vertexdist = sqrt((vertex2.x - vertex1.x) * (vertex2.x - vertex1.x) + (vertex2.y - vertex1.y) * (vertex2.y - vertex1.y));

   /* top left point, width, height */
   Rect croppedRect = Rect(min(vertex1.x, vertex2.x)-3, min(vertex1.y, vertex2.y)-3, (int)(abs(vertex1.x - vertex2.x))+6, (int)(abs(vertex1.y-vertex2.y))+6);
   Mat  croppedImage = m(croppedRect);
   rectangle(src, croppedRect, Scalar(0,0,255), 1);

#if 0
   cout << "(" << vertex1.x << "," << vertex1.y << ")" << "   " << "(" << vertex2.x << "," << vertex2.y << ")" << endl;
   cout << croppedRect.x << " " << croppedRect.y <<" " << croppedRect.width << " " << croppedRect.height << endl;
#endif

   Mat dst, cdst;
   Canny(croppedImage, dst, 50, 200, 3);
   cvtColor(dst, cdst, CV_GRAY2BGR);
   vector<Vec4i> lines;
   HoughLinesP(dst, lines, 1, CV_PI/180, 10, (int)(Vertexdist*2/3), 4 );
#if DEBUG
   if(debug)
      imshow("Debug Image", cdst );
#endif

   double maxDistance = 0;
   for( size_t i = 0; i < lines.size(); i++ )
   {
      Vec4i l = lines[i];
      line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
      double tmp = sqrt((l[2] - l[0]) * (l[2] - l[0]) + (l[3] - l[1]) * (l[3] - l[1]));

      if(maxDistance <= tmp)
      {
         maxDistance = max(maxDistance, tmp);
         /* Actualize HoughP1 and HoughP2 */
         HoughP1.x = l[0];
         HoughP1.y = l[1];
         HoughP2.x = l[2];
         HoughP2.y = l[3];
      }
   }

   Houghpts[0] = HoughP1;
   Houghpts[1] = HoughP2;

   /* sort Point vectors. Smaller x coordinate first */
   sort(Inputpts.begin(), Inputpts.end(), sortXCoord);
   sort(Houghpts.begin(), Houghpts.end(), sortXCoord);

   /* compute gradient of HoughLine and input Vertex */
   float gradHough,gradInputPoint;

   if((Houghpts[0].y -Houghpts[1].y) != 0 && (Houghpts[0].x -Houghpts[1].x) != 0) gradHough = (Houghpts[0].y -Houghpts[1].y)*1.0/(Houghpts[0].x -Houghpts[1].x)*1.0;
   else if((Houghpts[0].y -Houghpts[1].y) == 0 && (Houghpts[0].x -Houghpts[1].x) != 0) gradHough = 0;
   else if((Houghpts[0].y -Houghpts[1].y) != 0 && (Houghpts[0].x -Houghpts[1].x) == 0) gradHough = 1000000000000;
   else gradHough = -1;

   if((Inputpts[0].y -Inputpts[1].y) != 0 && (Inputpts[0].x -Inputpts[1].x) != 0) gradInputPoint = (Inputpts[0].y -Inputpts[1].y)*1.0/(Inputpts[0].x -Inputpts[1].x)*1.0;
   else if((Inputpts[0].y -Inputpts[1].y) == 0 && (Inputpts[0].x -Inputpts[1].x) != 0) gradInputPoint = 0;
   else if((Inputpts[0].y -Inputpts[1].y) != 0 && (Inputpts[0].x -Inputpts[1].x) == 0) gradInputPoint = 1000000000000;
   else gradInputPoint = -1;

   angleInputPts = atan(abs(gradInputPoint))* 180 / PI;
   angleHoughPts = atan (abs(gradHough))* 180 / PI;


#if DEBUG
   cout << "__________________________________"<< endl;
   cout << "Input Points: " << Inputpts[0] << "\t" << Inputpts[1] << endl;
   cout << "Hough Points: " << Houghpts[0] << "\t" << Houghpts[1] << endl;
   cout << "Angle:  " << angleInputPts <<  "\t" << angleHoughPts  << endl;
   cout << "Vertexdist = " << Vertexdist << "\t" << "HoughDis = " << maxDistance << endl;
   cout << "Distance error: " << abs(maxDistance - Vertexdist)/ Vertexdist << endl;
#endif

   if (abs(maxDistance - Vertexdist)/ Vertexdist < 0.15 &&  abs(angleHoughPts-angleInputPts) < (-0.0088*Vertexdist + 7.4))
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

