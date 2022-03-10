
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include "Point.h"
#include "Cluster.h"
#include <omp.h>
#include <string>
#include <vector>
#include <c++/8/sstream>


using namespace std;
using namespace std::chrono;

int num_point = 10000;
int num_cluster = 5;
int max_iterations = 20;

vector<Point> init_point();
vector<Cluster> init_cluster(vector<Point> points);
void compute_distance(vector<Point> &points, vector<Cluster> &clusters);
double euclidean_dist(Point point, Cluster cluster);
bool update_clusters(vector<Cluster> &clusters);
void draw_chart_gnu(vector<Point> &points);


int main() {
    int num_thread = 2;
    omp_set_num_threads(num_thread);
    printf("Number of points %d\n", num_point);
    printf("Number of clusters %d\n", num_cluster);
    printf("Number of processors: %d\n", omp_get_num_procs());
    printf("Number of threads: %d\n", num_thread);

    srand(int(time(NULL)));


    vector<Point> points;
    points = init_point();
    vector<Cluster> clusters;
    clusters = init_cluster(points);


    double time_point1 = omp_get_wtime();


    bool conv = true;
    int iterations = 0;

    printf("Starting iterate...\n");

    //The algorithm stops when iterations > max_iteration or when the clusters didn't move
    while(conv && iterations < max_iterations){

        iterations ++;

        compute_distance(points, clusters);

        conv = update_clusters(clusters);

        printf("Iteration %d done \n", iterations);

    }

    double time_point2 = omp_get_wtime();
    double duration = time_point2 - time_point1;

    printf("Number of iterations: %d, total time: %f seconds, time per iteration: %f seconds\n",
           iterations, duration, duration/iterations);

    try{
        printf("Drawing the chart...\n");
        draw_chart_gnu(points);
    }catch(int e){
        printf("Chart not available, gnuplot not found");
    }

    return 0;


}

//Initialize num_point Points
vector<Point> init_point(){


    string fname = "/home/federico/CLionProjects/kmeans_cuda/datasets/2D_data_3.csv";
    vector<vector<string>> content;
    vector<string> row;
    string line, word;
    fstream file (fname, ios::in);
    if(file.is_open())
    {
        while(getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while(getline(str, word, ','))
                row.push_back(word);
            content.push_back(row);
        }
    }
    vector<Point> points(num_point);
    Point *ptr = &points[0];
    cout<<"Datapoints:"<<"\n";
    for(int i=0;i<content.size();i++)
    {
        cout<<content[i][0]<<","<<content[i][0]<<"\n";
        Point* point = new Point(std::stod(content[i][0]), std::stod(content[i][1]));
        ptr[i] = *point;
    }
    return points;

}

//Initialize num_cluster Clusters
vector<Cluster> init_cluster(vector<Point> points){

    vector<Cluster> clusters(num_cluster);
    Cluster* ptr = &clusters[0];
    cout<<"Clusters:"<<"\n";

    for(int i = 0; i < num_cluster; i++){
        int n = rand() % (int) num_point;
        Cluster *cluster = new Cluster(points[n].get_x_coord(), points[n].get_y_coord());
        cout<<points[n].get_x_coord()<<", "<<points[n].get_y_coord()<<"\n";

        ptr[i] = *cluster;

    }

    return clusters;
}

//For each Point, compute the distance between each Cluster and assign the Point to the nearest Cluster
//The distance is compute through Euclidean Distance
//The outer for is parallel, with private=min_distance, min_index, points_size, clusters_size and clustes while the
//vector of Points is shared. The amount of computation performed per Point is constant, so static thread scheduling was chosen

void compute_distance(vector<Point> &points, vector<Cluster> &clusters){

    unsigned long points_size = points.size();
    unsigned long clusters_size = clusters.size();

    double min_distance;
    int min_index;


#pragma omp parallel default(shared) private(min_distance, min_index) firstprivate(points_size, clusters_size)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < points_size; i++) {

            Point &point = points[i];

            min_distance = euclidean_dist(point, clusters[0]);
            min_index = 0;

            for (int j = 1; j < clusters_size; j++) {

                Cluster &cluster = clusters[j];

                double distance = euclidean_dist(point, cluster);

                if (distance < min_distance) {

                    min_distance = distance;
                    min_index = j;
                }

            }
            point.set_cluster_id(min_index);
            clusters[min_index].add_point(point);

        }
    }
}

double euclidean_dist(Point point, Cluster cluster){

    double distance = sqrt(pow(point.get_x_coord() - cluster.get_x_coord(),2) +
                           pow(point.get_y_coord() - cluster.get_y_coord(),2));

    return distance;
}

//For each cluster, update the coords. If only one cluster moves, conv will be TRUE
//A parallel for was chosen for each cluster with lastprivate=conv
bool update_clusters(vector<Cluster> &clusters){

    bool conv = false;

    for(int i = 0; i < clusters.size(); i++){
        conv = clusters[i].update_coords();
        clusters[i].free_point();
    }

    return conv;
}

//Draw point plot with gnuplot
void draw_chart_gnu(vector<Point> &points){

    ofstream outfile("data.txt");

    for(int i = 0; i < points.size(); i++){

        Point point = points[i];
        outfile << point.get_x_coord() << " " << point.get_y_coord() << " " << point.get_cluster_id() << std::endl;

    }

    outfile.close();
    system("gnuplot -p -e \"plot 'data.txt' using 1:2:3 with points palette notitle\"");
    remove("data.txt");

}

