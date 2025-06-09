#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>
 
using namespace cv;
using namespace std;

//lifetime for lost bacteria
const int frame_lifetime = 10;
//average area of bacteria to assign IDs
int avg_area = 100;
//distance threshold to consider two objects as the same
const int dist_threshold = 10;

// Read each video file in the given directory 
vector<string> getFilesPath(DIR *video_directory, string dir){
    vector<string> video_files;
    string file;

    struct dirent *dirp;
    struct stat filestat;

    while ((dirp = readdir(video_directory))){
        file = dirp->d_name;
        
        //skip directory
        if (S_ISDIR( filestat.st_mode ))
            continue;
        if (file == "." || file == "..")
            continue;   
        cout<<file<<endl;
        video_files.push_back(dir + "/" + file);
        
    }
    closedir(video_directory);
    return video_files; 
}

/*
    Update counting statistics and average area of detected mass 
    Parameters:
        areas - vector of areas of objects
        old_mc - vector of old mass centers
        max_detected - reference to maximum number of detected bacteria
        sum_detected - reference to acumulator of detected bacteria
*/
void updateStats(vector<double> areas, vector<vector<Point2f>> move_mc ,int& max_detected, int& sum_detected){
    
    //calculate current average area
    double new_avg = 0;
    for (size_t i = 0; i < areas.size(); i++){
        new_avg += areas[i];
    }
    if(areas.size() > 0){
        new_avg /= areas.size();
        //update global average area
        avg_area = new_avg;//(avg_area + new_avg) / 2;
    }

    //update maximum detected elements
    if (move_mc.size() > max_detected)
        max_detected = move_mc.size();
    
    //update average detected elements
    sum_detected += move_mc.size();
}


/*
    Partial Contour extraction for ellipse fitting
    Parameters:
        img - current frame
        mass_centers - vector to store computed centers of mass for each contour
        areas - vector to store area of each contour
*/
void ellipse_fitting(Mat& img,vector<Point2f>& mass_centers, vector<double>& areas){
    vector<vector<Point>> contours;
    RotatedRect ellipse;
    Point2f center;

    //find full defined contours
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //count IDs based on fitting avg_area within area
    int n_size;
    areas.clear();
    
    //filter contours by area and fit ellipse
    for (size_t i = 0; i < contours.size(); i++){
        areas.push_back(contourArea(contours[i]));

        //check for enough points to fit an ellipse
        if (contours[i].size() >= 5) {
            ellipse = fitEllipse(contours[i]);
            //compute center of mass
            Moments mu = moments(contours[i]);

            //add center of mass n times according to area
            n_size = areas[i] / (avg_area*1.5);
            if (n_size < 1)
                n_size = 1; //minimum ID count is 1

            for (int j = 0; j < n_size; j++){
                if (mu.m00 != 0){
                    center = Point2f(mu.m10/mu.m00, mu.m01/mu.m00);
                    mass_centers.push_back(center);
                }else
                    mass_centers.push_back(ellipse.center);
            }
        }
    }
}


/*
    Delete objects from image based on contour area
    Parameters:
        img - current frame
        min_area - minimum area of countour to consider as bacteria
        max_area - maximum area of countour to consider as bacteria
    Returns:
        new_img - image with removed objects
*/
Mat remove_objects(Mat& img, double min_area, double max_area){
    //find all objects
    Mat img_labels, img_stats, img_centroids;
    int area;
    int nlabels = connectedComponentsWithStats(img,img_labels,img_stats,img_centroids);
    
    Mat new_img = img.clone();

    //clean each object (excluding background: label 0)
    for (int i = 1; i < nlabels; i++){
        //get object area
        area = img_stats.at<int>(i, CC_STAT_AREA);
        //remove object if area is out of bounds
        if (area < min_area || area > max_area){
            //get object bounding box
            Rect r = Rect(img_stats.at<int>(i, CC_STAT_LEFT), img_stats.at<int>(i, CC_STAT_TOP),
                          img_stats.at<int>(i, CC_STAT_WIDTH), img_stats.at<int>(i, CC_STAT_HEIGHT));
            //remove object from image
            new_img(r).setTo(Scalar(0,0,0));
        }
    }
    
    return new_img;
}



/*
    Compare old and new frames to find new objects, updating history of missed bacteria
    Parameters:
        move_mc - vector of displaced mass centers
        new_mc - vector of new mass centers
        lifetime - vector of lifetimes for each mass center [0]-index [1]-lifetime
*/
void compare_frames(vector<vector<Point2f>>& move_mc, vector<Point2f>& new_mc, vector<array<int,2>>& lifetime){
    
    int matched [move_mc.size()] = {0}; //array to store matched bacteria IDs
    bool match;

    //check for new objects
    for (size_t i = 0; i < new_mc.size(); i++){
        match = false;
        for (size_t j = 0; j < move_mc.size(); j++){  
            //if distance between centers is less than 10 pixels, consider it as the same object
            if (norm(new_mc[i] - move_mc[j].back()) < dist_threshold){
                match = true;
                //add displacement
                move_mc[j].push_back(new_mc[i]);
                matched[j] = 1;
                break;
            }
        }
        
        //add new mass center
        if (!match)
            move_mc.push_back(vector<Point2f>{new_mc[i]});
    }

    //update lifetime vector
    int move_size = move_mc.size();
    for(int j = 0; j < move_size; j++){
        match = false;
        for(size_t k = 0; k < lifetime.size(); k++){
            
            if (matched[j] == 1 && j == lifetime[k][0]){
                lifetime[k][1] = 0;//reset lifetime for matched bacteria
                match = true;
                break;
            }else if (matched[j] == 0 && j == lifetime[k][0]){
                //increment lifetime for unmatched bacteria
                lifetime[k][1]++;
                //if lifetime exceeds threshold, remove bacteria
                if (lifetime[k][1] > frame_lifetime){
                    move_mc.erase(move_mc.begin() + j);
                    lifetime.erase(lifetime.begin() + k);
                }
                match = true;
                break;
            }
        }
        if (!match){
            //add new bacteria to lifetime vector
            lifetime.push_back({j, 0});
        }
    }

    //update new mass centers
    new_mc.clear();
}


/*
    Draw ellipses and centers of mass on the image
    Parameters:
        img - current frame
        mass_centers - vector of mass centers

*/
void drawElements(Mat& img, vector<vector<Point2f>> move_mc){
    
    Point2f prev_center = Point2f(0,0);
    Point2f label;
    int same_mass = 0; //counter for shared mass centers

    for (size_t i = 0; i < move_mc.size(); i++){
        //draw center of mass
        circle(img, move_mc[i].back(), 3, Scalar(0, 0, 255), -1);
        //draw label
        
        //write ID for shared mass centers 
        if(i > 0 && prev_center ==  move_mc[i].back()){
            //displace label to avoid overlap
            label = Point2f(move_mc[i].back().x + (same_mass)*5, move_mc[i].back().y + 7);
            putText(img, "*", label, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            same_mass++;
        }else{
            same_mass = 0;
            putText(img, to_string(i+1), move_mc[i].back(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
        prev_center = move_mc[i].back();
    }

    //draw displacement of mass centers
    for (size_t i = 0; i < move_mc.size(); i++){
        for (size_t j = 0; j < move_mc[i].size() - 1; j++){
            //draw line between old and new mass centers
            if (j > 0)
                line(img, move_mc[i][j-1], move_mc[i][j], Scalar(255, 0, 0), 2);
        }
    }

    //draw detected bacteria count
    putText(img, to_string(move_mc.size()) + " bacterias", Point(10,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
    imshow("Labeled Bacteria",img);  
}



int main(int argc, char** argv ){
    if ( argc != 3 )
    {
        printf("Arguments: 1. Video path 2. Video index (0 for all)\n");
        return -1;
    }

    int frame_count = 0;
    int maxvideos;
    vector<string> video_files;
    VideoCapture capture;
    int video_index = stoi(argv[2]);

    //read video files
    DIR *video_dir = opendir(argv[1]);
    if (video_dir == NULL) {
        cout<<"Cannot open directory\n"<<endl;
        return 1;
    }
    video_files = getFilesPath(video_dir, argv[1]);

    //set total number of videos 
    if(video_index > 0 && video_index < video_files.size()){
        capture.open(video_files[video_index-1]);
        maxvideos = 1;
    }else if(video_index == 0){
        capture.open(video_files[0]);
        maxvideos = video_files.size();
    }else{
        cout<<"Invalid video index\n"<<endl;
        return 1;
    }
    
    if (!capture.isOpened()) {
        cerr << "Cannot initialize video" << endl;
        return 1;
    }

    //initialize background subtractor
    int history = 200; //frames to build background
    double dist_threshold = 12.0;//100.0; //distance between pixel and background model
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2(history,dist_threshold,false);


    //initialize thresholds
    int min_area = 6;       //minimum area of contour to consider as bacteria
    int max_area = 10000;   //maximum area of contour to consider as bacteria

    Mat img_input, gray, bkg_mask, img_clean;       //frame image processes
    vector<array<int,2>> lifetime;                        // missed elements lifetime
    vector<Point2f> new_mc;                         //center of mass for detected bacterias
    vector<vector<Point2f>> move_mc;                //displacement of mass centers
    vector<double> areas;                           //areas for detected elements                   
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11)); //kernel for morphological operations
    int max_detected = 0;   //maximum number of detected elements
    int sum_detected = 0;   //total number of detected elements
    auto key = 0;
    cout << "Press 's' to stop:" << endl;
    cout << "Press 'n' to next video:" << endl;
    
    //--------------------------------start video and frame capture
    for(int i = 1; i<=maxvideos; i++){
        while (key != 's') {
            // Capture frame-by-frame
            capture >> img_input;
            if (img_input.empty()) 
                break;

            //----Preprocessing
            cvtColor(img_input,gray, COLOR_BGR2GRAY);
            GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_REPLICATE);
            
            //----Background subtraction
            pBackSub->apply(gray, bkg_mask);
            imshow("Background Mask", bkg_mask);
            
            //----Mass characterization
            //filter bacteria by area
            img_clean = remove_objects(bkg_mask, min_area, max_area);
            //fill remaining elements
            morphologyEx(img_clean, img_clean,MORPH_CLOSE,kernel, Point(-1,-1), 1);
            //2nd filter after possibly connect small elements
            img_clean = remove_objects(img_clean, min_area, max_area);
            imshow("Closed Mask", img_clean);
            ellipse_fitting(img_clean, new_mc, areas);
            
            //----Match elements between frames
            compare_frames(move_mc, new_mc, lifetime);
            //----draw elements in frame
            drawElements(img_input, move_mc);

            //Update counting statistics
            updateStats(areas, move_mc, max_detected, sum_detected);

            key = waitKey(0);
            if(key == 'n')
                break;
            frame_count++;
        }

        printf("-----------RESULTS------------\n");
        printf("|       Frames =    %d\n", frame_count);
        printf("| Max detected =    %d\n", max_detected);
        printf("|Avg. detected =    %d\n", sum_detected / frame_count);

        //load next video
        if(maxvideos > 1 && i < maxvideos){
            capture.open(video_files[i]);
            frame_count = 0;
            avg_area = 100;
            max_detected = 0;
            sum_detected = 0;
        }      
    }

    destroyAllWindows();
    capture.release();

    return 0;
}