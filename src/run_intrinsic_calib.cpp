#include <iostream>

#include "camera.hpp"


int main(int argc, char **argv){

    std::string image_path = argv[1];

    camera cam;
    cam.calibrate(image_path, 30, 6, 8, true);


    return 0;
}