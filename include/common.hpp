#pragma once

#include <iostream>
#include <cmath>

struct Point3f{
    float x;
    float y;
    float z;
};

struct Point2f{
    float x;
    float y;

};

struct Vec3f{
    float x;
    float y;
    float z;
    float norm = std::sqrt(x*x + y*y + z*z);
};

struct Vec2f{
    float x;
    float y;
    float norm = std::sqrt(x*x + y*y);

};