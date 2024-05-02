/**
 * @file view.h
 * @author Bohan Cao (2110313@mail.nankai.edu.cn)
 * @brief 
 * @version 0.0.1
 * @date 2023-05-28
 * 
 * @copyright Copyright (c) 2023
 */
#ifndef HYPER_VIEW_H_
#define HYPER_VIEW_H_

#include <string>
#include <random>
#include <cmath>

#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

#include "space.h"

namespace Hyp {

class StdPiece {
public:
    #define k (Kokkos::sqrt(2*Kokkos::sqrt(2)/3) * space->t)
    /// @brief Init A, B, C
    /// @param space 
    inline StdPiece(Space *space, int n): data{
        Point(k, 0, space),
        Point(-k/2, k*Kokkos::sqrt(3)/2, space),
        Point(-k/2, -k*Kokkos::sqrt(3)/2, space)
    }, points(n, PointOnPiece(Point(0, 0, space), {
        Point(k, 0, space),
        Point(-k/2, k*Kokkos::sqrt(3)/2, space),
        Point(-k/2, -k*Kokkos::sqrt(3)/2, space)
    })) {
        for (int i=0; i<points.size(); i++) {
            int s = Kokkos::sqrt(points.size());
            points[i].m = (i/s+2)*k*s/points.size();
            points[i].n += (i%s-1)*k/points.size();
        }
    }

    std::vector<PointOnPiece> points;
    PieceData data;
    int step_num = 0;
};

class View {
public:
    View(Space *space);
    inline void equip(StdPiece *piece, StdPiece *piece2, int i, int rnd) {
        pieces[i] = piece; rnds[i] = rnd; pieces2[i] = piece2;
    }
    void update(int h, int f);
    void compute(int h, int f, double *data);

    // data
    StdPiece *pieces[8];
    int rnds[8];
    PieceData data[8];
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    std::uniform_real_distribution<double> dis01;
    std::normal_distribution<double> gau_dis;

    // parmeter (later 2 for MD)
    double range, T, gamma, dt;
    StdPiece *pieces2[8];
};

class FullView {
public:
    FullView(Space *space);
    inline void equip(StdPiece *piece, int i, int rnd) {
        pieces[i] = piece; rnds[i] = rnd;
    }
    void dump(std::string name);

    // data
    StdPiece *pieces[16];
    int rnds[16];
    PieceData data[16];
};

inline double energy(double dis) {
    if (dis > 2.5)
        return 0;
    // (1/distance)^6
    double dis6 = std::pow(1/dis, 6);
    double trunc6 = std::pow(1/2.5, 6);
    return 4 * dis6 * (dis6-1) - 4 * trunc6 * (trunc6-1);
}

inline double part_energy(double dis) {
    if (dis > 2.5)
        return 0;
    // (1/distance)^6
    double dis6 = std::pow(1/dis, 6);
    return 24 * dis6 * (2*dis6-1) / dis;
}

int get_area(Point p);
Line line_of_theta(Point p, double theta);
double angle_of(Point a, Point b);

}

#endif // HYPER_SPACE_H_