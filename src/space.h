/**
 * @file space.h
 * @author Bohan Cao (2110313@mail.nankai.edu.cn)
 * @brief 
 * @version 0.0.1
 * @date 2023-05-23
 * 
 * @copyright Copyright (c) 2023
 */
#ifndef HYPER_SPACE_H_
#define HYPER_SPACE_H_

#include <Kokkos_Core.hpp>

namespace Hyp {

class Space {
public:
    /// @brief Generate the hyperbolic space
    /// @param K Gaussian curature, must be less than 0
    inline Space(double t): t(t) {}
    
    // data
    double t;
};

class Point {
public:
    inline Point() = default;
    inline Point(double x, double y, Space *space): x(x), y(y), space(space) {}

    // data
    double x, y;
    Space *space;
    double *data = NULL;
};

class Line {
public:
    /// @brief Line between two points
    /// @param a point 1
    /// @param b point 2, must be in the same space as point 1
    Line(Point a, Point b);
    inline Line(Space *space): space(space) {}

    // data, theta = A^2/(A^2+B^2), correct = 1/sqrt(A^2+B^2-1)
    double theta, correct;
    int signA, signB;
    Space *space;
};

class PointOnLine {
public:
    inline PointOnLine(double phi, Line *l): varphi(phi), line(l) {}
    inline PointOnLine(Point p, Line *l): line(l) {
        varphi = p.space->t * Kokkos::asinh((l->signB * Kokkos::sqrt(1-l->theta) *
            p.x - l->signA * Kokkos::sqrt(l->theta)*p.y) / p.space->t);
    }

    /// @brief Find point having a distance of dist from p
    /// @param dist distance
    inline PointOnLine move(double dist) {
        return PointOnLine(varphi+dist, line);
    }

    inline PointOnLine move2(double dist) {
        if (varphi >= 0)
            return PointOnLine(varphi+dist, line);
        return PointOnLine(varphi-dist, line);
    }

    Point to_point();
    inline double operator-(PointOnLine p) { return varphi - p.varphi; }

    // data
    double varphi;
    Line *line;
};

Point focus(Line l1, Line l2);
// distance between two points, equals to other function
double dist(Point p1, Point p2);

// angle between two lines
inline double angle(Line l1, Line l2) {
    double AABB = l1.signA *l2.signA * Kokkos::sqrt(l1.theta*l2.theta) +
        l1.signB *l2.signB * Kokkos::sqrt((1-l1.theta) * (1-l2.theta));
    return Kokkos::acos(AABB * Kokkos::sqrt((l1.correct*l1.correct + 1) * 
        (l2.correct*l2.correct + 1)) - l1.correct * l2.correct);
}

// trianular area
class PieceData {
public:
    PieceData round(int i);
    Point pointA, pointB, pointC;
};

class PointOnPiece {
public:
    // inline PointOnPiece() = default;
    inline PointOnPiece(Point p, const PieceData &piece): piece(piece) {
        auto lBC = Line(piece.pointB, piece.pointC);
        auto lAp = Line(piece.pointA, p);
        
        auto f = focus(lAp, lBC);
        if (f.space == NULL) {
            data = (double*)0x100000;
            return;
        }
        // point
        auto f1 = PointOnLine(f, &lBC);
        auto B = PointOnLine(piece.pointB, &lBC);
        auto C = PointOnLine(piece.pointC, &lBC);
        auto f2 = PointOnLine(f, &lAp);
        auto p1 = PointOnLine(p, &lAp);
        auto A = PointOnLine(piece.pointA, &lAp);
        m = (C-B>0) ? f1-B : B-f1;
        n = (f2-A>0) ? p1-A : A-p1;
    }

    inline PointOnPiece move_to(const PieceData &n) {
        PointOnPiece other(*this);
        other.piece = n;
        return other;
    }
    Point to_point();

    // data
    PointOnPiece* last = NULL;
    double m, n;
    PieceData piece;
    double *data = NULL;
};

}

#endif // HYPER_SPACE_H_