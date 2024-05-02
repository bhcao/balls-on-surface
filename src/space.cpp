/**
 * @file space.cpp
 * @author Bohan Cao (2110313@mail.nankai.edu.cn)
 * @brief 
 * @version 0.0.1
 * @date 2023-05-28
 * 
 * @copyright Copyright (c) 2023
 */
#include "space.h"

namespace Hyp {

Line::Line(Point a, Point b) {
    double z1 = Kokkos::sqrt(a.x*a.x + a.y*a.y + a.space->t*a.space->t);
    double z2 = Kokkos::sqrt(b.x*b.x + b.y*b.y + a.space->t*a.space->t);
    double div = (a.x*b.y - a.y*b.x) * (a.x*b.y - a.y*b.x);
    double A2 = (z1*b.y - a.y*z2) * (z1*b.y - a.y*z2);
    double B2 = (z2*a.x - z1*b.x) * (z2*a.x - z1*b.x);
    space = a.space;
    theta = A2 / (A2 + B2);
    correct = Kokkos::sqrt(div / (A2 + B2 - div));
    signA = (z1*b.y - a.y*z2 > 0) ? 1 : -1;
    signB = (z2*a.x - z1*b.x > 0) ? 1 : -1;
    if (a.x*b.y - a.y*b.x < 0) {
        signA = -signA;
        signB = -signB;
    }
}

Point focus(Line l1, Line l2) {
    double C1 = l1.correct, A1 = l1.signA * Kokkos::sqrt(l1.theta*(C1*C1+1)),
        B1 = l1.signB * Kokkos::sqrt((1-l1.theta)*(C1*C1+1)),
        C2 = l2.correct, A2 = l2.signA * Kokkos::sqrt(l2.theta*(C2*C2+1)), 
        B2 = l2.signB * Kokkos::sqrt((1-l2.theta)*(C2*C2+1));
    double AB = A1*B2 - B1*A2, AC = A1*C2 - A2*C1, BC = B2*C1 - B1*C2;
    double k = l1.space->t / Kokkos::sqrt(AB*AB - AC*AC - BC*BC);

    if (AB*AB - AC*AC - BC*BC < 0)
        return Point(0, 0, NULL);
    if (AB > 0)
        return Point(k*BC, k*AC, l1.space);
    return Point(-k*BC, -k*AC, l1.space);
}

double dist(Point p1, Point p2) {
    if (p1.x == p2.x && p1.y == p2.y)
        return 0;
    double t2 = p1.space->t*p1.space->t;
    double z1 = Kokkos::sqrt(p1.x*p1.x + p1.y*p1.y + t2);
    double z2 = Kokkos::sqrt(p2.x*p2.x + p2.y*p2.y + t2);
    double A = z1*p2.y - p1.y*z2, B = z2*p1.x - z1*p2.x;
    double A2B2 = A*A + B*B;
    double v1 = B*p1.x-A*p1.y, v2 = B*p2.x-A*p2.y;
    return p1.space->t * Kokkos::abs(Kokkos::log((v1 + Kokkos::sqrt(v1*v1 + t2*A2B2))/
        (v2 + Kokkos::sqrt(v2*v2 + t2*A2B2))));
}

Point PointOnLine::to_point() {
    double sh = Kokkos::sinh(varphi/line->space->t);
    double ch = Kokkos::cosh(varphi/line->space->t) * line->correct;
    return Point(line->space->t * (line->signB * Kokkos::sqrt(1-line->theta)*sh + 
        line->signA * Kokkos::sqrt(line->theta)*ch), line->space->t * 
        (-line->signA * Kokkos::sqrt(line->theta)*sh + 
        line->signB * Kokkos::sqrt(1-line->theta)*ch), line->space);
}

Point PointOnPiece::to_point() {
    // get the focus
    auto lBC = Line(piece.pointB, piece.pointC);
    auto B = PointOnLine(piece.pointB, &lBC);
    auto C = PointOnLine(piece.pointC, &lBC);
    auto f = B.move((C-B > 0) ? m : -m).to_point();

    // get the point
    auto line = Line(piece.pointA, f);
    auto newf = PointOnLine(f, &line);
    auto A = PointOnLine(piece.pointA, &line);
    auto p = A.move((newf-A > 0) ? n : -n).to_point();
    p.data = data;
    return p;
}

PieceData PieceData::round(int rnd) {
    PieceData other(*this);
    switch (rnd) {
    case 0:
        break;
    case 1:
        other.pointA = pointB;
        other.pointB = pointC;
        other.pointC = pointA;
        break;
    case -1:
        other.pointA = pointC;
        other.pointB = pointA;
        other.pointC = pointB;
    }
    return other;
}

}