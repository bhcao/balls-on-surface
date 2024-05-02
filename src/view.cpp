/**
 * @file view.cpp
 * @author Bohan Cao (2110313@mail.nankai.edu.cn)
 * @brief 
 * @version 0.0.1
 * @date 2023-05-29
 * 
 * @copyright Copyright (c) 2023
 */
#include "view.h"

#include <stdio.h>
#include <vector>

#include <Kokkos_Vector.hpp>

namespace Hyp {

View::View(Space *space): gen(std::random_device()()), dis(-M_PI/2, M_PI/2), dis01(0, 1),
    gau_dis(0, 1.0) {
    double k1 = Kokkos::sqrt(2*Kokkos::sqrt(2)+2) * space->t;
    for (int i=0; i<8; i++) {
        data[i] = {
            Point(0, 0, space),
            Point(k1*Kokkos::cos(M_PI*i/4), k1*Kokkos::sin(M_PI*i/4), space),
            Point(k1*Kokkos::cos(M_PI*(i+1)/4), k1*Kokkos::sin(M_PI*(i+1)/4), space)
        };
    }
}

FullView::FullView(Space *space) {
    double k1 = Kokkos::sqrt(2*Kokkos::sqrt(2)+2) * space->t;
    double k2 = 2*Kokkos::sqrt(3*Kokkos::sqrt(2)+4) * space->t;
    for (int i=0; i<8; i++) {
        data[i] = {
            Point(0, 0, space),
            Point(k1*Kokkos::cos(M_PI*i/4), k1*Kokkos::sin(M_PI*i/4), space),
            Point(k1*Kokkos::cos(M_PI*(i+1)/4), k1*Kokkos::sin(M_PI*(i+1)/4), space)
        };

        data[i+8] = {
            Point(k2*Kokkos::cos(M_PI*(i/4.0+1.0/8)), k2*Kokkos::sin(M_PI*(i/4.0+1.0/8)), space),
            Point(k1*Kokkos::cos(M_PI*(i+1)/4), k1*Kokkos::sin(M_PI*(i+1)/4), space),
            Point(k1*Kokkos::cos(M_PI*i/4), k1*Kokkos::sin(M_PI*i/4), space)
        };
    }
}

Line line_of_theta(Point p, double theta) {
    double x = p.x, y = p.y;
    double xy2 = Kokkos::sqrt(x*x + y*y);
    double cos_ = Kokkos::cos(theta), sin_ = Kokkos::sin(theta) * p.space->t;
    double x_xy2 = x/xy2, y_xy2 = y/xy2;
    double txy = (x*x + y*y)*cos_*cos_ + p.space->t*p.space->t;
    if (Kokkos::isnan(x_xy2))
        x_xy2 = 1, y_xy2 = 0;
    
    Line line(p.space);
    line.correct = xy2 * cos_ / p.space->t;
    cos_ = cos_ * Kokkos::sqrt(x*x + y*y + p.space->t*p.space->t);
    line.signA = (x_xy2*cos_ + y_xy2*sin_ > 0) ? 1 : -1;
    line.signB = (y_xy2*cos_ - x_xy2*sin_ > 0) ? 1 : -1;
    line.theta = (x_xy2*cos_ + y_xy2*sin_) * (x_xy2*cos_ + y_xy2*sin_) / txy;

    return line;
}

double angle_of(Point a, Point b) {
    // x1,y1,z1  x2,y2,z2
    double z1 = Kokkos::sqrt(a.x*a.x + a.y*a.y + a.space->t*a.space->t);
    double z2 = Kokkos::sqrt(b.x*b.x + b.y*b.y + b.space->t*b.space->t);
    double x1 = a.x, y1 = a.y, x2 = b.x, y2 = b.y;

    // theta
    double C = x2*y1 - x1*y2, A = z1*y2 - y1*z2, B = x1*z2 - z1*x2;
    double delta = Kokkos::sqrt(A*A + B*B - C*C);
    double la = x1*x1 + y1*y1, lb = x2*x2 + y2*y2;
    double theta = Kokkos::acos(C * b.space->t / delta / Kokkos::sqrt(la));
    if (lb > la)
        return theta;
    return 2*M_PI - theta;
}

void View::update(int h, int f) {
    // get data
    auto space = pieces[0]->data.pointA.space;
    int num = 0;
    for (int i=0; i<8; i++)
        num += pieces[i]->points.size();
    Kokkos::vector<Point> points(num, Point(0, 0, space));
    
    int n = 0, denote = 0;
    for (int i=0; i<8; i++)
    for (int j=0; j<pieces[i]->points.size(); j++) {
        points[n] = pieces[i]->points[j].move_to(data[i].round(-rnds[i])).to_point();
        if (i==h && j==f) {
            denote = n;
        }
        n++;
    }

#define MD
#ifdef MD
    int h2 = -1, f2 = -1;
    for (int li=0; li<8; li++)
    for (int lj=0; lj<pieces2[li]->points.size(); lj++)
        if (pieces[h]->points[f].last == &pieces2[li]->points[lj]) {
            h2 = li; f2 = lj;
        }
    if (h2 == -1)
       h2 ++;

    Point last(0, 0, space);
    last = pieces2[h2]->points[f2].move_to(data[h2].round(-rnds[h2])).to_point();

    // random move
    double x = Kokkos::sqrt(2*T*dt*gamma) * gau_dis(gen);
    double y = Kokkos::sqrt(2*T*dt*gamma) * gau_dis(gen);

    for (int i=0; i<num; i++) {
        if (i != denote) {
            double len = -part_energy(dist(points[i], points[denote]));
            double theta = angle_of(points[denote], points[i]);
            if (Kokkos::isnan(theta))
                continue;
            x += len * Kokkos::cos(theta) * dt;
            y += len * Kokkos::sin(theta) * dt;
        }
    }

    double len = -2*dist(last, points[denote]);
    double theta = angle_of(points[denote], last);
    if (!Kokkos::isnan(theta)) {
        x += len * Kokkos::cos(theta) / dt;
        y += len * Kokkos::sin(theta) / dt;
    }

    len = Kokkos::sqrt(x*x + y*y) / (2 + dt*gamma) * dt;
    theta = Kokkos::atan(y/x);
    if (Kokkos::isnan(theta))
        theta = M_PI/2;

    auto line = line_of_theta(points[denote], theta);
    assert(!Kokkos::isnan(PointOnLine(points[denote], &line).varphi));
    auto new_point = PointOnLine(points[denote], &line).move(len).to_point();
    if ((x == 0 && y*(new_point.y - points[denote].y) < 0) || 
            x*(points[denote].x*new_point.y - points[denote].y*new_point.x) > 0)
        new_point = PointOnLine(points[denote], &line).move(-len).to_point();

    assert(!Kokkos::isnan(new_point.x));

{   
    double theta = 4 * Kokkos::atan(new_point.y / new_point.x) / M_PI;
    if (new_point.x == 0)
        theta = (new_point.y >= 0) ? 2 : -2;
    else if (new_point.x < 0)
        theta += 4;
    if (theta < 0)
        theta += 8;
    int n = static_cast<int>(theta);
    if (PointOnPiece(new_point, data[n].round(-rnds[n])).data == (double*)0x100000)
        n++;
    auto point = PointOnPiece(new_point, data[n].round(-rnds[n])).move_to(pieces[n]->data);

    if (h2 == h) {
        pieces2[h2]->points[f2] = pieces[h]->points[f];
        point.last = &pieces2[h2]->points[f2];
    } else {
        pieces2[h2]->points[f2] = pieces2[h2]->points[pieces2[h2]->points.size()-1];
        pieces2[h2]->points.pop_back();
        pieces2[h]->points.push_back(pieces[h]->points[f]);
        point.last = &pieces2[h]->points[pieces2[h]->points.size()-1];
    }

    if (n == h) {
        pieces[h]->points[f] = point;
    } else {
        pieces[h]->points[f] = pieces[h]->points[pieces[h]->points.size()-1];
        pieces[h]->points.pop_back();
        pieces[n]->points.push_back(point);
    }
}

#else // MC
    // random move
    double theta = dis(gen);
    double len = range * dis(gen) / M_PI;

    auto line = line_of_theta(points[denote], dis(gen));
    auto new_point = PointOnLine(points[denote], &line).move(len).to_point();

    // energy
    double denergy = 0;
    for (int i=0; i<num; i++) {
        if (i != denote) {
            denergy += energy(dist(points[i], new_point)) - 
                energy(dist(points[i], points[denote]));
        }
    }

    if (denergy < 0 || dis01(gen) < Kokkos::exp(-denergy/T)) {
        double theta = 4 * Kokkos::atan(new_point.y / new_point.x) / M_PI;
        if (new_point.x == 0)
            theta = (new_point.y >= 0) ? 2 : -2;
        else if (new_point.x < 0)
            theta += 4;
        if (theta < 0)
            theta += 8;
        int n = static_cast<int>(theta);
        auto point = PointOnPiece(new_point, data[n].round(-rnds[n])).move_to(pieces[n]->data);
        if (n == h) {
            pieces[h]->points[f] = point;
        } else {
            pieces[h]->points[f] = pieces[h]->points[pieces[h]->points.size()-1];
            pieces[h]->points.pop_back();
            pieces[n]->points.push_back(point);
        }
    }
#endif
}

void View::compute(int h, int f, double *input) {
    // get data
    auto space = pieces[0]->data.pointA.space;
    int num = 0;
    for (int i=0; i<8; i++)
        num += pieces[i]->points.size();
    Kokkos::vector<Point> points(num, Point(0, 0, space));
    int n = 0, denote = 0;
    for (int i=0; i<8; i++)
    for (int j=0; j<pieces[i]->points.size(); j++) {
        points[n] = pieces[i]->points[j].move_to(data[i].round(-rnds[i])).to_point();
        if (i==h && j==f) {
            denote = n;
            pieces[i]->points[j].data = input;
        }
        n++;
    }

    double en = 0, press = 0;

    // list: length, list_: identifier
    double list[6] = {10, 10, 10, 10, 10, 10};
    int list_[6] = {0};

    for (int i=0; i<num; i++) {
        if (i == denote)
            continue;
        double length = dist(points[i], points[denote]);
        en += energy(length);
        press += length * part_energy(length);
        for (int j=0; j<6; j++)
            if (length < list[j]) {
                for (int l=6; l>j; l--) {
                    list[l] = list[l-1];
                    list_[l] = list_[l-1];
                }
                list[j] = length;
                list_[j] = i;
                break;
            }
    }

    double vecx = 0, vecy = 0;
    double vecx2 = 0, vecy2 = 0;

    for (int i=0; i<6; i++) {
        double theta = angle_of(points[denote], points[list_[i]]);
        vecx += Kokkos::cos(6*theta);
        vecy += Kokkos::sin(6*theta);
        if (i<4) {
            vecx2 += Kokkos::cos(4*theta);
            vecy2 += Kokkos::sin(4*theta);
        }
    }

    pieces[h]->points[f].data[0] = (vecx*vecx + vecy*vecy) / 36;
    pieces[h]->points[f].data[1] = (vecx2*vecx2 + vecy2*vecy2) / 16;
    pieces[h]->points[f].data[2] = en;
    pieces[h]->points[f].data[3] = press;
}

void FullView::dump(std::string name) {
    // get data
    auto space = pieces[0]->data.pointA.space;
    int num = 0;
    for (int i=0; i<16; i++)
        num += pieces[i]->points.size();
    Kokkos::vector<Point> points(num, Point(0, 0, space));
    int n = 0;
    for (int i=0; i<16; i++)
    for (int j=0; j<pieces[i]->points.size(); j++) {
        points[n] = pieces[i]->points[j].move_to(data[i].round(-rnds[0])).to_point();
        n++;
    }

    // write
    double wid = 2*Kokkos::sqrt(3*Kokkos::sqrt(2)+4) * space->t;
    FILE *file = std::fopen(name.c_str(), (pieces[0]->step_num != 0) ? "a" : "w");
    std::fprintf(file, "ITEM: TIMESTEP\n%i\nITEM: NUMBER OF ATOMS\n%i\nITEM: BOX BOUNDS ss ss ss\n"
        "%.4f\t%.4f\n%.4f\t%.4f\n-0.01\t0.01\nITEM: ATOMS id type x y z hex quad\n", pieces[0]->step_num,
        num, -wid, wid, -wid, wid);
    for (int i=0; i<points.size(); i++)
        std::fprintf(file, "%i 1 %.6f %.6f 0.0 %.6f %.6f\n", i+1, points[i].x, points[i].y, 
            points[i].data[0], points[i].data[1]);

    std::fclose(file);
}

int get_area(Point p) {
    double theta = 2 * Kokkos::atan(p.y / p.x) / M_PI;
    if (p.x == 0)
        theta = (p.y >= 0) ? 1 : -1;
    else if (p.x < 0)
        theta += 2;
    theta += 2/3.0;
    if (theta < 0)
        theta += 4;
    if (theta >= 4)
        theta -= 4;
    int n = static_cast<int>(theta * 3 / 4);
    if (n == 2)
        n = -1;
    return n;
}

}