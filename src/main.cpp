/**
 * @file main.cpp
 * @author Bohan Cao (2110313@mail.nankai.edu.cn)
 * @brief 
 * @version 0.0.1
 * @date 2023-05-29
 * 
 * @copyright Copyright (c) 2023
 */
#include <iostream>
#include <random>
#include <cmath>

#include <Kokkos_Core.hpp>

#include "space.h"
#include "view.h"

int main() {

/*for (int n = 7; n < 28; n++) {
    double rho = 4/M_PI*n/t/t;
    std::cout << n << ": " << rho << "\n";
}20,pp+7*/

Kokkos::initialize(); {
// Kokkos::parallel_for(11, KOKKOS_LAMBDA(int pp) {

    auto s = Hyp::Space(5);
    int n = 16;

    /*auto line = Hyp::line_of_theta(Hyp::Point(1,1,&s), M_PI/4);
    auto pol = Hyp::PointOnLine(Hyp::Point(1,1,&s), &line).move2(1).to_point();
    std::printf("%.8f\t%.8f\t%.8f\n", pol.x, pol.y, angle_of(Hyp::Point(1,1,&s), pol));*/

    Hyp::StdPiece pieces[16]{{&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n},
        {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}};
    Hyp::StdPiece pieces2[16]{{&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n},
        {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}, {&s, n}};
    
    for (int li=0; li<16; li++)
    for (int lj=0; lj<n; lj++)
        pieces[li].points[lj].last = &pieces2[li].points[lj];

    Hyp::View view[6]{&s, &s, &s, &s, &s, &s};
    Hyp::FullView full[2]{&s, &s};

    int para[6][8][2] = {
        {{0, 0}, {7, 0}, {6, 0}, {1, 0}, {4, 0}, {3, 0}, {2, 0}, {5, 0}},
        {{8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}},
        {{0, -1}, {15, 1}, {14, -1}, {1, 1}, {6, -1}, {9, 1}, {8, -1}, {7, 1}},
        {{5, -1}, {10, 1}, {9, -1}, {6, 1}, {7, -1}, {8, 1}, {15, -1}, {0, 1}},
        {{1, -1}, {14, 1}, {13, -1}, {2, 1}, {3, -1}, {12, 1}, {11, -1}, {4, 1}},
        {{2, -1}, {13, 1}, {12, -1}, {3, 1}, {4, -1}, {11, 1}, {10, -1}, {5, 1}}
    };

    int para2[2][16][2] = {
        {{0, 0}, {7, 0}, {6, 0}, {1, 0}, {4, 0}, {3, 0}, {2, 0}, {5, 0}, 
         {15, 0}, {8, 0}, {9, 0}, {14, 0}, {11, 0}, {12, 0}, {13, 0}, {10, 0}},
        {{8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0},
         {7, 0}, {6, 0}, {5, 0}, {4, 0}, {3, 0}, {2, 0}, {1, 0}, {0, 0}}
    };

    for (int i=0; i<6; i++) {
        view[i].range = 0.5;
        view[i].T = 1;
        view[i].dt = 1e-5;
        view[i].gamma = 1e6;
        for (int j=0; j<8; j++)
            view[i].equip(&pieces[para[i][j][0]], &pieces2[para[i][j][0]], j, para[i][j][1]);
    }

    for (int i=0; i<2; i++)
        for (int j=0; j<16; j++)
            full[i].equip(&pieces[para2[i][j][0]], j, para2[i][j][1]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 16*n);

    double datas[16 * n][4];

/*for (int hj=0; hj<500; hj++) {

    for (int i=0; i<16*n; i++) {
        int h = 0, f = static_cast<int>(dis(gen));
        for (; h<16; h++)
            if (f >= pieces[h].points.size())
                f -= pieces[h].points.size();
            else
                break;
        auto point = pieces[h].points[f].to_point();
        int n = Hyp::get_area(point);
        
        // find hj
        int view_n = -1, new_h = -1;
        for (int l=0; l<6; l++)
        for (int m=0; m<8; m++) {
            if (para[l][m][0] == h && para[l][m][1] == n) {
                view_n = l;
                new_h = m;
                break;
            }
        }

        view[view_n].update(new_h, f);
    }

    for (int yu=0; yu<16; yu++)
        pieces[yu].step_num += 1;
}*/

// begin calculate
for (double T=100; T>=0; T-=0.1) {
    for (int i=0; i<6; i++) {
        view[i].T = 0;
    }
    int count = 0, hex = 0, quad = 0;
    double en = 0, press = 0;

for (int hj=0; hj<100; hj++) {

    for (int i=0; i<16*n; i++) {
        int h = 0, f = i; //static_cast<int>(dis(gen));
        for (; h<16; h++)
            if (f >= pieces[h].points.size())
                f -= pieces[h].points.size();
            else
                break;
        auto point = pieces[h].points[f].to_point();
        for (int jj=0; jj<16; jj++)
        for (int jk=0; jk<pieces2[jj].points.size(); jk++)
            if (pieces[h].points[f].last == &pieces2[jj].points[jk] && h != jj)
                std::printf("%i\t%i\t%i\t%i\n", h, f, jj, jk);

        int n = Hyp::get_area(point);
        
        // find hj
        int view_n = -1, new_h = -1;
        for (int l=0; l<6; l++)
        for (int m=0; m<8; m++) {
            if (para[l][m][0] == h && para[l][m][1] == n) {
                view_n = l;
                new_h = m;
                break;
            }
        }

        view[view_n].update(new_h, f);
    }

    for (int yu=0; yu<16; yu++)
        pieces[yu].step_num += 1;

    int count2 = 0;
    for (int i=0; i<16; i++)
    for (int j=0; j<pieces[i].points.size(); j++) {
        auto point = pieces[i].points[j].to_point();
        int n2 = Hyp::get_area(point);

        int view_n = -1, new_h = -1;
        for (int l=0; l<6; l++)
        for (int m=0; m<8; m++) {
            if (para[l][m][0] == i && para[l][m][1] == n2) {
                view_n = l;
                new_h = m;
                break;
            }
        }

        view[view_n].compute(new_h, j, datas[count2]);
        en += datas[count2][2];
        press += datas[count2][3];
        if (datas[count2][0] > 0.5)
            hex++;
        if (datas[count2][1] > 0.5)
            quad++;
        count2++;
        count++;
    }
}
    full[0].dump(std::to_string(n)+"_up.dump");
    full[1].dump(std::to_string(n)+"_down.dump");
    std::printf("%.8f\t%.8f\t%.8f\t%.8f\n", hex/(double)count, quad/(double)count, 
        en/500/2, (0*n+press/500/4) / (4*M_PI*25));
}

/*int count = 0, hex = 0, quad = 0;
double en = 0, press = 0;

for (int hj=0; hj<500; hj++) {

    for (int i=0; i<16*n; i++) {
        int h = 0, f = static_cast<int>(dis(gen));
        for (; h<16; h++)
            if (f >= pieces[h].points.size())
                f -= pieces[h].points.size();
            else
                break;
        auto point = pieces[h].points[f].to_point();
        for (int jj=0; jj<16; jj++)
        for (int jk=0; jk<pieces2[jj].points.size(); jk++)
            if (pieces[h].points[f].last == &pieces2[jj].points[jk] && h != jj)
                std::printf("%i\t%i\t%i\t%i\n", h, f, jj, jk);

        int n = Hyp::get_area(point);
        
        // find hj
        int view_n = -1, new_h = -1;
        for (int l=0; l<6; l++)
        for (int m=0; m<8; m++) {
            if (para[l][m][0] == h && para[l][m][1] == n) {
                view_n = l;
                new_h = m;
                break;
            }
        }

        view[view_n].update(new_h, f);
    }

    for (int yu=0; yu<16; yu++)
        pieces[yu].step_num += 1;

    int count2 = 0;
    for (int i=0; i<16; i++)
    for (int j=0; j<pieces[i].points.size(); j++) {
        auto point = pieces[i].points[j].to_point();
        int n2 = Hyp::get_area(point);

        int view_n = -1, new_h = -1;
        for (int l=0; l<6; l++)
        for (int m=0; m<8; m++) {
            if (para[l][m][0] == i && para[l][m][1] == n2) {
                view_n = l;
                new_h = m;
                break;
            }
        }

        view[view_n].compute(new_h, j, datas[count2]);
        en += datas[count2][2];
        press += datas[count2][3];
        if (datas[count2][0] > 0.5)
            hex++;
        if (datas[count2][1] > 0.5)
            quad++;
        count2++;
        count++;
    }
}
    full[0].dump(std::to_string(n)+"_up.dump");
    full[1].dump(std::to_string(n)+"_down.dump");
    //std::printf("%.8f\t%.8f\t%.8f\t%.8f\n", hex/(double)count, quad/(double)count, 
    //    en/500/2, (T*n+press/500/4) / (4*M_PI*25));
    std::printf("%i\t%.8f\t%.8f\t%.8f\n", n, hex/(double)count, quad/(double)count, 
        (press/500/4) / (4*M_PI*25));*/

//});
} Kokkos::finalize();

    return 0;
}
