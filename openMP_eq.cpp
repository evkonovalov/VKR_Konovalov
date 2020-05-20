#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <cmath>
#include <chrono>
#include <omp.h>

#define VECTORSIZE 4
#define FUNCTIONQUANTITY 2
#define GRIDSIZE 16
#define NUMTHREADS 8
#define ENDCRITERIA 0.5

double d = 6;
double l_a = 8;
double l_b = 8;
double l_c = 6;
double l_d = 6;

typedef struct {
    double mins[VECTORSIZE];
    double maxs[VECTORSIZE];
} Box;

typedef double (*box_func)(std::vector<double>& x);
double diam(Box &box) {
    double diam = 0;
    for (int i = 0; i < VECTORSIZE; i++) {
        diam = fmax(diam, fabs(box.maxs[i] - box.mins[i]));
    }
    return diam;
}
std::pair<Box, Box> split(Box &box) {
    std::pair<Box, Box> r;
    int maxIndex = 0;
    double max = fabs(box.maxs[0] - box.mins[0]);
    for (int i = 0; i < VECTORSIZE; i++) {
        double cur = fabs(box.maxs[i] - box.mins[i]);
        if (cur > max){
            maxIndex = i;
            max = cur;
        }
        r.first.mins[i] = box.mins[i];
        r.first.maxs[i] = box.maxs[i];
        r.second.mins[i] = box.mins[i];
        r.second.maxs[i] = box.maxs[i];
    }
    double border = box.mins[maxIndex] + fabs(box.maxs[maxIndex] - box.mins[maxIndex]) / 2;
    r.first.maxs[maxIndex] = border;
    r.second.mins[maxIndex] = border;
    return r;
}


double f1(std::vector<double>& x){
    return x[0]*x[0] + x[1]*x[1] - x[2]*x[2];
}

double f2(std::vector<double>& x){
    double d = 5;
    return (x[0]-d)*(x[0]-d) + x[1]*x[1] - x[3]*x[3];
}


double sum_func(box_func* func, std::vector<double>& x) {
    double eval = 0;
    for(int i = 0; i < FUNCTIONQUANTITY; i++) {
        double f = func[i](x);
        eval += f * f;
    }
    return eval;
}

double rr(){
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

bool harmony(Box& initial_box, box_func* func) {
    double bw_min = 0.001;
    double bw_max = 0.01;
    double par_min = 0.35;
    double par_max = 0.99;
    double r =1e-2;
    double NI = 10000;
    double F = 0.9;
    double hmcr = 0.95;
    int hms = fmin(2*VECTORSIZE,10);
    std::vector<std::vector<double>> solutions;
    int worst_num = -1;
    int best_num = -1;
    double best_eval = -1;
    double worst_eval = -1;
    for(int i = 0; i < hms; i++){
        std::vector<double> sol;
        sol.reserve(VECTORSIZE);
        for(int j = 0; j < VECTORSIZE; j++){
            sol.push_back(initial_box.mins[j] + rr()*(initial_box.maxs[j] - initial_box.mins[j]));
        }
        double cur_eval = sum_func(func,sol);
        if (best_eval == -1 or cur_eval < best_eval){
            best_eval = cur_eval;
            best_num = i;
        }
        if (worst_eval == -1 or cur_eval > worst_eval){
            worst_eval = cur_eval;
            worst_num = i;
        }
        solutions.push_back(sol);
    }
    for(int k = 0; k < NI and sum_func(func,solutions[best_num]) > r;k++){
        double cur_par = par_min + k * (par_max - par_min)/NI;
        double cur_bw = bw_max*exp(k*(log(bw_min/bw_max)/NI));
        std::vector<double> new_sol;

        for(int i = 0; i < VECTORSIZE; i++){
            double l = initial_box.mins[i];
            double u = initial_box.maxs[i];
            double new_value = rr();
            if (rr() < hmcr){
                int j_1 = 0;
                int j_2 = 0;
                while (j_1 == j_2) {
                    j_1 = rand() % (hms);
                    j_2 = rand() % (hms);
                }
                new_value = fmin(fmax(solutions[best_num][i] + F*(solutions[j_1][i] - solutions[j_2][i]),l),u);
                if (rr() < cur_par) {
                    double t = rr()*2-1;
                    new_value = fmin(fmax(new_value + t*cur_bw,l),u);
                }
            } else {
                new_value = l + rr()*(u-l);
            }
            new_sol.push_back(new_value);
        }
        double sol_eval = sum_func(func,new_sol);
        if (sol_eval < worst_eval) {
            solutions[worst_num] = new_sol;
        }
        worst_num = -1;
        best_num = -1;
        best_eval = -1;
        worst_eval = -1;
        for(int i = 0; i < hms; i++){
            double cur_eval = sum_func(func,solutions[i]);
            if (best_eval == -1 or cur_eval < best_eval){
                best_eval = cur_eval;
                best_num = i;
            }
            if (worst_eval == -1 or cur_eval > worst_eval){
                worst_eval = cur_eval;
                worst_num = i;
            }
        }
    }
    return best_eval < r;
}

std::pair<double,double> find_min_max(Box& box, box_func* func){
    std::vector<double> x(VECTORSIZE,0);
    for(int i = 0; i < VECTORSIZE; i++)
        x[i] = box.mins[i];
    double m_big, m_small;
    for(int i = 0; i < FUNCTIONQUANTITY; i++) {
        double min = func[i](x);
        double max = func[i](x);
        bool end = false;
        while(!end) {
            double cur = func[i](x);
            if (cur < min)
                min = cur;
            if(cur > max)
                max = cur;
            for (int j = 0; j < VECTORSIZE; j++) {
                if (x[j] < box.maxs[j]) {
                    x[j] += (box.maxs[j] - box.mins[j]) / GRIDSIZE;
                    break;
                } else {
                    if (j + 1 == VECTORSIZE)
                        end = true;
                    x[j] = box.mins[j];
                }
            }
        }
        if(i == 0) {
            m_big = max;
            m_small = min;
        } else {
            m_big = fmin(m_big,max);
            m_small = fmax(m_small,min);
        }
    }
    return std::pair<double,double>(m_small,m_big);
}

std::vector<Box> aprox(Box& initial_box, box_func* func, double delta) {
    std::vector<Box> l;
    l.push_back(initial_box);
    std::vector<Box> a;
    double c_d;
    while (!l.empty()) {
        std::vector<std::vector<Box>> temp(NUMTHREADS);
        std::vector<std::vector<Box>> a_temp(NUMTHREADS);
#pragma omp parallel num_threads(NUMTHREADS)
        {
#pragma omp for nowait
            for(int i = 0; i < l.size(); i++) {
                Box x = l[i];
                std::pair<double, double> mm = find_min_max(x, func);
                if (mm.first <= 0 and mm.second >= 0) {
                    c_d = diam(x);
                    if (c_d < delta) {
                        if (harmony(x, func)) {
                            a_temp[omp_get_thread_num()].push_back(x);
                        }
                    } else {
                        std::pair<Box, Box> sp = split(x);
                        temp[omp_get_thread_num()].push_back(sp.first);
                        temp[omp_get_thread_num()].push_back(sp.second);
                    }
                }

            }
        }
        l.clear();
        for(auto v: temp)
            for(auto x: v)
                l.push_back(x);
        for(auto v: a_temp)
            for(auto x: v)
                a.push_back(x);
        std::cout << "Main size: " << l.size() << " A size: " << a.size() << "\n";
    }
    return a;
}


int main() {
    srand(time(0));
    double mins[VECTORSIZE] = { -15,-15,8,8};
    double maxs[VECTORSIZE] = { 15,15,12,12};
    Box box;
    for (int i = 0; i < VECTORSIZE; i++) {
        box.mins[i] = mins[i];
        box.maxs[i] = maxs[i];
    }
    box_func func[FUNCTIONQUANTITY];
    func[0] = &f1;
    func[1] = &f2;
    harmony(box,func);
    auto begin = std::chrono::steady_clock::now();
    unsigned int start_time = clock();
    std::vector<Box> main = aprox(box,func,ENDCRITERIA);
    unsigned int end_time = clock();
    unsigned int search_time = end_time - start_time;
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "The time: " << elapsed_ms.count() << " ms\n";
    std::cout << (double) search_time / CLOCKS_PER_SEC << "\n";
    std::ofstream myfile;
    myfile.open("out.txt");
    myfile << VECTORSIZE << "\n";
    for (int i = 0; i < VECTORSIZE; i++) {
        myfile << mins[i] << " " << maxs[i] << "\n";
    }
    myfile << main.size() << "\n";
    for (auto p : main) {
        for (int i = 0; i < VECTORSIZE; i++) {
            myfile << p.mins[i] << " " << p.maxs[i] << " ";
        }
        myfile << "\n";
    }
    myfile << main.size() << "\n";
    myfile.close();
}
