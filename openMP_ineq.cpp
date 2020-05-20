#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <cmath>
#include <chrono>

#define VECTORSIZE 4
#define FUNCTIONQUANTITY 2
#define ENDCRITERIA 0.2
#define NETSIZE 32
#define NUMTHREADS 16

typedef struct {
    double mins[VECTORSIZE];
    double maxs[VECTORSIZE];
} Box;

typedef double (*box_func)(double* args);
void* findMinMax(Box& p, box_func* f, double* retmax, double* retmin);

void count_step(Box &p, int num_t, box_func* funcs, int s, int f, double *iter_mins, double *iter_maxs);

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



double f1(double* vector) {
    return sin(vector[0]) + cos(vector[1]);
}

double f2(double* vector) {
    return (vector[0] - 15)*(vector[0] - 15) + (vector[1] - 15)*(vector[1] - 15) - 100;
}

double f3(double* vector) {
    return (vector[0] - 5)*(vector[0] - 5) + (vector[1] - 5)*(vector[1] - 5) - 100;
}

double f4(double* vector) {
    return vector[0] + vector[1] + vector[2] - 100;
}

bool checkAxes(double* minsMaxes, Box& p, int numberOfAxes, int*axes) {
    bool flag = true;
    for (int i = 0; i < numberOfAxes; i++) {
        if (!((p.mins[axes[i] - 1] > minsMaxes[i * 2]) && (p.maxs[axes[i] - 1] < minsMaxes[i * 2 + 1]))) {
            flag = false;
            break;
        }
    }
    return flag;
}

void* findMinMax(Box& p, box_func* f, double* retmax, double* retmin){
    int iter = (int)pow((double)NETSIZE,(double)VECTORSIZE);
    int split_len = (int)(iter/ NUMTHREADS) + 1;
    double iter_maxs[NUMTHREADS * FUNCTIONQUANTITY];
    double iter_mins[NUMTHREADS * FUNCTIONQUANTITY];
    int num_t = 0;
    for (int i = 0; i < iter; i += split_len) {
        count_step(p, i / split_len, f, i, i + split_len > iter ? iter : i + split_len, iter_mins, iter_maxs);
    }

    for(int i = 0; i < NUMTHREADS; i++) {
        for (int j = 0; j < FUNCTIONQUANTITY; j++) {
            if(i == 0 || retmax[j] < iter_maxs[i * FUNCTIONQUANTITY + j])
                retmax[j] = iter_maxs[i * FUNCTIONQUANTITY + j];
            if(i == 0 || retmin[j] > iter_mins[i * FUNCTIONQUANTITY + j])
                retmin[j] = iter_mins[i * FUNCTIONQUANTITY + j];
        }
    }
}

void count_step(Box &p, int num_t, box_func* funcs, int s, int f, double *iter_mins, double *iter_maxs) {
    double local_maxs[FUNCTIONQUANTITY];
    double local_mins[FUNCTIONQUANTITY];
    double vec[VECTORSIZE];
    for(int i = s; i < f; i++){
        int cur_i = i;
        for (int j = 0; j < VECTORSIZE; j++) {
            vec[j] = p.mins[j] + ((double)fabs(p.maxs[j] - p.mins[j]) / NETSIZE) * (cur_i % NETSIZE);
            cur_i /= NETSIZE;
        }
        for (int j = 0; j < FUNCTIONQUANTITY; j++) {
            double out = funcs[j](vec);
            if (i == s || out > local_maxs[j])
                local_maxs[j] = out;
            if (i == s || out < local_mins[j])
                local_mins[j] = out;
        }
    }
    for (int j = 0; j < FUNCTIONQUANTITY; j++){
        iter_maxs[num_t * FUNCTIONQUANTITY + j] = local_maxs[j];
        iter_mins[num_t * FUNCTIONQUANTITY + j] = local_mins[j];
    }
}

int main(int argc, char** argv)
{
    auto begin = std::chrono::steady_clock::now();
    unsigned int start_time = clock();
    double mins[VECTORSIZE] = { 0,0,0};
    double maxs[VECTORSIZE] = { 30,30,30};
    Box box;
    for (int i = 0; i < VECTORSIZE; i++) {
        box.mins[i] = mins[i];
        box.maxs[i] = maxs[i];
    }
    std::vector<Box> temp;
    std::vector<Box> main;
    std::vector<Box> I;
    std::vector<Box> E;
    main.push_back(box);
    double curD = diam(box);
    double *rmin, *rmax;
    rmin = (double*)malloc(sizeof(double) * FUNCTIONQUANTITY);
    rmax = (double*)malloc(sizeof(double) * FUNCTIONQUANTITY);
    box_func func[FUNCTIONQUANTITY];
    func[0] = &f1;
    func[1] = &f2;
    func[2] = &f3;
    func[3] = &f4;

    while (curD > ENDCRITERIA && main.size() > 0) {
#pragma omp parallel num_threads(NUMTHREADS)
        {
#pragma omp for nowait
            for (int i = 0; i < main.size(); i++) {
                Box& p = main[i];
                findMinMax(p, func, rmax, rmin);
                double max = rmax[0];
                double min = rmin[0];
                for (int i = 1; i < FUNCTIONQUANTITY; i++) {
                    max = fmax(rmax[i - 1], rmax[i]);
                    min = fmax(rmin[i - 1], rmin[i]);
                }
                if (min > 0) {
#pragma omp critical
                    {
                        E.push_back(p);
                    }
                    continue;
                }
                if (max < 0) {
#pragma omp critical
                    {
                        I.push_back(p);
                    }
                    continue;
                }
                std::pair<Box, Box> sp = split(p);
#pragma omp critical
                {
                    temp.push_back(sp.first);
                    temp.push_back(sp.second);
                    curD = diam(sp.first);
                }
            }
        }
        std::cout << "Main size: " << main.size() << " Cur diam: " << curD << "\n";
        main.clear();
        main.insert(main.begin(), temp.begin(), temp.end());
        temp.clear();
    }
    free(rmax);
    free(rmin);
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
    myfile << I.size() << "\n";
    for (auto p : I) {
        for (int i = 0; i < VECTORSIZE; i++) {
            myfile << p.mins[i] << " " << p.maxs[i] << " ";
        }
        myfile << "\n";
    }
    myfile << E.size() << "\n";
    for (auto p : E) {
        for (int i = 0; i < VECTORSIZE; i++) {
            myfile << p.mins[i] << " " << p.maxs[i] << " ";
        }
        myfile << "\n";
    }
    myfile.close();
    return 0;
}
