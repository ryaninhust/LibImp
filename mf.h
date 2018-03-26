#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <random>

#include "omp.h"

using namespace std;


typedef double ImpFloat;
typedef double ImpDouble;
typedef long long int ImpInt;
typedef long long int ImpLong;

class Parameter {

public:
    ImpFloat lambda, w, a;
    ImpInt nr_pass, k, nr_threads;
    string model_path, predict_path;
    Parameter():lambda(0.1), w(1), a(0), nr_pass(20), k(10),nr_threads(1) {};
};

struct smat {
    vector<int> row_ptr;
    vector<long> col_idx;
    vector<long> val;
}


class ImpData {
public:
    string file_name;
    ImpLong l, m, n;
    smat R;
    smat RT;

    ImpData(string file_name): file_name(file_name), l(0), m(0), n(0) {};
    void read();
    void print_data_info();
};

class ImpProblem {
public:
    shared_ptr<ImpData> data;
    shared_ptr<ImpData> test_data;
    shared_ptr<Parameter> param;
    ImpProblem(shared_ptr<ImpData> &data, shared_ptr<Parameter> &param)
        :data(data), param(param) {};
    ImpProblem(shared_ptr<ImpData> &data, shared_ptr<ImpData> &test_data, shared_ptr<Parameter> &param)
        :data(data), test_data(test_data), param(param) {};

    vector<ImpFloat> W, H;
    vector<ImpFloat> WT, HT;

    ImpInt t;
    ImpDouble obj, reg, tr_loss;
    vector<ImpDouble> va_loss;

    ImpFloat start_time;
    ImpFloat U_time, C_time, W_time, H_time, I_time, R_time;

    ImpFloat w2_sum, h2_sum;
    ImpFloat w_sum, h_sum;
    vector<ImpFloat> wu, hv;


    ImpDouble cal_loss(ImpLong &l, smat &R);
    ImpDouble cal_reg();
    ImpDouble cal_tr_loss(ImpLong &l, smat &R);

    void update_w(ImpLong i, ImpDouble *wt, ImpDouble *ht);
    void update_h(ImpLong j, ImpDouble *wt, ImpDouble *ht);
    void save();
    void load();

    void initialize();
    void init_va_loss(ImpInt size);
    void solve();
    void update_R(ImpDouble *wt, ImpDouble *ht, bool add);

    void validate(const vector<ImpInt> &topks);
    void validate_ndcg(const vector<ImpInt> &topks);
    void predict_candidates(const ImpFloat* w, vector<ImpFloat> &Z);
    ImpLong precision_k(vector<ImpFloat> &Z, long i, const vector<ImpInt> &topks, vector<ImpLong> &hit_counts);
    ImpDouble ndcg_k(vector<ImpFloat> &Z, long i, const vector<ImpInt> &topks, vector<double> &ndcgs);
    
    void cache_w(ImpDouble *wt, ImpDouble *wu_th);
    void cache_h(ImpDouble *ht, ImpDouble *hv_th);

    void update_coordinates();
    void print_epoch_info();
    void print_header_info(vector<ImpInt> &topks);

    bool is_hit(const ImpData &data, long i , ImpLong argmax);
};

