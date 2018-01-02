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


typedef double FtrlFloat;
typedef double FtrlDouble;
typedef int FtrlInt;
typedef int FtrlLong;


class Node {
public:
    FtrlLong p_idx;
    FtrlLong q_idx;
    FtrlDouble  val;
    Node(){};
    Node(FtrlLong p_idx, FtrlLong q_idx, FtrlDouble val): p_idx(p_idx), q_idx(q_idx), val(val){};
};

class Parameter {

public:
    FtrlFloat lambda, w, a;
    FtrlInt nr_pass, k, nr_threads;
    Parameter():lambda(0.1), w(1), a(0), nr_pass(20), k(10),nr_threads(1) {};
};


class FtrlData {
public:
    string file_name;
    FtrlLong l, m, n;
    vector<Node> R;
    vector<vector<Node*>> P;
    vector<vector<Node*>> Q;

    FtrlData(string file_name): file_name(file_name), l(0), m(0), n(0) {};
    void transpose();
    void read();
    void print_data_info();
};

class FtrlProblem {
public:
    shared_ptr<FtrlData> data;
    shared_ptr<FtrlData> test_data;
    shared_ptr<Parameter> param;
    FtrlProblem(shared_ptr<FtrlData> &data, shared_ptr<FtrlData> &test_data, shared_ptr<Parameter> &param)
        :data(data), test_data(test_data), param(param) {};


    vector<FtrlFloat> W, H;
    vector<FtrlFloat> WT, HT;

    FtrlInt t;
    FtrlDouble obj, reg, tr_loss, va_loss;
    FtrlFloat start_time;
    FtrlFloat U_time, C_time, W_time, H_time, I_time, R_time;

    FtrlFloat w2_sum, h2_sum;
    FtrlFloat w_sum, h_sum;
    vector<FtrlFloat> wu, hv;


    FtrlDouble cal_loss(FtrlLong &l, vector<Node> &R);
    FtrlDouble cal_reg();
    FtrlDouble cal_tr_loss(FtrlLong &l, vector<Node> &R);

    void update_w(FtrlLong i, FtrlDouble *wt, FtrlDouble *ht);
    void update_h(FtrlLong j, FtrlDouble *wt, FtrlDouble *ht);

    void initialize();
    void solve();
    void update_R();
    void update_R(FtrlDouble *wt, FtrlDouble *ht, bool add);

    void validate(const FtrlInt &topk);
    void validate_ndcg(const FtrlInt &topk);
    void predict_candidates(const FtrlFloat* w, vector<FtrlFloat> &Z);
    FtrlLong precision_k(vector<FtrlFloat> &Z, const vector<Node*> &p, const vector<Node*> &tp, const FtrlInt &topk);
    FtrlDouble ndcg_k(vector<FtrlFloat> &Z, const vector<Node*> &p, const vector<Node*> &tp, const FtrlInt &topk);
    
    void cache_w(FtrlDouble *wt);
    void cache_h(FtrlDouble *ht);

    void update_coordinates();
    void print_epoch_info();
    void print_header_info();

    bool is_hit(const vector<Node*> p, FtrlLong argmax);
};

