#include "mf.h"

#define MIN_Z -10000;

double inner(const double *a, const double *b, const int k)
{
    double r = 0.0;
    for (int i = 0; i < k; i++)
        r += a[i]*b[i];
    return r;
}

void FtrlData::read() {
    string line;
    ifstream fs(file_name);
    
    while (getline(fs, line)) {
        istringstream iss(line);
        l++;

        FtrlLong p_idx, q_idx;

        iss >> p_idx;
        iss >> q_idx;

        FtrlFloat val;
        iss >> val;
        
        m = max(p_idx+1, m);
        n = max(q_idx+1, n);

        R.push_back(Node(p_idx, q_idx, val));
    }
}

void FtrlData::transpose() {
    P.resize(m);
    Q.resize(n);
    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &R[i];
        P[node->p_idx].push_back(node);
        Q[node->q_idx].push_back(node);
    }
}

void FtrlData::print_data_info() {
    cout << "Data: " << file_name << "\t";
    cout << "#m: " << m << "\t";
    cout << "#n: " << n << "\t";
    cout << "#l: " << l << "\t";
    cout << endl;
}

void FtrlProblem::initialize() {
    double time = omp_get_wtime();
    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    t = 0;
    tr_loss = 0.0, va_loss = 0.0, obj = 0.0, reg=0.0;
    U_time = 0.0, C_time = 0.0, I_time = 0.0, R_time = 0.0, W_time = 0.0, H_time = 0.0; 
    W.resize(k*m);
    H.resize(k*n);

    WT.resize(k*m);
    HT.resize(k*n);

    w2_sum.resize(k);
    h2_sum.resize(k);

    w_sum.resize(k);
    h_sum.resize(k);

    wu.resize(k);
    hv.resize(k);


    default_random_engine engine(0);
    uniform_real_distribution<FtrlFloat> distribution(0, 1.0/sqrt(k));

    for (FtrlInt d = 0; d < k; d++)
    {
        for (FtrlLong j = 0; j < m; j++) {
            W[j*k+d] = distribution(engine);
            WT[d*m+j] = W[j*k+d];
        }
        for (FtrlLong j = 0; j < n; j++) {
            if (data->Q[j].size() != 0 )
                H[j*k+d] = 0;
            else
                H[j*k+d] = distribution(engine);
            HT[d*n+j] = H[j*k+d];
        }
    }
    I_time += omp_get_wtime() - time;
    start_time = omp_get_wtime();
}

void FtrlProblem::print_header_info() {
    cout.width(4);
    cout << "iter";
    if (!test_data->file_name.empty()) {
        cout.width(13);
        cout << "va_p@10";
    }
    cout << "\n";
}

void FtrlProblem::print_epoch_info() {
    cout.width(4);
    cout << t+1;
    if (!test_data->file_name.empty()) {
        cout.width(13);
        cout << setprecision(4) << va_loss;
    }
    cout << "\n";
} 


void FtrlProblem::update_w(FtrlLong i, FtrlInt d) {
    double time = omp_get_wtime();    
    FtrlInt k = param->k;
    FtrlFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<vector<Node*>> &P = data->P;
    FtrlLong m = data->m, n = data->n;
    FtrlDouble w_val = W[i*k+d];
    FtrlDouble h = lambda*P[i].size(), g = 0;
    for (Node* p : P[i]) {
        FtrlDouble r = p->val;
        FtrlLong j = p->q_idx;
        FtrlDouble h_val = HT[d*n+j];
        g += ((1-w)*(r+h_val*w_val)+w*(1-a))*h_val;
        h += (1-w)*h_val*h_val;
    }
    h += w*h2_sum[d];

    FtrlDouble wTh = 0;
    for (FtrlInt d1 = 0; d1 < k; d1++) {
        wTh += hv[d1]*W[i*k+d1];
    }
    g += w*(a*h_sum[d]-wTh+w_val*h2_sum[d]);

    FtrlDouble new_w_val = g/h;
    U_time += omp_get_wtime() - time;
    time = omp_get_wtime();
    for (Node* p : P[i]) {
        FtrlLong j = p->q_idx;
        FtrlDouble h_val = HT[d*n+j];
        p->val += (w_val-new_w_val)*h_val;
    }
    W[i*k+d] = new_w_val;
    WT[d*m+i] = new_w_val;
    R_time += omp_get_wtime() - time;
}

void FtrlProblem::update_h(FtrlLong j, FtrlInt d) {
    double time = omp_get_wtime();
    FtrlInt k = param->k;
    FtrlFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<vector<Node*>> &Q = data->Q;
    FtrlLong m = data->m, n = data->n;
    FtrlDouble h_val = H[j*k+d];
    FtrlDouble h = lambda*Q[j].size(), g = 0;
    for (Node* q : Q[j]) {
        FtrlDouble r = q->val;
        FtrlLong i = q->p_idx;
        FtrlDouble w_val = WT[d*m+i];
        g += ((1-w)*(r+h_val*w_val)+w*(1-a))*w_val;
        h += (1-w)*w_val*w_val;
    }
    h += w*w2_sum[d];

    FtrlFloat wTh = 0;
    for (FtrlInt d1 = 0; d1 < k; d1++) {
        wTh += wu[d1]*H[j*k+d1];
    }
    g += w*(a*w_sum[d]-wTh+h_val*w2_sum[d]);

    FtrlDouble new_h_val = g/h;
    U_time += omp_get_wtime() - time;
    time = omp_get_wtime();
    for (Node* q : Q[j]) {
        FtrlLong i = q->p_idx;
        FtrlDouble w_val = WT[d*m+i];
        q->val += (h_val-new_h_val)*w_val;
    }
    H[j*k+d] = new_h_val;
    HT[d*n+j] = new_h_val;
    R_time += omp_get_wtime() - time;
}


FtrlDouble FtrlProblem::cal_loss(FtrlLong &l, vector<Node> &R) {
    FtrlInt k = param->k;
    FtrlDouble loss = 0, a = param->a;
    FtrlLong m = data->m, n = data->n;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &R[i];
        if (node->p_idx+1>data->m || node->q_idx+1>data->n)
            continue;
        FtrlDouble *w = W.data()+node->p_idx*k;
        FtrlDouble *h = H.data()+node->q_idx*k;
        FtrlDouble r = inner(w, h, k);
        loss += (r-node->val)*(r-node->val);
        loss -= param->w*(a-r)*(a-r);
    }
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (FtrlLong i = 0; i < m; i++) {
        for (FtrlLong j = 0; j < n; j++) {
            FtrlDouble *w = W.data()+i*k;
            FtrlDouble *h = H.data()+j*k;
            FtrlDouble r = inner(w, h, k);
            loss += param->w*(a-r)*(a-r);
        }
    }
    return loss;
}

FtrlDouble FtrlProblem::cal_tr_loss(FtrlLong &l, vector<Node> &R) {
    FtrlDouble loss = 0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &R[i];
        loss += node->val*node->val;
    }
    return loss;
}

void FtrlProblem::validate(const FtrlInt &topk) {
    FtrlInt k = param->k;
    FtrlLong n = data->n, m = data->m;
    vector<FtrlFloat> Z(n, 0);
    const vector<vector<Node*>> &P = data->P;
    const vector<vector<Node*>> &TP = test_data->P;
    const FtrlFloat* Wp = W.data();
    FtrlLong hit_count = 0;
    FtrlLong valid_samples = 0;
    for (FtrlLong i = 0; i < m; i++) {
        const vector<Node*> p = P[i];
        const vector<Node*> tp = TP[i];
        if (!tp.size()) {
            continue;
        }
        const FtrlFloat *w = Wp+i*k;
        predict_candidates(w, Z);
        hit_count += precision_k(Z, p, tp, topk);
        valid_samples++;
    }
    va_loss = hit_count/double(valid_samples*topk);
}

void FtrlProblem::validate_ndcg(const FtrlInt &topk) {
    FtrlInt k = param->k;
    FtrlLong n = data->n, m = data->m;
    vector<FtrlFloat> Z(n, 0);
    const vector<vector<Node*>> &P = data->P;
    const vector<vector<Node*>> &TP = test_data->P;
    const FtrlFloat* Wp = W.data();
    double ndcg = 0;
    FtrlLong valid_samples = 0;
    for (FtrlLong i = 0; i < m; i++) {
        const vector<Node*> p = P[i];
        const vector<Node*> tp = TP[i];
        if (!tp.size()) {
            continue;
        }
        const FtrlFloat *w = Wp+i*k;
        predict_candidates(w, Z);
        ndcg += ndcg_k(Z, p, tp, topk);
        valid_samples++;
    }
    va_loss = ndcg/double(valid_samples);
}

void FtrlProblem::predict_candidates(const FtrlFloat* w, vector<FtrlFloat> &Z) {
    FtrlInt k = param->k;
    FtrlLong n = data->n;
    FtrlFloat *Hp = H.data();
    for (FtrlLong j = 0; j < n; j++) {
        Z[j] = inner(w, Hp+j*k, k);
    }
}

bool FtrlProblem::is_hit(const vector<Node*> p, FtrlLong argmax) {
    for (Node* pp : p) {
        FtrlLong idx = pp->q_idx;
        if (idx == argmax)
            return true;
    }
    return false;
}

FtrlLong FtrlProblem::ndcg_k(vector<FtrlFloat> &Z, const vector<Node*> &p, const vector<Node*> &tp, const FtrlInt &topk) {

    FtrlInt valid_count = 0;
    double dcg = 0.0;
    double idcg = 0.0;
    while(valid_count < topk) {
        FtrlLong argmax = distance(Z.begin(), max_element(Z.begin(), Z.end()));
        if (is_hit(p, argmax)) {
            Z[argmax] = MIN_Z;
            continue;
        }
        if (is_hit(tp, argmax))
            dcg += 1/log2(valid_count+2);
        if (int (tp.size()) > valid_count)
           idcg += 1/log2(valid_count+2);
        valid_count++;
        Z[argmax] = MIN_Z;
    }
    return double(100*dcg/idcg);
}

FtrlLong FtrlProblem::precision_k(vector<FtrlFloat> &Z, const vector<Node*> &p, const vector<Node*> &tp, const FtrlInt &topk) {

    FtrlInt valid_count = 0;
    FtrlInt hit_count = 0;
    while(valid_count < topk) {
        FtrlLong argmax = distance(Z.begin(), max_element(Z.begin(), Z.end()));
        if (is_hit(p, argmax)) {
            Z[argmax] = MIN_Z;
            continue;
        }
        if (is_hit(tp, argmax)) {
            hit_count++;
        }
        valid_count++;
        Z[argmax] = MIN_Z;
    }
    return hit_count;
}


FtrlDouble FtrlProblem::cal_reg() {
    FtrlInt k = param->k;
    FtrlLong m = data->m, n = data->n;
    FtrlDouble reg = 0, lambda = param->lambda;
    const vector<vector<Node*>> &P = data->P;
    const vector<vector<Node*>> &Q = data->Q; 
    for (FtrlLong i = 0; i < m; i++) {
        FtrlLong nnz = P[i].size();
        FtrlDouble* w = W.data()+i*k;
        reg += nnz*lambda*inner(w, w, k);
    }
    for (FtrlLong i = 0; i < n; i++) {
        FtrlLong nnz = Q[i].size();
        FtrlDouble* h = H.data()+i*k;
        reg += nnz*lambda*inner(h, h, k);
    }
    return reg;
}

void FtrlProblem::update_R() {
    vector<Node> &R = data->R;
    FtrlLong l = data->l;
    FtrlInt k = param->k;
#pragma omp parallel for schedule(static)
    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &R[i];
        FtrlDouble *w = W.data()+node->p_idx*k;
        FtrlDouble *h = H.data()+node->q_idx*k;
        FtrlDouble r = inner(w, h, k);
        node->val -= r;
    }
}

void FtrlProblem::update_coordinates() {
    FtrlInt k = param->k;
    FtrlLong m = data->m, n = data->n;
    double time;
    for (FtrlInt d = 0; d < k; d++) {
         for (FtrlInt s = 0; s < 1; s++) {
            time = omp_get_wtime();
            cache_w(d);
            C_time += omp_get_wtime() - time;
            H_time += omp_get_wtime() - time;
            time = omp_get_wtime();
            for (FtrlLong j = 0; j < n; j++) {
                update_h(j, d);
            }
            H_time += omp_get_wtime() - time;
            time = omp_get_wtime();
            cache_h(d);
            C_time += omp_get_wtime() - time;
            W_time += omp_get_wtime() - time;
            time = omp_get_wtime();
            for (FtrlLong i = 0; i < m; i++) {
                update_w(i, d);
            }
            W_time += omp_get_wtime() - time;
        }
    }
}

void FtrlProblem::cache_w(FtrlInt& d) {
    FtrlLong m = data->m;
    FtrlInt k = param->k;
    FtrlFloat sq = 0, sum = 0;
    for (FtrlInt di = 0; di < k; di++) {
        wu[di] = 0;
    }
    for (FtrlInt j = 0; j < m; j++) {
        sq +=  W[j*k+d]*W[j*k+d];
        sum += W[j*k+d];
        for (FtrlInt di = 0; di < k; di++) {
            wu[di] += W[j*k+d]* W[j*k+di];
        }
    }
    w_sum[d] = sum;
    w2_sum[d] = sq;
}

void FtrlProblem::cache_h(FtrlInt& d) {
    FtrlLong n = data->n;
    FtrlInt k = param->k;
    FtrlFloat sq = 0, sum = 0;
    for (FtrlInt di = 0; di < k; di++) {
        hv[di] = 0;
    }
    for (FtrlInt j = 0; j < n; j++) {
        sq +=  H[j*k+d]*H[j*k+d];
        sum += H[j*k+d];
        for (FtrlInt di = 0; di < k; di++) {
            hv[di] += H[j*k+d]* H[j*k+di];
        }
    }
    h_sum[d] = sum;
    h2_sum[d] = sq;
}

void FtrlProblem::solve() {
    print_header_info();
    update_R();

    FtrlFloat stime = 0;
    for (t = 0; t < param->nr_pass; t++) {
        tr_loss = cal_loss(data->l, data->R);
        cout << tr_loss+cal_reg()<< endl;
        print_epoch_info();
        validate_ndcg(10);
        FtrlFloat ss = omp_get_wtime();
        update_coordinates();
        stime += (omp_get_wtime() - ss);
    }
    cout << "Total update time : " << stime << endl;
    cout << "W_time : " << W_time <<endl;
    cout << "H_time : " << H_time <<endl;
    cout << "I_time : " << I_time <<endl;
    cout << "C_time : " << C_time <<endl;
    cout << "U_time : " << U_time <<endl;
    cout << "R_time : " << R_time <<endl;
}

