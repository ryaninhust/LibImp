#include "mf.h"

double inner(double *a, double *b, int k)
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
    cout << endl;
}

void FtrlProblem::initialize() {
    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    t = 0;
    tr_loss = 0.0, va_loss = 0.0, obj = 0.0, reg=0.0;
    W.resize(k*m);
    H.resize(k*n);

    default_random_engine engine(0);
    uniform_real_distribution<FtrlFloat> distribution(0, 1.0/sqrt(k));

    for (FtrlInt d = 0; d < k; d++)
    {
        for (FtrlLong j = 0; j < m; j++)
            W[j*k+d] = distribution(engine);
        for (FtrlLong j = 0; j < n; j++)
            H[j*k+d] = distribution(engine);
    }
    start_time = omp_get_wtime();
}

void FtrlProblem::print_header_info() {
    cout.width(4);
    cout << "iter";
    cout.width(13);
    cout << "tr_rmse";
    if (!test_data->file_name.empty()) {
        cout.width(13);
        cout << "va_rmse";
    }
    cout.width(13);
    cout << "obj";
    cout.width(13);
    cout << "reg";
    cout << "\n";
}

void FtrlProblem::print_epoch_info() {
    cout.width(4);
    cout << t+1;
    cout.width(13);
    cout << fixed << setprecision(4);
    cout << setprecision(4) << tr_loss;
    if (!test_data->file_name.empty()) {
        cout.width(13);
        cout << setprecision(4) << va_loss;
    }
    cout.width(13);
    cout << scientific << obj;
    cout.width(13);
    cout << scientific << reg;
    cout << "\n";
} 


void FtrlProblem::update_w(FtrlLong i, FtrlInt d) {
    
    FtrlInt k = param->k;
    FtrlFloat lambda = param->lambda;
    const vector<vector<Node*>> &P = data->P;
    FtrlDouble w_val = W[i*k+d];
    FtrlDouble h = lambda*P[i].size(), g = 0;
    for (Node* p : P[i]) {
        FtrlDouble r = p->val;
        FtrlLong j = p->q_idx;
        FtrlDouble h_val = H[j*k+d];
        g += (r+h_val*w_val)*h_val;
        h += h_val*h_val;
    }
    FtrlDouble new_w_val = g/h;
    for (Node* p : P[i]) {
        FtrlLong j = p->q_idx;
        FtrlDouble h_val = H[j*k+d];
        p->val += (w_val-new_w_val)*h_val;
    }
    W[i*k+d] = new_w_val;
}

void FtrlProblem::update_h(FtrlLong j, FtrlInt d) {
    FtrlInt k = param->k;
    FtrlFloat lambda = param->lambda;
    const vector<vector<Node*>> &Q = data->Q;
    FtrlDouble h_val = H[j*k+d];
    FtrlDouble h = lambda*Q[j].size(), g = 0;
    for (Node* q : Q[j]) {
        FtrlDouble r = q->val;
        FtrlLong i = q->p_idx;
        FtrlDouble w_val = W[i*k+d];
        g += (r+h_val*w_val)*w_val;
        h += w_val*w_val;
    }
    FtrlDouble new_h_val = g/h;
    for (Node* q : Q[j]) {
        FtrlLong i = q->p_idx;
        FtrlDouble w_val = W[i*k+d];
        q->val += (h_val-new_h_val)*w_val;
    }
    H[j*k+d] = new_h_val;
}


FtrlDouble FtrlProblem::cal_loss(FtrlLong &l, vector<Node> &R) {
    FtrlInt k = param->k;
    FtrlDouble loss = 0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &R[i];
        if (node->p_idx+1>data->m || node->q_idx+1>data->n)
            continue;
        FtrlDouble *w = W.data()+node->p_idx*k;
        FtrlDouble *h = H.data()+node->q_idx*k;
        FtrlDouble r = inner(w, h, k);
        loss += (r-node->val)*(r-node->val);
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
    for (FtrlInt d = 0; d < k; d++) {
        for (FtrlLong j = 0; j < n; j++) {
            update_h(j, d);
        }
        for (FtrlLong i = 0; i < m; i++) {
            update_w(i, d);
        }
    }
}

void FtrlProblem::solve() {
    print_header_info();
    update_R();

    for (t = 0; t < param->nr_pass; t++) {
        tr_loss = cal_tr_loss(data->l, data->R);
        reg = cal_reg();
        obj = tr_loss + reg;
        tr_loss = sqrt(tr_loss/data->l);
        va_loss = sqrt(cal_loss(test_data->l, test_data->R)/test_data->l);
        print_epoch_info();
        update_coordinates();
    }
}

