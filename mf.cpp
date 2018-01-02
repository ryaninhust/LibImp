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

        p_idx --;
        q_idx --;

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
    //W.resize(k*m);
    //H.resize(k*n);

    WT.resize(k*m);
    HT.resize(k*n);

    wu.resize(n);
    hv.resize(m);


    default_random_engine engine(0);
    uniform_real_distribution<FtrlFloat> distribution(0, 1.0/sqrt(k));

    for (FtrlInt d = 0; d < k; d++)
    {
        for (FtrlLong j = 0; j < m; j++) {
            //W[j*k+d] = distribution(engine);
            WT[d*m+j] = distribution(engine); 
            //W[j*k+d] = 1/sqrt(k);
            //WT[d*m+j] = W[j*k+d];
        }
        for (FtrlLong j = 0; j < n; j++) {
            if (data->Q[j].size() != 0 )
               // H[j*k+d] = 0;
                HT[d*n+j] = 0;
            else
                HT[d*n+j] = distribution(engine);
                //H[j*k+d] = distribution(engine);
            //HT[d*n+j] = H[j*k+d];
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


void FtrlProblem::update_w(FtrlLong i, FtrlDouble *wt, FtrlDouble *ht) {
    //double time = omp_get_wtime();    
    //FtrlInt k = param->k;
    FtrlFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<Node*> &P = data->P[i];
    //FtrlLong m = data->m, n = data->n;
    FtrlDouble w_val = wt[i];
    FtrlDouble h = lambda*P.size(), g = 0;
    for (Node* p : P) {
        FtrlDouble r = p->val;
        FtrlLong j = p->q_idx;
        FtrlDouble h_val = ht[j];
        //TODO change r -> r hat
        g += ((1-w)*r+w*(1-a))*h_val;
        h += (1-w)*h_val*h_val;
    }
    h += w*h2_sum;
    g += w*(a*h_sum-hv[i]+w_val*h2_sum);

    FtrlDouble new_w_val = g/h;
    //U_time += omp_get_wtime() - time;
    //time = omp_get_wtime();
    /*for (Node* p : P) {
        FtrlLong j = p->q_idx;
        FtrlDouble h_val = HT[d*n+j];
        p->val += (w_val-new_w_val)*h_val;
    }*/
    //W[i*k+d] = new_w_val;
    wt[i] = new_w_val;
    //R_time += omp_get_wtime() - time;
}

void FtrlProblem::update_h(FtrlLong j, FtrlDouble *wt, FtrlDouble *ht) {
    //double time = omp_get_wtime();
    //FtrlInt k = param->k;
    FtrlFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<Node*> &Q = data->Q[j];
    //FtrlLong m = data->m, n = data->n;
    FtrlDouble h_val = ht[j];
    FtrlDouble h = lambda*Q.size(), g = 0;
    for (Node* q : Q) {
        FtrlDouble r = q->val;
        FtrlLong i = q->p_idx;
        FtrlDouble w_val = wt[i];
        //TODO change r -> r hat 
        g += ((1-w)*r+w*(1-a))*w_val;
        h += (1-w)*w_val*w_val;
    }
    h += w*w2_sum;
    g += w*(a*w_sum-wu[j]+h_val*w2_sum);

    FtrlDouble new_h_val = g/h;
    //if (j >= 0 && j <10)
    //     printf("U : %f, D : %f, H : %f\n", g, h, g/h);
    //U_time += omp_get_wtime() - time;
    //time = omp_get_wtime();
    /*for (Node* q : Q) {
        FtrlLong i = q->p_idx;
        FtrlDouble w_val = WT[d*m+i];
        q->val += (h_val-new_h_val)*w_val;
    }*/
   // H[j*k+d] = new_h_val;
    ht[j] = new_h_val;
    //R_time += omp_get_wtime() - time;
}


FtrlDouble FtrlProblem::cal_loss(FtrlLong &l, vector<Node> &R) {
    //TODO change W H to WT HT
    FtrlInt k = param->k;
    FtrlDouble loss = 0, a = param->a;
    FtrlLong m = data->m, n = data->n;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &R[i];
        if (node->p_idx+1>data->m || node->q_idx+1>data->n)
            continue;
        FtrlDouble *w = WT.data()+node->p_idx;
        FtrlDouble *h = HT.data()+node->q_idx;
        FtrlDouble r = 0;
        for (FtrlInt d = 0; d < k; d++)
            r += w[d*m] * h[d*n];
        loss += (r-node->val)*(r-node->val);
        loss -= param->w*(a-r)*(a-r);
    }
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (FtrlLong i = 0; i < m; i++) {
        for (FtrlLong j = 0; j < n; j++) {
            FtrlDouble *w = WT.data()+i;
            FtrlDouble *h = HT.data()+j;
            FtrlDouble r = 0.0;
            for (FtrlInt d = 0; d < k; d++)
                r += w[d*m] * h[d*n];
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
    //FtrlInt k = param->k;
    FtrlLong n = data->n, m = data->m;
    const vector<vector<Node*>> &P = data->P;
    const vector<vector<Node*>> &TP = test_data->P;
    const FtrlFloat* Wp = WT.data();
    double ndcg = 0;
    FtrlLong valid_samples = 0;
    for (FtrlLong i = 0; i < m; i++) {
        vector<FtrlFloat> Z(n, 0);
        const vector<Node*> p = P[i];
        const vector<Node*> tp = TP[i];
        if (!tp.size()) {
            continue;
        }
        const FtrlFloat *w = Wp+i;
        predict_candidates(w, Z);
        ndcg += ndcg_k(Z, p, tp, topk);
        valid_samples++;
    }
    va_loss = ndcg/double(valid_samples);
}

void FtrlProblem::predict_candidates(const FtrlFloat* w, vector<FtrlFloat> &Z) {
    FtrlInt k = param->k;
    FtrlLong n = data->n, m = data->m;
    FtrlFloat *Hp = HT.data();
    for(FtrlInt d = 0; d < k; d++) {
        for (FtrlLong j = 0; j < n; j++) {
            Z[j] += w[d*m]*Hp[d*n+j];
        }
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

FtrlDouble FtrlProblem::ndcg_k(vector<FtrlFloat> &Z, const vector<Node*> &p, const vector<Node*> &tp, const FtrlInt &topk) {

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
    //TODO change W to WT
    for (FtrlLong i = 0; i < m; i++) {
        FtrlLong nnz = P[i].size();
        FtrlDouble* w = WT.data()+i;
        FtrlDouble inner = 0.0;
        for (FtrlInt d = 0; d < k ; d++)
            inner += w[d*m] * w[d*m];
        reg += nnz*lambda*inner;
    }
    //TODO change H to HT
    for (FtrlLong i = 0; i < n; i++) {
        FtrlLong nnz = Q[i].size();
        FtrlDouble* h = HT.data()+i;
        FtrlDouble inner = 0.0;
        for (FtrlInt d = 0; d < k ; d++)
            inner += h[d*n]*h[d*n];
        reg += nnz*lambda*inner;
    }
    return reg;
}

void FtrlProblem::update_R() {
    vector<Node> &R = data->R;
    FtrlLong l = data->l, m = data->m, n = data->n;
    FtrlInt k = param->k;
#pragma omp parallel for schedule(static)
    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &R[i];
        FtrlDouble *w = WT.data()+node->p_idx;
        FtrlDouble *h = HT.data()+node->q_idx;
        FtrlDouble r = 0.0;
        for (FtrlInt d = 0; d < k ; d++)
            r += w[d*m]*h[d*n];
        node->val -= r;
    }
}

void FtrlProblem::update_R(FtrlDouble *wt, FtrlDouble *ht, bool add) {
    vector<Node> &R = data->R;
    if (add)
        for (Node r : R) 
            r.val += wt[r.p_idx]*ht[r.q_idx];
    else
        for (Node r : R) 
            r.val -= wt[r.p_idx]*ht[r.q_idx];
}


void FtrlProblem::update_coordinates() {
    FtrlInt k = param->k;
    FtrlLong m = data->m, n = data->n;
    double time;
    for (FtrlInt d = 0; d < k; d++) {
         //TODO create hat{R} 
         FtrlDouble *u = &WT[d*m];
         FtrlDouble *v = &HT[d*n];
         update_R(u, v, true);
         //cout<<"rank: "<<d<<endl;
         for (FtrlInt s = 0; s < 5; s++) {
            time = omp_get_wtime();
            cache_w(u);
            C_time += omp_get_wtime() - time;
            H_time += omp_get_wtime() - time;
            time = omp_get_wtime();
            for (FtrlLong j = 0; j < n; j++) {
                if (data->Q[j].size())
                    update_h(j, u, v);
            }
            H_time += omp_get_wtime() - time;
            time = omp_get_wtime();
            cache_h(v);
            C_time += omp_get_wtime() - time;
            W_time += omp_get_wtime() - time;
            time = omp_get_wtime();
            for (FtrlLong i = 0; i < m; i++) {
                if (data->P[i].size())
                    update_w(i, u, v);
            }
            W_time += omp_get_wtime() - time;
        }
        time = omp_get_wtime();
        update_R(u, v, false); 
        R_time += omp_get_wtime() - time;
    }
}

void FtrlProblem::cache_w(FtrlDouble *wt) {
    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    FtrlFloat sq = 0, sum = 0;
    for (FtrlLong i = 0; i < n; i++) {
        wu[i] = 0;
    }
    for (FtrlInt j = 0; j < m; j++) {
        sq +=  wt[j]*wt[j];
        sum += wt[j];
    }
    for (FtrlInt di = 0; di < k; di++) {
        FtrlDouble uTWt = 0;
        for (FtrlLong j = 0; j < m; j++) {
            uTWt += wt[j] * WT[di*m+j];
        }
        for (FtrlLong i = 0; i < n; i++) {
            wu[i] += uTWt * HT[di*n+i];
        }
    }
    w_sum = sum;
    w2_sum = sq;
}

void FtrlProblem::cache_h(FtrlDouble *ht) {
    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    FtrlFloat sq = 0, sum = 0;
    for (FtrlLong j = 0; j < m; j++) {
        hv[j] = 0;
    }
    for (FtrlInt j = 0; j < n; j++) {
        sq +=  ht[j]*ht[j];
        sum += ht[j];
    }
    for (FtrlInt di = 0; di < k; di++) {
        FtrlDouble uTWt = 0;
        for (FtrlLong i = 0; i < n; i++) {
            uTWt += ht[i] * HT[di*n+i];
        }
        for (FtrlLong j = 0; j < m; j++) {
            hv[j] += uTWt * WT[di*m+j];
        }
    }

    h_sum = sum;
    h2_sum = sq;
}

void FtrlProblem::solve() {
    print_header_info();
    update_R();
    FtrlFloat test_time = 0;
    FtrlFloat stime = 0;
    for (t = 0; t < param->nr_pass; t++) {
        FtrlFloat ss1 = omp_get_wtime();
        tr_loss = cal_loss(data->l, data->R);
        cout << tr_loss+cal_reg()<< endl;
        validate_ndcg(10);
        test_time += (omp_get_wtime() - ss1);
        print_epoch_info();
        FtrlFloat ss = omp_get_wtime();
        update_coordinates();
        stime += (omp_get_wtime() - ss);
    }
    cout << "Total update time : " << stime << endl;
    cout << "Total test time : " << test_time << endl;
    cout << "W_time : " << W_time <<endl;
    cout << "H_time : " << H_time <<endl;
    cout << "I_time : " << I_time <<endl;
    cout << "C_time : " << C_time <<endl;
    cout << "U_time : " << U_time <<endl;
    cout << "R_time : " << R_time <<endl;
}

