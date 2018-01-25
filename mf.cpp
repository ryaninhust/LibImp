#include "mf.h"
#define MIN_Z -10000;

double inner(const double *a, const double *b, const int k)
{
    double r = 0.0;
    for (int i = 0; i < k; i++)
        r += a[i]*b[i];
    return r;
}

void FtrlProblem::save() {

    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    ofstream f(param->model_path);
    if(!f.is_open())
        return ;

    f << "m " << m << endl;
    f << "n " << n << endl;
    f << "k " << k << endl;

    auto write = [&] (FtrlFloat *ptr, FtrlLong size, char prefix)
    {
        for(FtrlLong i = 0; i < param->k ; i++)
        {
            FtrlFloat *ptr1 = ptr + i*size;
            f << prefix << i << " ";
            //if(isnan(ptr1[0]))
            if(false)
            {
                f << "F ";
                for(FtrlLong d = 0; d < size; d++)
                    f << 0 << " ";
            }
            else
            {
                f << "T ";
                for(FtrlLong d = 0; d < size; d++)
                    f << ptr1[d] << " ";
            }
            f << endl;
        }

    };

    write(WT.data(), m, 'w');
    write(HT.data(), n, 'h');

    f.close();

}

void FtrlProblem::load() {
    ifstream f(param->model_path);
    if(!f.is_open())
        return ;
    string dummy;

    f >> dummy >> data->m >> dummy >> data->n >>
         dummy >> param->k ;
    auto read = [&] (FtrlFloat  *ptr, FtrlLong size)
    {
        for(FtrlInt i = 0; i < param->k; i++)
        {
            FtrlFloat *ptr1 = ptr + i*size;
            f >> dummy >> dummy;
            if(dummy.compare("F") == 0) // nan vector starts with "F"
                for(FtrlLong  d = 0; d < size; d++)
                {
                    f >> ptr1[d];
                }
            else
                for( FtrlLong d = 0; d < size; d++)
                    f >> ptr1[d];
        }
    };

    WT.resize(param->k*data->m);
    HT.resize(param->k*data->n);

    read(WT.data(), data->m);
    read(HT.data(), data->n);

    f.close();

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

        //p_idx --;
        //q_idx --;

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
    for (FtrlLong i = 0; i < m; i++) {
        if (P[i].size())
            for (Node* p : P[i])
                RT.push_back(Node(p->p_idx, p->q_idx, p->val));
    }
    PT.resize(m);
    QT.resize(n);

    for (FtrlLong i = 0; i < l; i++) {
        Node* node = &RT[i];
        PT[node->p_idx].push_back(node);
        QT[node->q_idx].push_back(node);
    }
    R.clear();
    R.resize(l);
    for (FtrlLong i = 0; i < n; i++) {
        if (QT[i].size())
            for (Node* q : QT[i])
                R.push_back(Node(q->p_idx, q->q_idx, q->val));
    }
    P.clear();
    Q.clear();
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
    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    t = 0;
    tr_loss = 0.0, va_loss = 0.0, obj = 0.0, reg=0.0;

    WT.resize(k*m);
    HT.resize(k*n);

    wu.resize(n);
    hv.resize(m);


    default_random_engine engine(0);
    uniform_real_distribution<FtrlFloat> distribution(0, 1.0/sqrt(k));
#pragma omp parallel for schedule(static)
    for (FtrlInt d = 0; d < k; d++)
    {
        for (FtrlLong j = 0; j < m; j++) {
            WT[d*m+j] = distribution(engine); 
        }
        for (FtrlLong j = 0; j < n; j++) {
            if (data->Q[j].size())
                HT[d*n+j] = 0;
            else
                HT[d*n+j] = distribution(engine);
        }
    }
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
        cout << setprecision(3) << va_loss*100;
    }
    cout << "\n";
} 

void FtrlProblem::print_epoch_info_test() {
    cout.width(4);
    cout << t+1;
    cout.width(5);
    cout << " test";
    if (!test_data->file_name.empty()) {
        cout.width(8);
        cout << setprecision(3) << va_loss*100;
    }
    cout << "\n";
}

void FtrlProblem::update_w(FtrlLong i, FtrlDouble *wt, FtrlDouble *ht) {
    FtrlFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<Node*> &P = data->PT[i];
    FtrlDouble w_val = wt[i];
    FtrlDouble h = lambda*P.size(), g = 0;
    for (Node* p : P) {
        FtrlDouble r = p->val;
        FtrlLong j = p->q_idx;
        FtrlDouble h_val = ht[j];
        g += ((1-w)*r+w*(1-a))*h_val;
        h += (1-w)*h_val*h_val;
    }
    h += w*h2_sum;
    g += w*(a*h_sum-hv[i]+w_val*h2_sum);

    FtrlDouble new_w_val = g/h;
    //W[i*k+d] = new_w_val;
    wt[i] = new_w_val;
}

void FtrlProblem::update_h(FtrlLong j, FtrlDouble *wt, FtrlDouble *ht) {
    FtrlFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<Node*> &Q = data->Q[j];
    FtrlDouble h_val = ht[j];
    FtrlDouble h = lambda*Q.size(), g = 0;
    for (Node* q : Q) {
        FtrlDouble r = q->val;
        FtrlLong i = q->p_idx;
        FtrlDouble w_val = wt[i];
        g += ((1-w)*r+w*(1-a))*w_val;
        h += (1-w)*w_val*w_val;
    }
    h += w*w2_sum;
    g += w*(a*w_sum-wu[j]+h_val*w2_sum);

    FtrlDouble new_h_val = g/h;
   // H[j*k+d] = new_h_val;
    ht[j] = new_h_val;
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
    FtrlLong n = data->n, m = data->m;
    const vector<vector<Node*>> &P = data->PT;
    const vector<vector<Node*>> &TP = test_data->PT;
    const FtrlFloat* Wp = WT.data();
    FtrlLong hit_count = 0;
    FtrlLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples, hit_count)
    for (FtrlLong i = 0; i < m; i++) {
        vector<FtrlFloat> Z(n, 0);
        const vector<Node*> p = P[i];
        const vector<Node*> tp = TP[i];
        if (!tp.size()) {
            continue;
        }
        const FtrlFloat *w = Wp+i;
        predict_candidates(w, Z);
        hit_count += precision_k(Z, p, tp, topk);
        valid_samples++;
    }
    va_loss = hit_count/double(valid_samples*topk);
}

void FtrlProblem::validate_test(const FtrlInt &topk) {
    FtrlLong n = data->n, m = data->m;
    const vector<vector<Node*>> &P = data->PT;
    const vector<vector<Node*>> &TP = test_data_2->PT;
    const FtrlFloat* Wp = WT.data();
    FtrlLong hit_count = 0;
    FtrlLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples, hit_count)
    for (FtrlLong i = 0; i < m; i++) {
        vector<FtrlFloat> Z(n, 0);
        const vector<Node*> p = P[i];
        const vector<Node*> tp = TP[i];
        if (!tp.size()) {
            continue;
        }
        const FtrlFloat *w = Wp+i;
        predict_candidates(w, Z);
        hit_count += precision_k(Z, p, tp, topk);
        valid_samples++;
    }
    va_loss = hit_count/double(valid_samples*topk);
}

void FtrlProblem::validate_ndcg(const FtrlInt &topk) {
    FtrlLong n = data->n, m = data->m;
    const vector<vector<Node*>> &P = data->PT;
    const vector<vector<Node*>> &TP = test_data->PT;
    const FtrlFloat* Wp = WT.data();
    double ndcg = 0;
    FtrlLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples, ndcg)
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

void FtrlProblem::predict_item(const FtrlInt &topk) {
    if ( param->predict_path=="")
        param->predict_path = "result";
    FtrlLong n = data->n, m = data->m;
    const FtrlFloat* Wp = WT.data();
    ofstream f(param->predict_path);
    if(!f.is_open())
        cout<<"Writing result fail"<<endl;
        return ;

    for (FtrlLong i = 0; i < m; i++) {
        vector<FtrlFloat> Z(n, 0);
        const FtrlFloat *w = Wp+i;
        predict_candidates(w, Z);
        for (FtrlLong item = 0; item < topk; item++) {
            FtrlLong argmax = distance(Z.begin(), max_element(Z.begin(), Z.end()));
            Z[argmax] = MIN_Z;
            f << argmax << " ";
        }
        f<<endl;
    }
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
    vector<Node> &RT = data->RT;
    FtrlLong l = data->l;
    if (add) {
#pragma omp parallel for schedule(static)
        for (FtrlLong i = 0; i < l; i++ ) {
            Node r = R[i]; 
            r.val += wt[r.p_idx]*ht[r.q_idx];
        }
#pragma omp parallel for schedule(static)
        for (FtrlLong i = 0; i < l; i++ ) {
            Node r = RT[i]; 
            r.val += wt[r.p_idx]*ht[r.q_idx];
        }
    } else {
#pragma omp parallel for schedule(static)
        for (FtrlLong i = 0; i < l; i++ ) {
            Node r = R[i]; 
            r.val -= wt[r.p_idx]*ht[r.q_idx];
        }
#pragma omp parallel for schedule(static)
        for (FtrlLong i = 0; i < l; i++ ) {
            Node r = RT[i]; 
            r.val -= wt[r.p_idx]*ht[r.q_idx];
        }
    }
}


void FtrlProblem::update_coordinates() {
    FtrlInt k = param->k;
    FtrlLong m = data->m, n = data->n;
    FtrlInt nr_th = param->nr_threads;
    vector<FtrlDouble> hv_th(m*nr_th,0.0);
    vector<FtrlDouble> wu_th(n*nr_th,0.0);
    for (FtrlInt d = 0; d < k; d++) {
         FtrlDouble *u = &WT[d*m];
         FtrlDouble *v = &HT[d*n];
         update_R(u, v, true);
         for (FtrlInt s = 0; s < 5; s++) {
            cache_w(u, wu_th.data());
#pragma omp parallel for schedule(guided)
            for (FtrlLong j = 0; j < n; j++) {
                if (data->Q[j].size())
                    update_h(j, u, v);
            }
            cache_h(v, hv_th.data());
#pragma omp parallel for schedule(guided)
            for (FtrlLong i = 0; i < m; i++) {
                if (data->P[i].size())
                    update_w(i, u, v);
            }
        }
        update_R(u, v, false); 
    }
}

void FtrlProblem::cache_w(FtrlDouble *wt, FtrlDouble *wu_th) {
    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    FtrlFloat sq = 0, sum = 0;
    FtrlInt nr_th = param->nr_threads;
#pragma omp parallel for schedule(static)
    for (FtrlInt num_th = 0; num_th < nr_th; num_th++)
        for (FtrlLong i = 0; i < n; i++)
            wu_th[n*num_th+i] = 0;
#pragma omp parallel for schedule(static)
    for (FtrlLong i = 0; i < n; i++) {
        wu[i] = 0;
    }
#pragma omp parallel for schedule(static) reduction(+:sq,sum)
    for (FtrlInt j = 0; j < m; j++) {
        sq +=  wt[j]*wt[j];
        sum += wt[j];
    }
#pragma omp parallel for schedule(static)
    for (FtrlInt di = 0; di < k; di++) {
        FtrlInt num_th = omp_get_thread_num();
        FtrlDouble uTWt = 0;
        for (FtrlLong j = 0; j < m; j++) {
            uTWt += wt[j] * WT[di*m+j];
        }
        for (FtrlLong i = 0; i < n; i++) {
            wu_th[n*num_th+i] += uTWt * HT[di*n+i];
        }
    }
#pragma omp parallel for schedule(static)
    for (FtrlLong i = 0; i < n; i++)
        for(FtrlInt num_th = 0; num_th < nr_th; num_th++)
            wu[i] += wu_th[num_th*n+i];
    w_sum = sum;
    w2_sum = sq;
}

void FtrlProblem::cache_h(FtrlDouble *ht, FtrlDouble *hv_th) {
    FtrlLong m = data->m, n = data->n;
    FtrlInt k = param->k;
    FtrlFloat sq = 0, sum = 0;
    FtrlInt nr_th =  param->nr_threads;
#pragma omp parallel for schedule(static)
    for (FtrlInt num_th = 0; num_th < nr_th; num_th++)
        for (FtrlLong j = 0; j < m; j++)
            hv_th[m*num_th+j] = 0;
#pragma omp parallel for schedule(static)
    for (FtrlLong j = 0; j < m; j++) {
        hv[j] = 0;
    }
#pragma omp parallel for schedule(static) reduction(+:sq,sum)
    for (FtrlInt j = 0; j < n; j++) {
        sq +=  ht[j]*ht[j];
        sum += ht[j];
    }
#pragma omp parallel for schedule(static)
    for (FtrlInt di = 0; di < k; di++) {
        FtrlDouble uTWt = 0;
        FtrlInt num_th = omp_get_thread_num();
        for (FtrlLong i = 0; i < n; i++) {
            uTWt += ht[i] * HT[di*n+i];
        }
        for (FtrlLong j = 0; j < m; j++) {
            hv_th[m*num_th+j] += uTWt * WT[di*m+j];
        }
    }
#pragma omp parallel for schedule(static)
    for (FtrlLong j = 0; j < m; j++) {
        for(FtrlInt num_th = 0; num_th < nr_th; num_th++) {
            hv[j] += hv_th[m*num_th+j];
        }
    }
    h_sum = sum;
    h2_sum = sq;
}

void FtrlProblem::solve() {
    cout<<"Using "<<param->nr_threads<<" threads"<<endl;
    print_header_info();
    update_R();
    for (t = 0; t < param->nr_pass; t++) {
        update_coordinates();
        validate(10);
        print_epoch_info();
        if (t%3 == 2 && test_with_two_data) {
            validate_test(10);
            print_epoch_info_test();
        }
    }
    if (param->predict_path != "")
        predict_item(10);
    if (param->model_path != "")
        save();
}

