#include "mf.h"
#include <cstring>
#define MIN_Z -10000;

double inner(const double *a, const double *b, const int k)
{
    double r = 0.0;
    for (int i = 0; i < k; i++)
        r += a[i]*b[i];
    return r;
}

void ImpProblem::save() {

    if (param->model_path == "") {
        const char *ptr = strrchr(&*data->file_name.begin(), '/');
        if(!ptr)
            ptr = data->file_name.c_str();
        else
            ++ptr;
        param->model_path = string(ptr) + ".model";
    }
    ImpLong m = data->m, n = data->n;
    ImpInt k = param->k;
    ofstream f(param->model_path);
    if(!f.is_open())
        return ;
    f << "f " << 0 << endl;
    f << "m " << m << endl;
    f << "n " << n << endl;
    f << "k " << k << endl;
    f << "b " << 0 << endl;
    auto write = [&] (ImpFloat *ptr, ImpLong size, char prefix)
    {
        for(ImpLong i = 0; i < size ; i++)
        {
            ImpFloat *ptr1 = ptr + i ;
            f << prefix << i << " ";
            //if(isnan(ptr1[0]))
            if(false)
            {
                f << "F ";
                for(ImpLong d = 0; d < param->k; d++)
                    f << 0 << " ";
            }
            else
            {
                f << "T ";
                for(ImpLong d = 0; d < param->k; d++)
                    f << ptr1[d*size] << " ";
            }
            f << endl;
        }

    };

    write(WT.data(), m, 'w');
    write(HT.data(), n, 'h');

    f.close();

}

void ImpProblem::load() {
    ifstream f(param->model_path);
    if(!f.is_open())
        return ;
    string dummy;

    f >> dummy >> dummy >> dummy >> data->m >> dummy >> data->n >>
         dummy >> param->k >> dummy >> dummy;
    auto read = [&] (ImpFloat  *ptr, ImpLong size)
    {
        for(ImpInt i = 0; i < size; i++)
        {
            ImpFloat *ptr1 = ptr + i;
            f >> dummy >> dummy;
            if(dummy.compare("F") == 0) // nan vector starts with "F"
                for(ImpLong  d = 0; d < param->k; d++)
                {
                    f >> ptr1[d*size];
                }
            else
                for( ImpLong d = 0; d < param->k; d++)
                    f >> ptr1[d*size];
        }
    };

    WT.resize(param->k*data->m);
    HT.resize(param->k*data->n);

    read(WT.data(), data->m);
    read(HT.data(), data->n);

    f.close();

}

void ImpData::read() {
    string line;
    ifstream fs(file_name);
    //TODO read l m n;
    l = 100000;
    m = 3000;
    n = 2000;
    R.row_ptr.resize(m+1);
    RT.row_ptr.resize(n+1);
    R.col_idx.resize(l);
    RT.col_idx.resize(l);
    R.val.resize(l);
    RT.val.resize(l);
    vector<ImpLong> perm(l);
    ImpLong idx = 0;
    while (getline(fs, line)) {
        istringstream iss(line);
        
        ImpLong p_idx, q_idx;
        iss >> p_idx;
        iss >> q_idx;

        ImpFloat val;
        iss >> val;

        R.row_ptr[p_idx+1]++;
        RT.row_ptr[q_idx+1]++;
        R.col_idx[idx]  = p_idx;
        RT.col_idx[idx] = q_idx;
        RT.val[idx] = val;
        perm[idx] = idx;
    }
    sort(perm.begin(), perm.end(), compare);
    for(idx = 0; idx < l; idx++ ) {
       R.col_idx[idx] = R.col_idx[perm[idx]];
       R.val[idx] = RT.val[perm[idx]];
    }
    for(ImpLong i = 1; i < m+1; ++i) {
        R.row_ptr[i] += R.row_ptr[i-1];
    }
    for(ImpLong j = 1; j < n+1; ++i) {
        RT.row_ptr[j] += RT.row_ptr[j-1];
    }
    for(ImpLong i = 0; i < m; ++i) {
        for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; j++) {
            ImpLong c = R.col_idx[j];
            RT.val[RT.row_ptr[c]] = R.val[j];
            RT.row_ptr[c]++;
    }
    for(ImpLong j = n; j > 0; --j)
        RT.row_ptr[j] = RT.row_ptr[j-1];
    Rt.row_ptr[0] = 0;
}

bool compare(size_t i, size_t j) {
    return R.col_idx[i] < R.col_idx[j] || (R.col_idx == R.col_idx[j] && RT.col_idx[i] < RT.col_idx[j]);
}

void ImpData::print_data_info() {
    cout << "Data: " << file_name << "\t";
    cout << "#m: " << m << "\t";
    cout << "#n: " << n << "\t";
    cout << "#l: " << l << "\t";
    cout << endl;
}

void ImpProblem::initialize() {
    ImpLong m = data->m, n = data->n;
    ImpInt k = param->k;
    t = 0;
    tr_loss = 0.0; obj = 0.0, reg=0.0;

    W.resize(m*k);
    H.resize(n*k);

    WT.resize(k*m);
    HT.resize(k*n);

    wu.resize(n);
    hv.resize(m);


    default_random_engine engine(0);
    uniform_real_distribution<ImpFloat> distribution(0, 1.0/sqrt(k));
#pragma omp parallel for schedule(static)
    for (ImpInt d = 0; d < k; d++)
    {
        for (ImpLong j = 0; j < m; j++) {
            WT[d*m+j] = distribution(engine); 
            W[j*k+d] = WT[d*m+j];
        }
        for (ImpLong j = 0; j < n; j++) {
            if (data->Q[j].size()) {
                HT[d*n+j] = 0;
                H[j*k+d] = HT[d*n+j];
            } else {
                HT[d*n+j] = distribution(engine);
                H[j*k+d] = HT[d*n+j];
            }
        }
    }
    start_time = omp_get_wtime();
}

void ImpProblem::init_va_loss(ImpInt size) {
    va_loss.resize(size);
    for (ImpInt i = 0; i < size ; i++) {
        va_loss[i] = 0.0;
    }
}

void ImpProblem::print_header_info(vector<ImpInt> &topks) {
    cout.width(4);
    cout << "iter";
    if (!test_data->file_name.empty()) {
        for (ImpInt i = 0; i < ImpInt(va_loss.size()); i++ ) {
            cout.width(12);
            cout << "va_p@" << topks[i];
        }
    }
    cout << endl;
}

void ImpProblem::print_epoch_info() {
    cout.width(4);
    cout << t+1;
    if (!test_data->file_name.empty()) {
        for (ImpInt i = 0; i < ImpInt(va_loss.size()); i++ ) {
            cout.width(13);
            cout << setprecision(3) << va_loss[i]*100;
        }
    }
    cout << endl;
} 

void ImpProblem::update_w(ImpLong i, ImpDouble *wt, ImpDouble *ht) {
    ImpFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<Node*> &P = data->PT[i];
    ImpDouble w_val = wt[i];
    ImpDouble h = lambda*P.size(), g = 0;
    for (Node* p : P) {
        ImpDouble r = p->val;
        ImpLong j = p->q_idx;
        ImpDouble h_val = ht[j];
        g += ((1-w)*r+w*(1-a))*h_val;
        h += (1-w)*h_val*h_val;
    }
    h += w*h2_sum;
    g += w*(a*h_sum-hv[i]+w_val*h2_sum);

    ImpDouble new_w_val = g/h;
    //W[i*k+d] = new_w_val;
    wt[i] = new_w_val;
}

void ImpProblem::update_h(ImpLong j, ImpDouble *wt, ImpDouble *ht) {
    ImpFloat lambda = param->lambda, a = param->a, w = param->w;
    const vector<Node*> &Q = data->Q[j];
    ImpDouble h_val = ht[j];
    ImpDouble h = lambda*Q.size(), g = 0;
    for (Node* q : Q) {
        ImpDouble r = q->val;
        ImpLong i = q->p_idx;
        ImpDouble w_val = wt[i];
        g += ((1-w)*r+w*(1-a))*w_val;
        h += (1-w)*w_val*w_val;
    }
    h += w*w2_sum;
    g += w*(a*w_sum-wu[j]+h_val*w2_sum);

    ImpDouble new_h_val = g/h;
   // H[j*k+d] = new_h_val;
    ht[j] = new_h_val;
}


ImpDouble ImpProblem::cal_loss(ImpLong &l, vector<Node> &R) {
    ImpInt k = param->k;
    ImpDouble loss = 0, a = param->a;
    ImpLong m = data->m, n = data->n;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong i = 0; i < l; i++) {
        Node* node = &R[i];
        if (node->p_idx+1>data->m || node->q_idx+1>data->n)
            continue;
        ImpDouble *w = WT.data()+node->p_idx;
        ImpDouble *h = HT.data()+node->q_idx;
        ImpDouble r = 0;
        for (ImpInt d = 0; d < k; d++)
            r += w[d*m] * h[d*n];
        loss += (r-node->val)*(r-node->val);
        loss -= param->w*(a-r)*(a-r);
    }
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong i = 0; i < m; i++) {
        for (ImpLong j = 0; j < n; j++) {
            ImpDouble *w = WT.data()+i;
            ImpDouble *h = HT.data()+j;
            ImpDouble r = 0.0;
            for (ImpInt d = 0; d < k; d++)
                r += w[d*m] * h[d*n];
            loss += param->w*(a-r)*(a-r);
        }
    }
    return loss;
}

ImpDouble ImpProblem::cal_tr_loss(ImpLong &l, vector<Node> &R) {
    ImpDouble loss = 0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong i = 0; i < l; i++) {
        Node* node = &R[i];
        loss += node->val*node->val;
    }
    return loss;
}

void ImpProblem::validate(const vector<ImpInt> &topks) {
    ImpLong n = data->n, m = data->m;
    ImpInt nr_th = param->nr_threads;
    const vector<vector<Node*>> &P = data->PT;
    const vector<vector<Node*>> &TP = test_data->PT;
    const ImpFloat* Wp = WT.data();
    vector<ImpLong> hit_counts(nr_th*topks.size(),0);
    ImpLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < m; i++) {
        vector<ImpFloat> Z(n, 0);
        const vector<Node*> p = P[i];
        const vector<Node*> tp = TP[i];
        if (!tp.size()) {
            continue;
        }
        const ImpFloat *w = Wp+i;
        predict_candidates(w, Z);
        precision_k(Z, p, tp, topks, hit_counts);
        valid_samples++;
    }
    for (ImpInt i = 0; i < int(topks.size()); i++) {
        va_loss[i] = 0;
    }

    for (ImpLong num_th = 0; num_th < nr_th; num_th++) {
        for (ImpInt i = 0; i < int(topks.size()); i++) {
            va_loss[i] += hit_counts[i+num_th*topks.size()];
        }
    }

    for (ImpInt i = 0; i < int(topks.size()); i++) {
        va_loss[i] /= double(valid_samples*topks[i]);
    }
}

void ImpProblem::validate_ndcg(const vector<ImpInt> &topks) {
    ImpLong n = data->n, m = data->m;
    ImpInt nr_th = param->nr_threads;
    const vector<vector<Node*>> &P = data->PT;
    const vector<vector<Node*>> &TP = test_data->PT;
    const ImpFloat* Wp = WT.data();
    vector<double> ndcgs(nr_th*topks.size(),0);
    ImpLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < m; i++) {
        vector<ImpFloat> Z(n, 0);
        const vector<Node*> p = P[i];
        const vector<Node*> tp = TP[i];
        if (!tp.size()) {
            continue;
        }
        const ImpFloat *w = Wp+i;
        predict_candidates(w, Z);
        ndcg_k(Z, p, tp, topks, ndcgs);
        valid_samples++;
    }
    for (ImpInt i = 0; i < int(topks.size()); i++) {
        va_loss[i] = 0;
    }

    for (ImpLong num_th = 0; num_th < nr_th; num_th++) {
        for (ImpInt i = 0; i < int(topks.size()); i++) {
            va_loss[i] += ndcgs[i+num_th*topks.size()];
        }
    }

    for (ImpInt i = 0; i < int(topks.size()); i++) {
        va_loss[i] /= double(valid_samples);
    }
}

void ImpProblem::predict_candidates(const ImpFloat* w, vector<ImpFloat> &Z) {
    ImpInt k = param->k;
    ImpLong n = data->n, m = data->m;
    ImpFloat *Hp = HT.data();
    for(ImpInt d = 0; d < k; d++) {
        for (ImpLong j = 0; j < n; j++) {
            Z[j] += w[d*m]*Hp[d*n+j];
        }
    }
}

bool ImpProblem::is_hit(const vector<Node*> p, ImpLong argmax) {
    for (Node* pp : p) {
        ImpLong idx = pp->q_idx;
        if (idx == argmax)
            return true;
    }
    return false;
}

ImpDouble ImpProblem::ndcg_k(vector<ImpFloat> &Z, const vector<Node*> &p, const vector<Node*> &tp, const vector<ImpInt> &topks, vector<double> &ndcgs) {
    ImpInt state = 0;
    ImpInt valid_count = 0;
    vector<double> dcg(topks.size(),0.0);
    vector<double> idcg(topks.size(),0.0);
    ImpInt num_th = omp_get_thread_num();
    while(state < int(topks.size()) ) {
        while(valid_count < topks[state]) {
            ImpLong argmax = distance(Z.begin(), max_element(Z.begin(), Z.end()));
            if (is_hit(p, argmax)) {
                Z[argmax] = MIN_Z;
                continue;
            }
            if (is_hit(tp, argmax))
                dcg[state] += 1/log2(valid_count+2);
            if (int (tp.size()) > valid_count)
                idcg[state] += 1/log2(valid_count+2);
            valid_count++;
            Z[argmax] = MIN_Z;
        }
        state++;
    }

    for (ImpInt i = 1; i < int(topks.size()); i++) {
        dcg[i] += dcg[i-1];
        idcg[i] += idcg[i-1];
    }

    for (ImpInt i = 0; i < int(topks.size()); i++) {
        ndcgs[i+num_th*topks.size()] += dcg[i]/idcg[i];
    }
    return 0.0;
    //return double(dcg/idcg);
}

ImpLong ImpProblem::precision_k(vector<ImpFloat> &Z, const vector<Node*> &p, const vector<Node*> &tp, const vector<ImpInt> &topks, vector<ImpLong> &hit_counts) {
    ImpInt state = 0;
    ImpInt valid_count = 0;
    vector<ImpInt> hit_count(topks.size(), 0);
    ImpInt num_th = omp_get_thread_num();
    while(state < int(topks.size()) ) {
        while(valid_count < topks[state]) {
            ImpLong argmax = distance(Z.begin(), max_element(Z.begin(), Z.end()));
            if (is_hit(tp, argmax)) {
                hit_count[state]++;
            }
            valid_count++;
            Z[argmax] = MIN_Z;
        }
        state++;
    }

    for (ImpInt i = 1; i < int(topks.size()); i++) {
        hit_count[i] += hit_count[i-1];
    }
    for (ImpInt i = 0; i < int(topks.size()); i++) {
        hit_counts[i+num_th*topks.size()] += hit_count[i];
    }
    return 0;
    //return hit_count;
}


ImpDouble ImpProblem::cal_reg() {
    ImpInt k = param->k;
    ImpLong m = data->m, n = data->n;
    ImpDouble reg = 0, lambda = param->lambda;
    const vector<vector<Node*>> &P = data->P;
    const vector<vector<Node*>> &Q = data->Q; 
    //TODO change W to WT
    for (ImpLong i = 0; i < m; i++) {
        ImpLong nnz = P[i].size();
        ImpDouble* w = WT.data()+i;
        ImpDouble inner = 0.0;
        for (ImpInt d = 0; d < k ; d++)
            inner += w[d*m] * w[d*m];
        reg += nnz*lambda*inner;
    }
    //TODO change H to HT
    for (ImpLong i = 0; i < n; i++) {
        ImpLong nnz = Q[i].size();
        ImpDouble* h = HT.data()+i;
        ImpDouble inner = 0.0;
        for (ImpInt d = 0; d < k ; d++)
            inner += h[d*n]*h[d*n];
        reg += nnz*lambda*inner;
    }
    return reg;
}
/*
void ImpProblem::update_R() {
    smat &R = data->R;
    smat &RT = data->RT;
    ImpLong l = data->l, m = data->m, n = data->n;
    ImpInt k = param->k;
#pragma omp parallel for schedule(guided)
    for (ImpLong i = 0; i < m; i++) {
        for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; ++j) {
            ImpDouble *w = W.data()+i*k;
            ImpDouble *h = H.data()+R.col_idx[j]*k;
            ImpDouble r = 0.0;
            for (ImpInt d = 0; d < k ; ++d)
                r += w[d]*h[d];
            R.val[j] -= r;
        }
    }
#pragma omp parallel for schedule(guided)
    for (ImpLong j = 0; j < n; j++) {
        for(ImpLong i = RT.row_ptr[j]; i < RT.row_ptr[j+1]; ++i) {
            Node* node = &RT[i];
            ImpDouble *w = W.data()+R.col_idx[i]*k;
            ImpDouble *h = H.data()+j*k;
            ImpDouble r = 0.0;
            for (ImpInt d = 0; d < k ; ++d)
                r += w[d]*h[d];
            RT.val[i] -= r;
        }
    }
}*/

void ImpProblem::update_R(ImpDouble *wt, ImpDouble *ht, bool add) {
    smat &R = data->R;
    smat &RT = data->RT;
    ImpLong l = data->l, m = data->m, n = data->n;
    if (add) {
#pragma omp parallel for schedule(guided)
        for (ImpLong i = 0; i < m; ++i) {
            ImpDouble w = wt[i];
            for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; ++j) {
                R.val[j] += w*ht[R.col_idx[j]];
            }
        }
#pragma omp parallel for schedule(guided)
        for (ImpLong j = 0; j < n; ++j) {
            ImpDouble h = ht[j];
            for(ImpLong i = RT.row_ptr[ji]; i < RT.row_ptr[j+1]; ++i) {
                RT.val[i] += wt[RT.col_idx[i]]*h;
            }
        }
    } else {
#pragma omp parallel for schedule(guided)
        for (ImpLong i = 0; i < m; ++i) {
            ImpDouble w = wt[i];
            for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; ++j) {
                R.val[j] -= w*ht[R.col_idx[j]];
            }
        }
#pragma omp parallel for schedule(guided)
        for (ImpLong j = 0; j < n; ++j) {
            ImpDouble h = ht[j];
            for(ImpLong i = RT.row_ptr[ji]; i < RT.row_ptr[j+1]; ++i) {
                RT.val[i] -= wt[RT.col_idx[i]]*h;
            }
        }
    }
}


void ImpProblem::update_coordinates() {
    ImpInt k = param->k;
    ImpLong m = data->m, n = data->n;
    ImpInt nr_th = param->nr_threads;
    vector<ImpDouble> hv_th(m*nr_th,0.0);
    vector<ImpDouble> wu_th(n*nr_th,0.0);
    for (ImpInt d = 0; d < k; d++) {
         ImpDouble *u = &WT[d*m];
         ImpDouble *v = &HT[d*n];
         update_R(u, v, true);
         for (ImpInt s = 0; s < 5; s++) {
            cache_w(u, wu_th.data());
#pragma omp parallel for schedule(guided)
            for (ImpLong j = 0; j < n; j++) {
                if (data->Q[j].size())
                    update_h(j, u, v);
            }
            cache_h(v, hv_th.data());
#pragma omp parallel for schedule(guided)
            for (ImpLong i = 0; i < m; i++) {
                if (data->P[i].size())
                    update_w(i, u, v);
            }
        }
        update_R(u, v, false); 
    }
}

void ImpProblem::cache_w(ImpDouble *wt, ImpDouble *wu_th) {
    ImpLong m = data->m, n = data->n;
    ImpInt k = param->k;
    ImpFloat sq = 0, sum = 0;
    ImpInt nr_th = param->nr_threads;
#pragma omp parallel for schedule(static)
    for (ImpInt num_th = 0; num_th < nr_th; num_th++)
        for (ImpLong i = 0; i < n; i++)
            wu_th[n*num_th+i] = 0;
#pragma omp parallel for schedule(static)
    for (ImpLong i = 0; i < n; i++) {
        wu[i] = 0;
    }
#pragma omp parallel for schedule(static) reduction(+:sq,sum)
    for (ImpInt j = 0; j < m; j++) {
        sq +=  wt[j]*wt[j];
        sum += wt[j];
    }
#pragma omp parallel for schedule(static)
    for (ImpInt di = 0; di < k; di++) {
        ImpInt num_th = omp_get_thread_num();
        ImpDouble uTWt = 0;
        for (ImpLong j = 0; j < m; j++) {
            uTWt += wt[j] * WT[di*m+j];
        }
        for (ImpLong i = 0; i < n; i++) {
            wu_th[n*num_th+i] += uTWt * HT[di*n+i];
        }
    }
#pragma omp parallel for schedule(static)
    for (ImpLong i = 0; i < n; i++)
        for(ImpInt num_th = 0; num_th < nr_th; num_th++)
            wu[i] += wu_th[num_th*n+i];
    w_sum = sum;
    w2_sum = sq;
}

void ImpProblem::cache_h(ImpDouble *ht, ImpDouble *hv_th) {
    ImpLong m = data->m, n = data->n;
    ImpInt k = param->k;
    ImpFloat sq = 0, sum = 0;
    ImpInt nr_th =  param->nr_threads;
#pragma omp parallel for schedule(static)
    for (ImpInt num_th = 0; num_th < nr_th; num_th++)
        for (ImpLong j = 0; j < m; j++)
            hv_th[m*num_th+j] = 0;
#pragma omp parallel for schedule(static)
    for (ImpLong j = 0; j < m; j++) {
        hv[j] = 0;
    }
#pragma omp parallel for schedule(static) reduction(+:sq,sum)
    for (ImpInt j = 0; j < n; j++) {
        sq +=  ht[j]*ht[j];
        sum += ht[j];
    }
#pragma omp parallel for schedule(static)
    for (ImpInt di = 0; di < k; di++) {
        ImpDouble uTWt = 0;
        ImpInt num_th = omp_get_thread_num();
        for (ImpLong i = 0; i < n; i++) {
            uTWt += ht[i] * HT[di*n+i];
        }
        for (ImpLong j = 0; j < m; j++) {
            hv_th[m*num_th+j] += uTWt * WT[di*m+j];
        }
    }
#pragma omp parallel for schedule(static)
    for (ImpLong j = 0; j < m; j++) {
        for(ImpInt num_th = 0; num_th < nr_th; num_th++) {
            hv[j] += hv_th[m*num_th+j];
        }
    }
    h_sum = sum;
    h2_sum = sq;
}

void ImpProblem::solve() {
    cout<<"Using "<<param->nr_threads<<" threads"<<endl;
    init_va_loss(6);
    vector<ImpInt> topks(6,0);
    topks[0] = 5; topks[1] = 10; topks[2] = 20;
    topks[3] = 40; topks[4] = 80; topks[5] = 100;
    print_header_info(topks);
    for (t = 0; t < param->nr_pass; t++) {
        update_coordinates();
        validate(topks);
        print_epoch_info();
        //if (t%3 == 2 && test_with_two_data) {
        //    validate_test(10);
        //    print_epoch_info_test();
        //}
    }
}

