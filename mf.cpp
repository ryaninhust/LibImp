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
            ptr++;
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
            ImpFloat *ptr1 = ptr + i*param->k ;
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
                    f << ptr1[d] << " ";
            }
            f << endl;
        }

    };

    write(W.data(), m, 'w');
    write(H.data(), n, 'h');

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
    while (getline(fs, line)) {
        istringstream iss(line);
        l++;
        ImpLong p_idx, q_idx;
        iss >> p_idx;
        iss >> q_idx;
        p_idx--;
        q_idx--;
        m = max(p_idx+1, m);
        n = max(q_idx+1, n);
    }
    fs.close();
    fs.clear();
    fs.open(file_name);
    R.row_ptr.resize(m+1);
    RT.row_ptr.resize(n+1);
    R.col_idx.resize(l);
    RT.col_idx.resize(l);
    R.val.resize(l);
    RT.val.resize(l);
    vector<ImpLong> perm;
    perm.resize(l);
    ImpLong idx = 0;
    while (getline(fs, line)) {
        istringstream iss(line);
        
        ImpLong p_idx, q_idx;
        iss >> p_idx;
        iss >> q_idx;
        p_idx--;
        q_idx--;

        ImpFloat val;
        iss >> val;

        R.row_ptr[p_idx+1]++;
        RT.row_ptr[q_idx+1]++;
        R.col_idx[idx]  = p_idx;
        RT.col_idx[idx] = q_idx;
        RT.val[idx] = val;
        perm[idx] = idx;
        idx++;
    }
    sort(perm.begin(), perm.end(),Compare(R.col_idx.data(), RT.col_idx.data()));
    
    for(idx = 0; idx < l; idx++ ) {
       R.col_idx[idx] = RT.col_idx[perm[idx]];
       R.val[idx] = RT.val[perm[idx]];
    }
    for(ImpLong i = 1; i < m+1; i++) {
        R.row_ptr[i] += R.row_ptr[i-1];
    }
    for(ImpLong j = 1; j < n+1; j++) {
        RT.row_ptr[j] += RT.row_ptr[j-1];
    }
    for(ImpLong i = 0; i < m; i++) {
        for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; j++) {
            ImpLong c = R.col_idx[j];
            RT.col_idx[RT.row_ptr[c]] = i;
            RT.val[RT.row_ptr[c]] = R.val[j];
            RT.row_ptr[c]++;
        }
    }
    for(ImpLong j = n; j > 0; j--)
        RT.row_ptr[j] = RT.row_ptr[j-1];
    RT.row_ptr[0] = 0;
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

    gamma_w.resize(n);
    gamma_h.resize(m);


    default_random_engine engine(0);
    uniform_real_distribution<ImpFloat> distribution(0, 1.0/sqrt(k));
#pragma omp parallel for schedule(static)
    for (ImpInt d = 0; d < k; d++)
    {
        for (ImpLong j = 0; j < m; j++) {
            WT[d*m+j] = distribution(engine); 
            //WT[d*m+j] = 1/sqrt(k); 
            W[j*k+d] = WT[d*m+j];
        }
        for (ImpLong j = 0; j < n; j++) {
            if (data->RT.row_ptr[j+1]!=data->RT.row_ptr[j]) {
                HT[d*n+j] = 0;
                H[j*k+d] = HT[d*n+j];
            } else {
                HT[d*n+j] = distribution(engine);
                //HT[d*n+j] = 0;
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

void ImpProblem::update(const smat &R, ImpLong i, vector<ImpFloat> &gamma, ImpFloat *u, ImpFloat *v) {
    ImpFloat lambda = param->lambda, a = param->a, w = param->w;
    ImpInt k = param->k;
    ImpDouble u_val = u[i];
    ImpDouble h = lambda*(R.row_ptr[i+1] - R.row_ptr[i]), g = 0;
    for (ImpLong idx = R.row_ptr[i]; idx < R.row_ptr[i+1]; idx++) {
        ImpDouble r = R.val[idx];
        ImpLong j = R.col_idx[idx];
        ImpDouble v_val = v[j];
        g += ((1-w)*r+w*(1-a))*v_val;
        h += (1-w)*v_val*v_val;
    }
    h += w*sq;
    g += w*(a*sum-gamma[i]+u_val*sq);
    
    //if(i<10)
    //    printf("U : %f, D : %f, H : %f\n", g, h, g/h);

    ImpDouble new_u_val = g/h;
    //ut[i*k] = new_u_val;
    u[i] = new_u_val;
}

/*void ImpProblem::update_h(ImpLong j,ImpInt d, ImpDouble *wt, ImpDouble *ht) {
    ImpFloat lambda = param->lambda, a = param->a, w = param->w;
    ImpInt k = param->k;
    const smat &RT = data->RT;
    ImpDouble h_val = ht[j];
    ImpDouble h = lambda*(RT.row_ptr[j+1] - RT.row_ptr[j]), g = 0;
    for (ImpLong idx = RT.row_ptr[j]; idx < RT.row_ptr[j+1]; idx++) {
        ImpDouble r = RT.val[idx];
        ImpLong i = RT.col_idx[idx];
        ImpDouble w_val = wt[i];
        g += ((1-w)*r+w*(1-a))*w_val;
        h += (1-w)*w_val*w_val;
    }
    h += w*w2_sum;
    g += w*(a*w_sum-wu[j]+h_val*w2_sum);

    ImpDouble new_h_val = g/h;
    H[j*k+d] = new_h_val;
    ht[j] = new_h_val;
}*/


ImpDouble ImpProblem::cal_loss(ImpLong &l, smat &R) {
    ImpInt k = param->k;
    ImpDouble loss = 0, a = param->a;
    ImpLong m = data->m, n = data->n;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong i = 0; i < m; i++) {
        ImpDouble *w = W.data()+i*k;
        for(ImpLong idx = R.row_ptr[i]; idx < R.row_ptr[i+1]; idx++) {
            if (R.col_idx[idx] > data->n)
                continue;
            ImpDouble *h = H.data()+ R.col_idx[idx]*k ;
            ImpDouble r = 0;
            for (ImpInt d = 0; d < k; d++)
                r += w[d] * h[d];
            loss += R.val[idx]*R.val[idx];
            loss -= param->w*(a-r)*(a-r);
        }
    }
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong i = 0; i < m; i++) {
        ImpDouble *w = W.data()+i*k;
        for (ImpLong j = 0; j < n; j++) {
            ImpDouble *h = H.data()+j*k;
            ImpDouble r = 0.0;
            for (ImpInt d = 0; d < k; d++)
                r += w[d] * h[d];
            loss += param->w*(a-r)*(a-r);
        }
    }
    return loss;
}

ImpDouble ImpProblem::cal_tr_loss(ImpLong &l, smat &R) {
    ImpDouble loss = 0;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong idx = 0; idx < l; idx++)
        loss += R.val[idx]*R.val[idx];
    return loss;
}

void ImpProblem::validate(const vector<ImpInt> &topks) {
    ImpLong n = data->n, m = data->m;
    ImpInt nr_th = param->nr_threads, k = param->k;
    const smat &testR = test_data->R;
    const ImpFloat* Wp = W.data();
    vector<ImpLong> hit_counts(nr_th*topks.size(),0);
    ImpLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < m; i++) {
        vector<ImpFloat> Z(n, 0);
        if (testR.row_ptr[i+1]==testR.row_ptr[i]) {
            continue;
        }
        const ImpFloat *w = Wp+i*k;
        predict_candidates(w, Z);
        precision_k(Z, i, topks, hit_counts);
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
    ImpInt nr_th = param->nr_threads, k = param->k;
    const smat &testR = test_data->R;
    const ImpFloat* Wp = W.data();
    vector<double> ndcgs(nr_th*topks.size(),0);
    ImpLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < m; i++) {
        vector<ImpFloat> Z(n, 0);
        if (testR.row_ptr[i+1]==testR.row_ptr[i]) {
            continue;
        }
        const ImpFloat *w = Wp+i*k;
        predict_candidates(w, Z);
        ndcg_k(Z, i, topks, ndcgs);
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
    ImpLong n = data->n;
    ImpFloat *Hp = HT.data();
    for(ImpInt d = 0; d < k; d++) {
        for (ImpLong j = 0; j < n; j++) {
            Z[j] += w[d]*Hp[d*n+j];
        }
    }
}

bool ImpProblem::is_hit(const smat &R, ImpLong i, ImpLong argmax) {
    for (ImpLong idx = R.row_ptr[i]; idx < R.row_ptr[i+1]; idx++) {
        ImpLong j = R.col_idx[idx];
        if (j == argmax)
            return true;
    }
    return false;
}

ImpDouble ImpProblem::ndcg_k(vector<ImpFloat> &Z, ImpLong i, const vector<ImpInt> &topks, vector<double> &ndcgs) {
    ImpInt state = 0;
    ImpInt valid_count = 0;
    vector<double> dcg(topks.size(),0.0);
    vector<double> idcg(topks.size(),0.0);
    ImpInt num_th = omp_get_thread_num();
    while(state < int(topks.size()) ) {
        while(valid_count < topks[state]) {
            ImpLong argmax = distance(Z.begin(), max_element(Z.begin(), Z.end()));
            if (is_hit(data->R, i, argmax)) {
                Z[argmax] = MIN_Z;
                continue;
            }
            if (is_hit(test_data->R, i, argmax))
                dcg[state] += 1.0/log2(valid_count+2);
            if (test_data->R.row_ptr[i+1] - test_data->R.row_ptr[i] > valid_count)
                idcg[state] += 1.0/log2(valid_count+2);
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

ImpLong ImpProblem::precision_k(vector<ImpFloat> &Z, ImpLong i, const vector<ImpInt> &topks, vector<ImpLong> &hit_counts) {
    ImpInt state = 0;
    ImpInt valid_count = 0;
    vector<ImpInt> hit_count(topks.size(), 0);
    ImpInt num_th = omp_get_thread_num();
    while(state < int(topks.size()) ) {
        while(valid_count < topks[state]) {
            ImpLong argmax = distance(Z.begin(), max_element(Z.begin(), Z.end()));
            if (is_hit(test_data->R, i, argmax)) {
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
    smat &R = data->R;
    smat &RT = data->RT;

    for (ImpLong i = 0; i < m; i++) {
        ImpLong nnz = R.row_ptr[i+1] - R.row_ptr[i];
        ImpDouble* w = W.data()+i*k;
        ImpDouble inner = 0.0;
        for (ImpInt d = 0; d < k ; d++)
            inner += w[d] * w[d];
        reg += nnz*lambda*inner;
    }

    for (ImpLong j = 0; j < n; j++) {
        ImpLong nnz = RT.row_ptr[j+1] - RT.row_ptr[j];
        ImpDouble* h = H.data()+j*k;
        ImpDouble inner = 0.0;
        for (ImpInt d = 0; d < k ; d++)
            inner += h[d]*h[d];
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
        for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; j++) {
            ImpDouble *w = W.data()+i*k;
            ImpDouble *h = H.data()+R.col_idx[j]*k;
            ImpDouble r = 0.0;
            for (ImpInt d = 0; d < k ; d++)
                r += w[d]*h[d];
            R.val[j] -= r;
        }
    }
#pragma omp parallel for schedule(guided)
    for (ImpLong j = 0; j < n; j++) {
        for(ImpLong i = RT.row_ptr[j]; i < RT.row_ptr[j+1]; i++) {
            Node* node = &RT[i];
            ImpDouble *w = W.data()+R.col_idx[i]*k;
            ImpDouble *h = H.data()+j*k;
            ImpDouble r = 0.0;
            for (ImpInt d = 0; d < k ; d++)
                r += w[d]*h[d];
            RT.val[i] -= r;
        }
    }
}*/

void ImpProblem::update_R(ImpDouble *wt, ImpDouble *ht, bool add) {
    smat &R = data->R;
    smat &RT = data->RT;
    ImpLong m = data->m, n = data->n;
    if (add) {
#pragma omp parallel for schedule(guided)
        for (ImpLong i = 0; i < m; i++) {
            ImpDouble w = wt[i];
            for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; j++) {
                R.val[j] += w*ht[R.col_idx[j]];
            }
        }
#pragma omp parallel for schedule(guided)
        for (ImpLong j = 0; j < n; j++) {
            ImpDouble h = ht[j];
            for(ImpLong i = RT.row_ptr[j]; i < RT.row_ptr[j+1]; i++) {
                RT.val[i] += wt[RT.col_idx[i]]*h;
            }
        }
    } else {
#pragma omp parallel for schedule(guided)
        for (ImpLong i = 0; i < m; i++) {
            ImpDouble w = wt[i];
            for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; j++) {
                R.val[j] -= w*ht[R.col_idx[j]];
            }
        }
#pragma omp parallel for schedule(guided)
        for (ImpLong j = 0; j < n; j++) {
            ImpDouble h = ht[j];
            for(ImpLong i = RT.row_ptr[j]; i < RT.row_ptr[j+1]; i++) {
                RT.val[i] -= wt[RT.col_idx[i]]*h;
            }
        }
    }
}


void ImpProblem::update_coordinates() {
    ImpInt k = param->k;
    ImpLong m = data->m, n = data->n;
    double cache_time = 0.0;
    double update_time = 0.0;
    double cu_time = 0.0;

    double r_time = 0.0;
    double time, time2;
    for (ImpInt d = 0; d < k; d++) {
         ImpDouble *u = &WT[d*m];
         ImpDouble *v = &HT[d*n];
         ImpDouble *ut = &W[d];
         ImpDouble *vt = &H[d];
         time = omp_get_wtime();
         update_R(u, v, true);
         r_time += omp_get_wtime() - time;
         time2 = omp_get_wtime();
         for (ImpInt s = 0; s < 1; s++) {
            time = omp_get_wtime();
            cache(WT, H, gamma_w, u, m, n);
            cache_time += omp_get_wtime() - time;
            //cout<<"H"<<d<<endl;
            time = omp_get_wtime();
#pragma omp parallel for schedule(guided)
            for (ImpLong j = 0; j < n; j++) {
                if (data->RT.row_ptr[j+1]!=data->RT.row_ptr[j])
                    update(data->RT, j, gamma_w, v, u);
            }
            update_time += omp_get_wtime() - time;
#pragma omp parallel for schedule(static)
            for (ImpLong j = 0; j < n; j++)
                vt[j*k] = v[j];
            time = omp_get_wtime();
            cache(HT, W, gamma_h, v, n, m);
            cache_time += omp_get_wtime() - time;
            //cout<<"W"<<d<<endl;
            time = omp_get_wtime();
#pragma omp parallel for schedule(guided)
            for (ImpLong i = 0; i < m; i++) {
                if (data->R.row_ptr[i+1]!=data->R.row_ptr[i])
                    update(data->R, i, gamma_h, u, v);
            }
            update_time += omp_get_wtime() - time;
#pragma omp parallel for schedule(static)
            for (ImpLong i = 0; i < m; i++)
                ut[i*k] = u[i];
        }
        cu_time += omp_get_wtime() -time2;
        time = omp_get_wtime();
        update_R(u, v, false);
        r_time += omp_get_wtime() - time;
    }
    cout<< "cache time : "<< cache_time << endl;
    cout<< "update time: "<< update_time<< endl;
    cout<< "ca+up time : "<< cu_time<< endl;
    cout<< "r time     : "<< r_time <<endl;
}

void ImpProblem::cache(vector<ImpFloat> &WT, vector<ImpFloat> &H, vector<ImpFloat> &gamma, ImpFloat *ut, ImpLong m, ImpLong n) {
    ImpInt k = param->k;
    ImpFloat sq_ = 0, sum_ = 0;
    vector<ImpDouble> alpha(k,0);
#pragma omp parallel for schedule(static)
    for (ImpLong j = 0; j < n; j++) {
        gamma[j] = 0;
    }
#pragma omp parallel for schedule(static) reduction(+:sq_,sum_)
    for (ImpInt i = 0; i < m; i++) {
        sq_ +=  ut[i]*ut[i];
        sum_ += ut[i];
    }
#pragma omp parallel for schedule(static)
    for (ImpInt d = 0; d < k; d++) {
        for (ImpLong i = 0; i < m; i++) {
            alpha[d] += ut[i] * WT[d*m+i];
        }
    }
#pragma omp parallel for schedule(static)
    for (ImpLong j = 0; j < n; j++) {
        for (ImpInt d = 0; d < k; d++) {
            gamma[j] += alpha[d] * H[j*k+d];
        }
    }
    sum = sum_;
    sq = sq_;
}
/*
void ImpProblem::cache_h(ImpDouble *vt) {
    ImpLong m = data->m, n = data->n;
    ImpInt k = param->k;
    ImpFloat sq = 0, sum = 0;
    vector<ImpDouble> vTHt(k,0); 
#pragma omp parallel for schedule(static)
    for (ImpLong i = 0; i < m; i++) {
        hv[i] = 0;
    }
#pragma omp parallel for schedule(static) reduction(+:sq,sum)
    for (ImpInt j = 0; j < n; j++) {
        sq +=  vt[j]*vt[j];
        sum += vt[j];
    }
#pragma omp parallel for schedule(static)
    for (ImpInt d = 0; d < k; d++) {
        for (ImpLong j = 0; j < n; j++) {
            vTHt[d] += vt[j]*HT[d*n+j];
        }
    }
#pragma omp parallel for schedule(static)
    for (ImpLong i = 0; i < m; i++) {
        for (ImpInt d = 0; d < k; d++) {
            hv[i] += vTHt[d]*W[i*k+d];
        }
    }

    h_sum = sum;
    h2_sum = sq;
}*/

void ImpProblem::solve() {
    cout<<"Using "<<param->nr_threads<<" threads"<<endl;
    init_va_loss(6);
    vector<ImpInt> topks(6,0);
    topks[0] = 5; topks[1] = 10; topks[2] = 20;
    topks[3] = 40; topks[4] = 80; topks[5] = 100;
    print_header_info(topks);
    double time = omp_get_wtime();
    for (t = 0; t < param->nr_pass; t++) {
        update_coordinates();
        //validate(topks);
        //print_epoch_info();
    }
    cout<<"Training Time: "<< omp_get_wtime() - time <<endl;
    //save();
}

