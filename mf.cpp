#include "mf.h"
#include <cstring>
#define MIN_Z -10000;
//#include <immintrin.h>
int ALIGNByte = 32;


ImpDouble* impMalloc(ImpInt k)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, ALIGNByte, sizeof(ImpDouble)*k)) cout <<"Bad alloc"<<endl;
    return (ImpDouble*)ptr;
}



/*double inner(const ImpFloat *p, const ImpFloat *q, const int k)
{

    __m128d XMM = _mm_setzero_pd();

    for(ImpInt d = 0; d < k; d += 2)
        XMM = _mm_add_pd(XMM, _mm_mul_pd(
                  _mm_load_pd(p+d), _mm_load_pd(q+d)));
    XMM = _mm_hadd_pd(XMM, XMM);
    ImpFloat product;
    _mm_store_sd(&product, XMM);
    return product;

    double r = 0.0;
    for (int i = 0; i < k; i++)
        r += p[i]*q[i];
    return r;

    __m256d XMM = _mm256_setzero_pd();
    for(ImpInt d = 0; d < k; d += 4) {
        XMM = _mm256_add_pd(XMM, _mm256_mul_pd(
                  _mm256_load_pd(p+d), _mm256_load_pd(q+d)));
    }
    XMM = _mm256_add_pd(XMM, _mm256_permute2f128_pd(XMM, XMM, 1));
    XMM = _mm256_hadd_pd(XMM, XMM);
    ImpDouble product;
    _mm_store_sd(&product, _mm256_castpd256_pd128(XMM));
    return product;
}*/

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

    write(WT, m, 'p');
    write(HT, n, 'q');

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

    read(WT, data->m_real);
    read(HT, data->n_real);

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

        m = max(p_idx+1, m);
        n = max(q_idx+1, n);
    }

    m_real = m;
    n_real = n;
    /*ImpInt mul = ALIGNByte/8;
    if ( m%mul != 0) m = ((m/mul)+1)*mul;
    if ( n%mul != 0) n = ((n/mul)+1)*mul;*/
    fs.close();
    fs.clear();
    fs.open(file_name);
    l += l*SAMPLE_SIZE;
    R.row_ptr.resize(m+1);
    RT.row_ptr.resize(n+1);
    R.col_idx.resize(l);
    RT.col_idx.resize(l);
    R.val.resize(l);
    RT.val.resize(l);
    vector<ImpLong> perm;
    perm.resize(l);
    ImpLong idx = 0;
    for( ImpInt repeat = 0; repeat < SAMPLE_SIZE; repeat++) {
        for(ImpInt i = 0; i < l/(SAMPLE_SIZE+1); i++ ) {
            ImpLong p_idx, q_idx;
            p_idx = rand()%m;
            q_idx = rand()%n;

            ImpFloat val;
            val = a;
        
            R.row_ptr[p_idx+1]++;
            RT.row_ptr[q_idx+1]++;
            R.col_idx[idx]  = p_idx;
            RT.col_idx[idx] = q_idx;
            RT.val[idx] = val;
            perm[idx] = idx;
            idx++;
        }
    }
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

    WT = impMalloc(m*k);
    HT = impMalloc(n*k);


    default_random_engine engine(0);
    uniform_real_distribution<ImpFloat> distribution(0, 1.0/sqrt(k));
    for (ImpInt d = 0; d < k; d++)
    {
        for (ImpLong j = 0; j < m; j++) {
            WT[d*m+j] = distribution(engine); 
        }
        for (ImpLong j = 0; j < n; j++) {
            if (data->RT.row_ptr[j+1]!=data->RT.row_ptr[j]) {
                HT[d*n+j] = 0.0;
            } else {
                HT[d*n+j] = distribution(engine);
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

void ImpProblem::update(const smat &R, ImpLong i, ImpFloat *u, ImpFloat *v) {
    ImpFloat lambda = param->lambda;
    //ImpInt k = param->k;
    ImpDouble h = lambda*(R.row_ptr[i+1] - R.row_ptr[i]), g = 0;
    for (ImpLong idx = R.row_ptr[i]; idx < R.row_ptr[i+1]; idx++) {
        ImpDouble r = R.val[idx];
        ImpLong j = R.col_idx[idx];
        ImpDouble v_val = v[j];
        g += r*v_val;
        h += v_val*v_val;
    }

    ImpDouble new_u_val = g/h;
    //ut[i*k] = new_u_val;
    u[i] = new_u_val;
}

ImpDouble ImpProblem::cal_loss(ImpLong &l, smat &R) {
    ImpInt k = param->k;
    ImpDouble loss = 0, a = param->a;
    ImpLong m = data->m, n = data->n;
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong i = 0; i < m; i++) {
        ImpDouble *w = WT+i;
        for(ImpLong idx = R.row_ptr[i]; idx < R.row_ptr[i+1]; idx++) {
            if (R.col_idx[idx] > data->n)
                continue;
            ImpDouble *h = HT + R.col_idx[idx] ;
            ImpDouble r = 0;
            for (ImpInt d = 0; d < k; d++)
                r += w[d*m] * h[d*n];
            loss += R.val[idx]*R.val[idx];
            loss -= param->w*(a-r)*(a-r);
        }
    }
#pragma omp parallel for schedule(static) reduction(+:loss)
    for (ImpLong i = 0; i < m; i++) {
        ImpDouble *w = WT+i;
        for (ImpLong j = 0; j < n; j++) {
            ImpDouble *h = HT+j;
            ImpDouble r = 0.0;
            for (ImpInt d = 0; d < k; d++)
                r += w[d*m] * h[d*n];
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
    ImpLong n = data->n;
    ImpLong m = min(data->m,test_data->m);
    ImpInt nr_th = param->nr_threads;
    const smat &testR = test_data->R;
    const ImpFloat* Wp = WT;
    vector<ImpLong> hit_counts(nr_th*topks.size(),0);
    ImpLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < m; i++) {
        vector<ImpFloat> Z(n, 0);
        if (testR.row_ptr[i+1]==testR.row_ptr[i]) {
            continue;
        }
        const ImpFloat *w = Wp+i;
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
    ImpLong n = data->n;
    ImpLong m = min(data->m,test_data->m);
    ImpInt nr_th = param->nr_threads;
    const smat &testR = test_data->R;
    const ImpFloat* Wp = WT;
    vector<double> ndcgs(nr_th*topks.size(),0);
    ImpLong valid_samples = 0;
#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < m; i++) {
        vector<ImpFloat> Z(n, 0);
        if (testR.row_ptr[i+1]==testR.row_ptr[i]) {
            continue;
        }
        const ImpFloat *w = Wp+i;
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
    ImpLong m = data->m;
    ImpFloat *Hp = HT;
    for(ImpInt d = 0; d < k; d++) {
        for (ImpLong j = 0; j < n; j++) {
            Z[j] += w[d*m]*Hp[d*n+j];
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
            if (is_hit(data->R, i, argmax)) {
                Z[argmax] = MIN_Z;
                continue;
             }
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
        ImpDouble* w = WT+i;
        ImpDouble inner = 0.0;
        for (ImpInt d = 0; d < k ; d++)
            inner += w[d*m] * w[d*m];
        reg += nnz*lambda*inner;
    }

    for (ImpLong j = 0; j < n; j++) {
        ImpLong nnz = RT.row_ptr[j+1] - RT.row_ptr[j];
        ImpDouble* h = HT+j*k;
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
        for(ImpLong j = R.row_ptr[i]; j < R.row_ptr[i+1]; j++) {
            ImpDouble *w = W+i*k;
            ImpDouble *h = H+R.col_idx[j]*k;
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
            ImpDouble *w = W+R.col_idx[i]*k;
            ImpDouble *h = H+j*k;
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
    //double cache_time = 0.0;
    double update_time = 0.0;
    //double sync_time =0.0;
    double r_time = 0.0;
    double time;
    for (ImpInt d = 0; d < k; d++) {
         ImpDouble *u = &WT[d*m];
         ImpDouble *v = &HT[d*n];
         time = omp_get_wtime();
         update_R(u, v, true);
         r_time += omp_get_wtime() - time;
         time = omp_get_wtime();
         for (ImpInt s = 0; s < 5; s++) {
#pragma omp parallel for schedule(guided)
            for (ImpLong j = 0; j < n; j++) {
                if (data->RT.row_ptr[j+1]!=data->RT.row_ptr[j])
                    update(data->RT, j, v, u);
            }
#pragma omp parallel for schedule(guided)
            for (ImpLong i = 0; i < m; i++) {
                if (data->R.row_ptr[i+1]!=data->R.row_ptr[i])
                    update(data->R, i, u, v);
            }
        }
        update_time += omp_get_wtime() -time;
        time = omp_get_wtime();
        update_R(u, v, false);
        r_time += omp_get_wtime() - time;
    }
    /*//cout<< "cache time : "<< cache_time << endl;
    cout<< "update time: "<< update_time<< endl;
    //cout<< "matrix vector p1: "<< mv1_time<< endl;
    //cout<< "matrix vector p2: "<< mv2_time<< endl;
    //cout<< "sync time  : "<< sync_time<<endl;
    cout<< "r time     : "<< r_time <<endl;*/
}

/*void ImpProblem::cache(ImpDouble* WT_, ImpDouble* H_, vector<ImpFloat> &gamma, ImpFloat *ut, ImpLong m, ImpLong n) {
    ImpInt k = param->k;
    ImpFloat sq_ = 0, sum_ = 0;
    void *ptr = NULL;
    if (posix_memalign(&ptr, ALIGNByte, sizeof(ImpDouble)*k)) cout <<"Bad alloc at cache"<<endl;
    ImpDouble* alpha = (ImpDouble*)ptr;

#pragma omp parallel for schedule(static)
    for (ImpLong j = 0; j < n; j++) {
        gamma[j] = 0;
    }

    //sum_ = cblas_ddot(n, ut, 1, &y, 0);
    //sq_ = cblas_dnrm2(n, ut, 1);
    //sq_ = sq_*sq_;

#pragma omp parallel for schedule(static) reduction(+:sq_,sum_)
    for (ImpInt i = 0; i < m; i++) {
        sq_ +=  ut[i]*ut[i];
        sum_ += ut[i];
    }
    //cblas_dgemv(CblasRowMajor, CblasNoTrans, k, m, 1, WT_.data(), k, ut, 1, 0, alpha.data(), 1);
    double time = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (ImpInt d = 0; d < k; d++) {
        alpha[d] = inner(WT_+d*m, ut, m);
    }
    mv1_time += omp_get_wtime() -time;
    time = omp_get_wtime();
    //cblas_dgemv(CblasRowMajor, CblasNoTrans, k, n, 1, H_.data(), n, alpha.data(), 1, 0, gamma.data(), 1);
#pragma omp parallel for schedule(static)
    for (ImpLong j = 0; j < n; j++) {
        gamma[j] = inner(H_+j*k,alpha, k);
    }
    mv2_time += omp_get_wtime() -time;
    sum = sum_;
    sq = sq_;
    free(ptr);
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
        validate(topks);
        print_epoch_info();
    }
    cout<<"Training Time: "<< omp_get_wtime() - time <<endl;
    save();
}

