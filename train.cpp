#include <iostream>
#include <cstring>
#include <stdexcept>

#include "mf.h"

#include <fenv.h>


struct Option
{
    shared_ptr<Parameter> param;
    ImpInt verbose;
    string data_path, test_path, test_path_2;
    bool test_with_two_data;
};

string basename(string path)
{
    const char *ptr = strrchr(&*path.begin(), '/');
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

bool is_numerical(char *str)
{
    int c = 0;
    while(*str != '\0')
    {
        if(isdigit(*str))
            c++;
        str++;
    }
    return c > 0;
}

string train_help()
{
    return string(
    "usage: train [options] training_set_file test_set_file\n"
    "\n"
    "options:\n"
    "-l <lambda_2>: set regularization coefficient on r regularizer (default 0.1)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-p <path>: set path to test set\n"
    "-r <path>: set path to save result\n"
    "-m <path>: set path to save model\n"
    "-w <path>: set weights to the negatives (default 1)\n"
    "-a <path>: set labels to the negatives (default 0)\n"
    "-c <threads>: set number of cores\n"
    "-k <rank>: set number of rank\n"
    "-s set sample scheme (default -1)\n"
    "   -- 0 occur proabability\n"
    "   -- 1     1/(p+1)\n"
    "   -- 2     1/log(p)\n"
    "   -- 3     1 - p\n"
    "   -- -1     uniform\n"
    );
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option option;
    option.test_with_two_data = false;
    option.verbose = 1;
    option.param = make_shared<Parameter>();
    int i = 0;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-l") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l\
                                        regularization coefficient\
                                        after -l");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-l should be followed by a number");
            option.param->lambda = atof(argv[i]);
        }
        else if(args[i].compare("-k") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify rank after -k");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-k should be followed by a number");
            option.param->k = atoi(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param->nr_pass = atoi(argv[i]);
        }
        else if(args[i].compare("-w") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param->w = atof(argv[i]);
        }
        else if(args[i].compare("-a") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param->a = atof(argv[i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->scheme = atof(argv[i]);
        }
        else if(args[i].compare("-m") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -m");
            i++;

            option.param->model_path = string(args[i]);
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -r");
            i++;

            option.param->predict_path = string(args[i]);
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;

            option.test_path = string(args[i]);
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -v");
            i++;

            option.test_path_2 = string(args[i]);
            option.test_with_two_data = true;
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw invalid_argument("training data not specified");
    option.data_path = string(args[i++]);

    return option;
}

int main(int argc, char *argv[])
{
    try
    {
        Option option = parse_option(argc, argv);
        omp_set_num_threads(option.param->nr_threads);

        shared_ptr<ImpData> data = make_shared<ImpData>(option.data_path, option.param->a, 1);
        shared_ptr<ImpData> test_data = make_shared<ImpData>(option.test_path);

        data->read(option.param->scheme);

        if (!test_data->file_name.empty()) {
            test_data->read(option.param->scheme);
        }

        ImpProblem prob(data, test_data, option.param);
        prob.initialize();
        prob.solve();
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}
