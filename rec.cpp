#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "load_rating.hpp"


using husky::lib::Aggregator;
using husky::lib::AggregatorFactory;


typedef Eigen::MatrixXd MatrixT;
typedef Eigen::VectorXd VectorT;
typedef std::map<int , std::pair<MatrixT,MatrixT>> MIPair;


int dim = 40;


//default feature dimension : 4
class FeatureT {
    public:
    using KeyT = int;

    FeatureT() = default;
    explicit FeatureT(const KeyT& k) : id_num(k) {
        feature = MatrixT::Random(dim , 1);
    }
    const KeyT& id() const { return id_num; }
//    const MatrixT& feature() const { return feature; }

    KeyT id_num;
    MatrixT feature;

};


template <bool is_sparse>
void train_recsys() {

    //int num_feature = 2;

    // get model config parameters
    double lambda_ = std::stod(husky::Context::get_param("lambda_"));
    int num_iter = std::stoi(husky::Context::get_param("n_iter"));


    auto& train_set = husky::ObjListStore::create_objlist<rating_obj>();
    auto& test_set = husky::ObjListStore::create_objlist<rating_obj>();

    //define feature list
    auto& user_list = husky::ObjListStore::create_objlist<FeatureT>();
    auto& item_list = husky::ObjListStore::create_objlist<FeatureT>();


    // load data
    load_rating(husky::Context::get_param("train"), train_set);
    load_rating(husky::Context::get_param("test"), test_set);

    globalize(train_set);
    globalize(test_set);


    auto add_to_list = [](MIPair& pairs , const std::pair<int , std::pair<MatrixT,MatrixT>>& p){
        auto iter = pairs.find(p.first);
        if (iter!=pairs.end()){
            iter->second.first += p.second.first;
            iter->second.second += p.second.second;
        }
        else{
            pairs.insert(p);
        }
    };

    Aggregator<MIPair> user_aggregator(MIPair(), [add_to_list](MIPair& a, const MIPair& b) {
        for (auto& c : b){
            add_to_list(a , c);}
        });
    user_aggregator.to_reset_each_iter();

    Aggregator<MIPair> item_aggregator(MIPair(), [add_to_list](MIPair& a, const MIPair& b) {
        for (auto& c : b){
            add_to_list(a , c);}
         });
    item_aggregator.to_reset_each_iter();



    //first . initialization
    husky::list_execute(train_set, [&item_list,&user_list](rating_obj& this_obj) {
        auto us_id = this_obj.user_id;
        auto it_id = this_obj.item_id;
        if (user_list.find(us_id) == nullptr){
            FeatureT u_obj(us_id);
            user_list.add_object(u_obj);
        }
        if (item_list.find(it_id) == nullptr){
            FeatureT i_obj(it_id);
            item_list.add_object(i_obj);
        }
    });


    // get the number of global records
    Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });
    num_samples_agg.update(train_set.get_size());

    Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
    num_test_agg.update(test_set.get_size());

    //get the number of unique user
    Aggregator<int> num_user_agg(0, [](int& a, const int& b) { a += b; });
    num_user_agg.update(user_list.get_size());

    //get the number of unique item
    Aggregator<int> num_item_agg(0, [](int& a, const int& b) { a += b; });
    num_item_agg.update(item_list.get_size());

    AggregatorFactory::sync();

    int num_samples = num_samples_agg.get_value();
    int num_test = num_test_agg.get_value();
    int num_users = num_user_agg.get_value();
    int num_items = num_item_agg.get_value();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "lambda =  " << lambda_;
        husky::LOG_I << "iteration number = " << num_iter;
        husky::LOG_I << "Training set size = " << num_samples;
        husky::LOG_I << "Test set size = " << num_test;
        husky::LOG_I << "user size = " << num_users;
        husky::LOG_I << "item size = " << num_items;
    }



    // Aggregators for regulator and error loss
    Aggregator<double> regulator_agg(0.0, [](double& a, const double& b) { a += b; });
    regulator_agg.to_reset_each_iter();
    Aggregator<double> error_agg(0.0, [](double& a, const double& b) { a += b; });
    error_agg.to_reset_each_iter();


    // Main loop
    auto start = std::chrono::steady_clock::now();
    for (int k = 0; k < num_iter; k++) {
        // update user_list
        husky::list_execute(train_set, [&item_list,&add_to_list,&user_aggregator,&lambda_](rating_obj& this_obj) {
            auto us_id = this_obj.user_id;
            auto it_id = this_obj.item_id;
            //husky::LOG_I << "user_id"<<this_obj.user_id;
            //husky::LOG_I << "item_id"<<this_obj.item_id;
            //husky::LOG_I << "rating"<<this_obj.rating;

            auto ptr = item_list.find(it_id);
            if (ptr != nullptr)
            {
                auto i_feature = ptr->feature;
                auto divider = i_feature * i_feature.transpose();
                auto dividend = i_feature * this_obj.rating;
                husky::ASSERT_MSG(!std::isnan(divider.sum()), "error 1");
                husky::ASSERT_MSG(!std::isnan(dividend.sum()), "error 2");
                // if (std::isnan(divider.sum())) {
                //     husky::ASSERT_MSG << "divider zero!!";
                // }
                // if (std::isnan(dividend.sum())) {
                //     husky::ASSERT_MSG << "dividend zero!!";
                // }
                user_aggregator.update(add_to_list, std::make_pair(us_id,std::make_pair(divider,dividend)));
            }
        });

        husky::lib::AggregatorFactory::sync();

        for (auto& i : user_aggregator.get_value()){
            auto ptr = user_list.find(i.first);
            if (ptr != nullptr){
                //husky::LOG_I << "i.second.first"<<i.second.first;
                auto u_divider = i.second.first + lambda_ * MatrixT::Identity(dim,dim);
                //husky::LOG_I << "u_divider"<<u_divider;
                auto u_dividend = i.second.second;
                //husky::LOG_I << "i.second.second"<<i.second.second;
                ptr->feature = u_divider.inverse() * u_dividend;
                //husky::LOG_I << "ptr->feature"<<ptr->feature;

            }
        }



        // update item_list
        husky::list_execute(train_set, [&add_to_list,&user_list,&item_aggregator,&lambda_](rating_obj& this_obj) {
            auto us_id = this_obj.user_id;
            auto it_id = this_obj.item_id;
            auto ptr = user_list.find(us_id);
            if (ptr != nullptr)
            {
                auto u_feature = ptr->feature;
                //husky::LOG_I << "u_feature"<<u_feature;
                auto divider = u_feature * u_feature.transpose();
                auto dividend = u_feature * this_obj.rating;
                //husky::LOG_I << "divider"<<divider;
                //husky::LOG_I << "dividend"<<dividend;
                item_aggregator.update(add_to_list, std::make_pair(it_id , std::make_pair(divider,dividend)));
            }
        });


        husky::lib::AggregatorFactory::sync();




        for (auto& i : item_aggregator.get_value()){
            //husky::LOG_I << i.first;
            //husky::LOG_I << i.second.first;
            //husky::LOG_I << i.second.second;
            auto ptr = item_list.find(i.first);
            if (ptr != nullptr){
                //husky::LOG_I << "i.second.first"<<i.second.first;
                auto i_divider = i.second.first + lambda_ * MatrixT::Identity(dim,dim);
                //husky::LOG_I << "i_divider"<<i_divider;
                auto i_dividend = i.second.second;
                ptr->feature = i_divider.inverse() * i_dividend;
                //husky::LOG_I << "ptr->feature"<<ptr->feature;

            }
        }

        //calculate trainset loss
        husky::list_execute(train_set, [&error_agg,&regulator_agg,&user_list,&item_list,&lambda_](rating_obj& this_obj) {
            auto us_id = this_obj.user_id;
            auto it_id = this_obj.item_id;
            auto ptr1 = user_list.find(us_id);
            auto ptr2 = item_list.find(it_id);
            if ((ptr1 != nullptr) && (ptr2 != nullptr))
            {
                auto u_feature = ptr1->feature;
                auto i_feature = ptr2->feature;
                //husky::LOG_I <<"u_feature"<<u_feature;
                //husky::LOG_I <<"i_feature"<<i_feature;
                //husky::LOG_I << u_feature.transpose() * i_feature;
                double predict = (u_feature.transpose() * i_feature)(0,0);
                double err = this_obj.rating-predict;
                //husky::LOG_I << "predict"<<predict;
                //husky::LOG_I << "rating"<<this_obj.rating;
                //husky::LOG_I << "error"<<pow(err,2);
                error_agg.update(pow(err,2));
                //double regu_part = (u_feature.array().pow(2).sum()+i_feature.array().pow(2).sum())*lambda_;
                //regulator_agg.update(regu_part);
            }
        });

        husky::lib::AggregatorFactory::sync();

        //double train_loss = error_agg.get_value()+regulator_agg.get_value();
        double train_loss = sqrt(error_agg.get_value()/num_samples);
        if (husky::Context::get_global_tid() == 0) {
            husky::LOG_I << k << " epoch :";
            husky::LOG_I << "loss on train set: " << train_loss;
        }

        //calculate RMSE on the testset
        husky::list_execute(test_set, [&error_agg,&user_list,&item_list](rating_obj& this_obj) {
            auto us_id = this_obj.user_id;
            auto it_id = this_obj.item_id;
            auto ptr1 = user_list.find(us_id);
            auto ptr2 = item_list.find(it_id);
            if ((ptr1 != nullptr) && (ptr2 != nullptr))
            {
                auto u_feature = ptr1->feature;
                auto i_feature = ptr2->feature;
                double predict = (u_feature.transpose() * i_feature)(0,0);
                double err = this_obj.rating-predict;
                error_agg.update(pow(err,2));
            }
        });
        husky::lib::AggregatorFactory::sync();
        double test_loss = sqrt(error_agg.get_value()/num_test);
        if (husky::Context::get_global_tid() == 0) {
            husky::LOG_I << "loss on test set: " << test_loss;
        }

    }

    // Show running time
    auto end = std::chrono::steady_clock::now();
    if (husky::Context::get_global_tid() == 0) {
        husky::LOG_I << "Time per iter: "
                     << std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count() / num_iter;
    }


}



int main(int argc, char** argv) {

    std::vector<std::string> args(
        {"hdfs_namenode", "hdfs_namenode_port", "train", "test", "n_iter", "lambda_"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(train_recsys<false>);
        return 0;
    }
    return 1;
}

