#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() != ground_truth.size()) {
    cout << "Size of estimations and ground truth data don't match." << endl;
    return rmse;
  }

  if (estimations.size() == 0) {
    cout << "No estimations." << endl;
    return rmse;
  }

  for (size_t i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    VectorXd residual_squared = residual.array() * residual.array();
    rmse += residual_squared;
  }

  rmse = rmse / estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float sum = px * px + py * py;
  float sqrt_sum = sqrt(sum);
  float comb_sum = sum  * sqrt_sum;

  if (fabs(sum) < 1e-4) {
    cout << "Can't divide by zero while calculating Jacobian." << endl;
    return Hj;
  }

  Hj << px / sqrt_sum                , py / sqrt_sum                 , 0           , 0            ,
        -py / sum                    , px / sum                      , 0           , 0            ,
        py * (vx*py - vy*px)/comb_sum, px * (vy*px - vx*py)/comb_sum , px/sqrt_sum , py/sqrt_sum  ;

  return Hj;
}
