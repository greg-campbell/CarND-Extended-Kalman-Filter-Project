#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = (F_ * P_ * Ft) + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd zp = H_ * x_;
  VectorXd y = z - zp;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = (P_ * H_.transpose()) * S.inverse();

  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho = sqrt(px*px + py*py);
  float phi = atan2(py, px);

  if (fabs(rho) < 1e-6) {
    cout << "Error in EKF - rho equals zero." << endl;
    return;
  }

  VectorXd zp = VectorXd(3);
  zp << rho, phi, (px*vx + py*vy)/rho;

  VectorXd y = z - zp;

  while (y(1) > M_PI) {
      y(1) -= 2*M_PI;
  }
  while (y(1) < -M_PI) {
      y(1) += 2*M_PI;
  }

  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = (P_ * H_.transpose()) * S.inverse();

  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_; 
}
