#ifndef __EKF_H__
#define __EKF_H__
#define ARM_MATH_CM7
#include <arm_math.h>
#include <stdlib.h>
#include <stdint.h>

typedef void (*state_func_t)(arm_matrix_instance_f32* x, arm_matrix_instance_f32* u, float dt, arm_matrix_instance_f32* x_pred);
typedef void (*meas_func_t)( arm_matrix_instance_f32* x, arm_matrix_instance_f32* z_pred);
typedef void (*jacobian_func_t)(arm_matrix_instance_f32* x, arm_matrix_instance_f32* u, float dt, arm_matrix_instance_f32* jacobian);

typedef struct {
    uint16_t dim_x;
    uint16_t dim_z;

    arm_matrix_instance_f32 x;      // state vector (dim_x x 1)
    arm_matrix_instance_f32 P;      // state covariance (dim_x x dim_x)
    arm_matrix_instance_f32 Q;      // process noise cov (dim_x x dim_x)
    arm_matrix_instance_f32 R;      // measurement noise cov (dim_z x dim_z)

    /* Function pointers for nonlinear models */
    state_func_t f;
    meas_func_t h;
    jacobian_func_t F_jacobian;
    jacobian_func_t H_jacobian;

    /* Workspace matrices */
    arm_matrix_instance_f32 Fx;     // Jacobian of f (dim_x x dim_x)
    arm_matrix_instance_f32 Hx;     // Jacobian of h (dim_z x dim_x)
    arm_matrix_instance_f32 K;      // Kalman gain (dim_x x dim_z)

    /* Buffers for matrix data storage (allocated by user) */
    float* x_data;
    float* P_data;
    float* Q_data;
    float* R_data;
    float* Fx_data;
    float* Hx_data;
    float* K_data;
} EKF_HandleTypeDef;

void EKF_Init(EKF_HandleTypeDef* ekf,
              uint16_t dim_x,
              uint16_t dim_z,
              float* x_buffer,
              float* P_buffer,
              float* Q_buffer,
              float* R_buffer,
              float* Fx_buffer,
              float* Hx_buffer,
              float* K_buffer,
              state_func_t f,
              meas_func_t h,
              jacobian_func_t F_jacobian,
              jacobian_func_t H_jacobian);

void EKF_Predict(EKF_HandleTypeDef* ekf,  arm_matrix_instance_f32* u, float dt);
void EKF_Update(EKF_HandleTypeDef* ekf,  arm_matrix_instance_f32* z);
void EKF_Step(EKF_HandleTypeDef* ekf,  arm_matrix_instance_f32* u,  arm_matrix_instance_f32* z, float dt);

#endif // __EKF_H__
