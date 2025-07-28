/*
 * Extended-KF.c
 *
 *  Created on: Jul 21, 2025
 *      Author: yassine jelassi
 */
#include "main.h"
#include "Extended-KF.h"
#include <string.h>
float X_data_updated_probe;
float X_data_predicted_probe;
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
              jacobian_func_t H_jacobian) {
    ekf->dim_x = dim_x;
    ekf->dim_z = dim_z;

    ekf->x_data = x_buffer;
    ekf->P_data = P_buffer;
    ekf->Q_data = Q_buffer;
    ekf->R_data = R_buffer;
    ekf->Fx_data = Fx_buffer;
    ekf->Hx_data = Hx_buffer;
    ekf->K_data = K_buffer;

    arm_mat_init_f32(&ekf->x, dim_x, 1, ekf->x_data);
    arm_mat_init_f32(&ekf->P, dim_x, dim_x, ekf->P_data);
    arm_mat_init_f32(&ekf->Q, dim_x, dim_x, ekf->Q_data);
    arm_mat_init_f32(&ekf->R, dim_z, dim_z, ekf->R_data);
    arm_mat_init_f32(&ekf->Fx, dim_x, dim_x, ekf->Fx_data);
    arm_mat_init_f32(&ekf->Hx, dim_z, dim_x, ekf->Hx_data);
    arm_mat_init_f32(&ekf->K, dim_x, dim_z, ekf->K_data);

    // Initialize function pointers
    ekf->f = f;
    ekf->h = h;
    ekf->F_jacobian = F_jacobian;
    ekf->H_jacobian = H_jacobian;
}

void EKF_Predict(EKF_HandleTypeDef* ekf,  arm_matrix_instance_f32* u, float dt) {
    float x_pred_data[ekf->dim_x];
    arm_matrix_instance_f32 x_pred;
    arm_mat_init_f32(&x_pred, ekf->dim_x, 1, x_pred_data);

    ekf->f(&ekf->x, u, dt, &x_pred);

    ekf->F_jacobian(&ekf->x, u, dt, &ekf->Fx);

    // x = f(x,u)
    /*
     * TODO replace the memcpy with cmsis
     */
    memcpy(ekf->x.pData, x_pred.pData, sizeof(float)*ekf->dim_x);

    // P = F P F^T + Q
    float temp_data[ekf->dim_x * ekf->dim_x];
    arm_matrix_instance_f32 temp, FxT;
    arm_mat_init_f32(&temp, ekf->dim_x, ekf->dim_x, temp_data);
    arm_mat_init_f32(&FxT, ekf->dim_x, ekf->dim_x, temp_data); // reuse buffer for transpose

    arm_mat_mult_f32(&ekf->Fx, &ekf->P, &temp);
    arm_mat_trans_f32(&ekf->Fx, &FxT);
    arm_mat_mult_f32(&temp, &FxT, &ekf->P);
    arm_mat_add_f32(&ekf->P, &ekf->Q, &ekf->P);
    X_data_predicted_probe=ekf->x.pData[0];
}

void EKF_Update(EKF_HandleTypeDef* ekf,  arm_matrix_instance_f32* z) {
    ekf->H_jacobian(&ekf->x, NULL, 0, &ekf->Hx);

    float z_pred_data[ekf->dim_z];
    arm_matrix_instance_f32 z_pred;
    arm_mat_init_f32(&z_pred, ekf->dim_z, 1, z_pred_data);

    ekf->h(&ekf->x, &z_pred);

    // y = z - h(x)
    float y_data[ekf->dim_z];
    arm_matrix_instance_f32 y;
    arm_mat_init_f32(&y, ekf->dim_z, 1, y_data);
    arm_mat_sub_f32(z, &z_pred, &y);

    // S = H P H^T + R
    float HP_data[ekf->dim_z * ekf->dim_x];
    float HxT_data[ekf->dim_x * ekf->dim_z];
    float S_data[ekf->dim_z * ekf->dim_z];
    arm_matrix_instance_f32 HP, HxT, S;
    arm_mat_init_f32(&HP, ekf->dim_z, ekf->dim_x, HP_data);
    arm_mat_init_f32(&HxT, ekf->dim_x, ekf->dim_z, HxT_data);
    arm_mat_init_f32(&S, ekf->dim_z, ekf->dim_z, S_data);

    arm_mat_mult_f32(&ekf->Hx, &ekf->P, &HP);
    arm_mat_trans_f32(&ekf->Hx, &HxT);
    arm_mat_mult_f32(&HP, &HxT, &S);
    arm_mat_add_f32(&S, &ekf->R, &S);

    // Compute inverse of S
    arm_matrix_instance_f32 S_inv;
    float S_inv_data[ekf->dim_z * ekf->dim_z];
    arm_mat_init_f32(&S_inv, ekf->dim_z, ekf->dim_z, S_inv_data);
    if (arm_mat_inverse_f32(&S, &S_inv) != ARM_MATH_SUCCESS) {
        // Handle inversion failure (singular matrix)
        return;
    }

    // K = P H^T S^{-1}
    float PHt_data[ekf->dim_x * ekf->dim_z];
    arm_matrix_instance_f32 PHt;
    arm_mat_init_f32(&PHt, ekf->dim_x, ekf->dim_z, PHt_data);

    arm_mat_mult_f32(&ekf->P, &HxT, &PHt);
    arm_mat_mult_f32(&PHt, &S_inv, &ekf->K);

    // x = x + K y
    float Ky_data[ekf->dim_x];
    arm_matrix_instance_f32 Ky;
    arm_mat_init_f32(&Ky, ekf->dim_x, 1, Ky_data);
    arm_mat_mult_f32(&ekf->K, &y, &Ky);
    arm_mat_add_f32(&ekf->x, &Ky, &ekf->x);

    // P = (I - K H) P
    float KH_data[ekf->dim_x * ekf->dim_x];
    arm_matrix_instance_f32 KH;
    arm_mat_init_f32(&KH, ekf->dim_x, ekf->dim_x, KH_data);
    arm_mat_mult_f32(&ekf->K, &ekf->Hx, &KH);

    float I_data[ekf->dim_x * ekf->dim_x];
    arm_matrix_instance_f32 I, I_minus_KH;
    arm_mat_init_f32(&I, ekf->dim_x, ekf->dim_x, I_data);
    arm_mat_init_f32(&I_minus_KH, ekf->dim_x, ekf->dim_x, I_data); // reuse I_data buffer

    // Create identity matrix
    for (uint16_t i = 0; i < ekf->dim_x * ekf->dim_x; i++) I_data[i] = 0.0f;
    for (uint16_t i = 0; i < ekf->dim_x; i++) I_data[i * ekf->dim_x + i] = 1.0f;

    arm_mat_sub_f32(&I, &KH, &I_minus_KH);
    arm_mat_mult_f32(&I_minus_KH, &ekf->P, &ekf->P);

    X_data_updated_probe = ekf->x.pData[0];
}

void EKF_Step(EKF_HandleTypeDef* ekf,  arm_matrix_instance_f32* u,  arm_matrix_instance_f32* z, float dt) {
    EKF_Predict(ekf, u, dt);
    EKF_Update(ekf, z);

}


