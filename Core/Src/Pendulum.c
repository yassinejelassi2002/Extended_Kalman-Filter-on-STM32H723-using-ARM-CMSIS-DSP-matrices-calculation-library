


#include "pendulum.h"

/* Generate White Noise */
float generate_white_noise(float NOISE_AMPLITUDE ) {
    // Simple uniform white noise in range [-NOISE_AMPLITUDE, NOISE_AMPLITUDE]
    return ((float)rand() / RAND_MAX - 0.5f) * 2.0f * NOISE_AMPLITUDE;
}

/*Update Pendulum Dynamics */
void update_pendulum(void) {
	float  process_noise = 0.02f;
    float noise = generate_white_noise(process_noise);
    /* θ̈ = - (g / L) * sin(θ) + noise */
    float theta_ddot = - (G / L) * sinf(thetap) + noise;

    /* Euler integration */
    theta_dotp += theta_ddot*DT ;
    thetap += theta_dotp*DT ;
}



