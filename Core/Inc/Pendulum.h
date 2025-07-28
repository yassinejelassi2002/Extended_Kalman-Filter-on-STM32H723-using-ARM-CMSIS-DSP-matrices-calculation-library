/*
 * Pendulum.h
 *
 *  Created on: Jul 23, 2025
 *      Author: yassine jelassi
 */

// pendulum.h
#ifndef __PENDULUM_H__
#define __PENDULUM_H__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* Constants */
#define G               9.81f       // Gravity (m/s^2)
#define L               0.2f        // Pendulum length (m)
#define DT              0.01f      // Time step (s)

#ifndef PI
#define PI                 3.14159265358979f
#endif
/* State Variables */
extern float thetap;        // Angle (rad)
extern float theta_dotp;    // Angular velocity (rad/s)
/* Function Prototypes */

/* Updates the pendulum's state using Euler integration */
void update_pendulum(void);


/* Generates a white noise sample */
float generate_white_noise(float NOISE_AMPLITUDE);

#endif // __PENDULUM_H__

