/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".

    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _PARTICLE_HPP
#define _PARTICLE_HPP
#include "types.hpp"
#include <cmath>

struct ParticleAoS {
public:
  ParticleAoS() { init(); }
  void init() {
    pos[0] = 0.;
    pos[1] = 0.;
    pos[2] = 0.;
    vel[0] = 0.;
    vel[1] = 0.;
    vel[2] = 0.;
    acc[0] = 0.;
    acc[1] = 0.;
    acc[2] = 0.;
    mass = 0.;
  }
  real_type pos[3];
  real_type vel[3];
  real_type acc[3];
  real_type mass;
};

struct ParticleSoA {
public:
  ParticleSoA(int n) { init(n); }
  ~ParticleSoA() {
    delete[] pos_x;
    delete[] pos_y;
    delete[] pos_z;
    delete[] vel_x;
    delete[] vel_y;
    delete[] vel_z;
    delete[] acc_x;
    delete[] acc_y;
    delete[] acc_z;
    delete[] mass;
  }
  void init(int n) {
    pos_x = new real_type[n];
    pos_y = new real_type[n];
    pos_z = new real_type[n];
    vel_x = new real_type[n];
    vel_y = new real_type[n];
    vel_z = new real_type[n];
    acc_x = new real_type[n];
    acc_y = new real_type[n];
    acc_z = new real_type[n];
    mass  = new real_type[n];
  }
  real_type *pos_x, *pos_y, *pos_z;
  real_type *vel_x, *vel_y, *vel_z;
  real_type *acc_x, *acc_y, *acc_z;
  real_type *mass;
};
#endif
