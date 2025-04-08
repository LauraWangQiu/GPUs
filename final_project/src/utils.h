#pragma once
#include <SDL_stdinc.h>

/**
* @brief Structure to store r, g, b and a channels for the color
*/
struct Color {
    Uint8 r, g, b, a;
};

/**
* @brief Structure to store particle information like position,
* velocity, color, time left
*/
struct Particle {
    float posX, posY;   // position
    float velX, velY;   // velocity
    Color color;        // color
    float timeLeft;     // time left until disappear
};
