#pragma once
#include "utils.h"
#include <vector>

struct SDL_Window;
struct SDL_Renderer;
struct ImGuiContext;

/**
* @brief Main Loop
*/
class Loop {
private:
    const char* windowTitle;  // Window title
    int windowWidth;          // Window width
    int windowHeight;         // Window height
    SDL_Window* window;       // Reference to the SDL Window
    SDL_Renderer* renderer;   // Reference to the graphics interface

    ImGuiContext* imguiContext;   // ImGui context
    bool imguiInit;               // Flag to check if ImGui is initialized
    bool imguiInitRender;         // Flag to check if ImGui Renderer is initialized

    bool exit;          // Condition value for the continuous execution of the main loop
    Uint32 lastTime;    // Time from last time
    float deltaTime;    // Time from last frame

    /**
    * @brief Finishes the main loop establishing "exit" boolean to false value
    */
    void quit();

    /**
    * @brief Manages keyboard, mouse, window events
    */
    void handleEvents();
    /**
    * @brief Updates the current alive entities
    */
    void update();
    /**
    * @brief Removes the entities marked as not alive
    */
    void refresh();
    /**
    * @brief Renders on screen the current alive entities
    */
    void render();

private:
    Color backgroundCol;             // Background color
    Color particleCol;               // Particles color
    float particleTimeLeft;          // Particles alive time
    std::vector<Particle> particles; // All particles

    int numParticlesToGen;           // Number of particles to generate at a time

    /**
    * @brief Renders simulation
    */
    void renderSimulation();

    /**
    * @brief Renders interface
    */
    void renderInterface();

public:
    /**
    * Constructor of the main loop
    */
    Loop();
    /**
    * Destructor of the main loop. Destroys the window, renderer, entities...
    */
    ~Loop();
    /**
    * Inits SDL Window and Renderer
    */
    bool init();
    /**
    * Runs the main loop
    */
    void run();
};
