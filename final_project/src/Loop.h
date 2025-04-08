#pragma once

struct SDL_Window;
struct SDL_Renderer;
struct ImGuiContext;

/**
* @brief Main Loop
*/
class Loop {
private:
    SDL_Window* window;       // Reference to the SDL Window
    SDL_Renderer* renderer;   // Reference to the graphics interface

    ImGuiContext* imguiContext;   // ImGui context
    bool imguiInit;               // Flag to check if ImGui is initialized
    bool imguiInitRender;         // Flag to check if ImGui Renderer is initialized

    bool exit;   // Condition value for the continuous execution of the main loop

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
