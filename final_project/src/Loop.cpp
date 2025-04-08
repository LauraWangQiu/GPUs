#include "Loop.h"
#include "defs.h"
#include <string>
#include <SDL.h>
#include <SDL_render.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>
using namespace std;

Loop::Loop()
    : window(nullptr), renderer(nullptr), imguiContext(nullptr), imguiInit(false), imguiInitRender(false), exit(false) {
}

Loop::~Loop() {
    if (renderer != nullptr) SDL_DestroyRenderer(renderer);
    if (window != nullptr) SDL_DestroyWindow(window);

    if (imguiInitRender) ImGui_ImplSDLRenderer2_Shutdown();
    if (imguiInit) ImGui_ImplSDL2_Shutdown();
    if (imguiContext != nullptr) ImGui::DestroyContext();

    SDL_Quit();

    window = nullptr;
    renderer = nullptr;
}

bool Loop::init() {

    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        debugLog(("SDL_Init Error: " + string(SDL_GetError())).c_str());
        return false;
    }

    Uint32 windowFlags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    window = SDL_CreateWindow("PruebaSDL", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 640, 480, windowFlags);
    if (window == nullptr) {
        debugLog(("SDL_CreateWindow Error: " + string(SDL_GetError())).c_str());
        return false;
    }

    Uint32 rendererFlags = SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC;
    renderer = SDL_CreateRenderer(window, -1, rendererFlags);
    if (renderer == nullptr) {
        debugLog(("SDL_CreateRenderer Error: " + string(SDL_GetError())).c_str());
        return false;
    }

    IMGUI_CHECKVERSION();
    imguiContext = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.WantCaptureKeyboard = true;
    io.WantCaptureMouse = true;
    ImGui::StyleColorsDark();
    imguiInit = ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    if (!imguiInit) {
        debugLog("ImGui_ImplSDL2_InitForSDLRenderer Error");
        return false;
    }
    imguiInitRender = ImGui_ImplSDLRenderer2_Init(renderer);
    if (!imguiInit) {
        debugLog("ImGui_ImplSDLRenderer2_Init Error");
        return false;
    }

    return true;
}

void Loop::run() {
    while (!exit) {
        handleEvents();
        update();
        refresh();
        render();
    }
}

void Loop::handleEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event) != 0) {
        ImGui_ImplSDL2_ProcessEvent(&event);

        switch (event.type) {
        case SDL_WINDOWEVENT:
            switch (event.window.event) {
            case SDL_WINDOWEVENT_CLOSE: quit(); break;
            default: break;
            }
            break;
        case SDL_KEYDOWN:
            switch (event.key.keysym.sym) {
            case SDLK_ESCAPE: quit(); break;
            default: break;
            }
            break;
        case SDL_QUIT: quit(); break;
        default: break;
        }
    }
}

void Loop::update() { }

void Loop::refresh() { }

void Loop::render() {
    if (renderer != nullptr) {
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        //ImGui::ShowDemoWindow();

        //ImGui::Begin("Text Box");
        //ImGui::Text("Hello world!");
        //ImGui::End();

        // ... more ImGui widgets here

        ImGui::Render();

        SDL_RenderClear(renderer);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
    }
}

void Loop::quit() { exit = true; }
