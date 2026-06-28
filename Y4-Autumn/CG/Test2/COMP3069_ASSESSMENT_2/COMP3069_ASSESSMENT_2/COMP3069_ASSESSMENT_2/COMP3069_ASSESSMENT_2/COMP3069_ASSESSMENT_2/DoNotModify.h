#include <SDL.h>


struct vec3
{
    float x, y, z;
};

struct vertex
{
    vec3 pos;
};

struct triangle
{
    vertex v1, v2, v3;
};

#define PIXEL_W 800
#define PIXEL_H 600
vec3 colour_buffer[PIXEL_W][PIXEL_H];


SDL_Window* window;
SDL_Renderer* renderer;
void COMP3069StartSDL()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(PIXEL_W, PIXEL_H, 0, &window, &renderer);
}

void COMP3069StopSDL()
{
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

bool COMP3069PressedEscape()
{
    SDL_Event event;

    if (SDL_PollEvent(&event) && event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
    {
        return true;
    }

    return false;
}

void COMP3069DisplayColourBuffer()
{
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    for (int pixel_y = 0; pixel_y < PIXEL_H; ++pixel_y)
    {
        for (int pixel_x = 0; pixel_x < PIXEL_W; ++pixel_x)
        {
            float pixel_r = colour_buffer[pixel_x][pixel_y].x;
            float pixel_g = colour_buffer[pixel_x][pixel_y].y;
            float pixel_b = colour_buffer[pixel_x][pixel_y].z;

            SDL_SetRenderDrawColor(renderer, pixel_r, pixel_g, pixel_b, 255);
            SDL_RenderDrawPoint(renderer, pixel_x, pixel_y);
        }
    }

    SDL_RenderPresent(renderer);

}
