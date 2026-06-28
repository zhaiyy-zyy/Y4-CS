#define _USE_MATH_DEFINES 
#include <iostream>
#include <SDL.h>
#include <Windows.h>
#undef main

#include "DoNotModify.h"



float vertices[] =
{
    //pos:     
    -.5f, .75f, 0.f,         //tl
    .5f, .75f, 0.f,          //tr
    .0f, .0f, 0.f,           //b

    //rectangle:a widthto-height ratio of 1:2
    // pos:Rectangle Triangle 1:rectangle at the centre of the screen spac
     0.0f,  0.0f, 0.0f,   //tl:centre of screen after viewport
     0.0f, -1.0f, 0.0f,   //bl
     0.5f, -1.0f, 0.0f,   //br

     // pos:Rectangle Triangle 2:rectangle at the centre of the screen spac
      0.0f,  0.0f, 0.0f,   //tl
      0.5f, -1.0f, 0.0f,   //br
      0.5f,  0.0f, 0.0f,   //tr
};


void ClearColourBuffer(int r, int g, int b)
{
    for (int y = 0; y < PIXEL_H; y++)
    {
        for (int x = 0; x < PIXEL_W; x++)
        {
            colour_buffer[x][y].x = r;
            colour_buffer[x][y].y = g;
            colour_buffer[x][y].z = b;
        }
    }
}

triangle* AssembleTriangles(float* verts, int n_verts, int* n_tris)
{
    *n_tris = n_verts / 3;

    triangle* tris = (triangle*)malloc(sizeof(triangle) * (*n_tris));

    for (int tc = 0; tc < (*n_tris); tc++)
    {
        // read original vertices
        float x1 = verts[(tc * 9) + 0];
        float y1 = verts[(tc * 9) + 1];

        float x2 = verts[(tc * 9) + 3];
        float y2 = verts[(tc * 9) + 4];

        float x3 = verts[(tc * 9) + 6];
        float y3 = verts[(tc * 9) + 7];

        // Transform triangle (index 0)
        if (tc == 0)
        {
            // scale first
            float s = 0.6f;
            x1 *= s;  y1 *= s;
            x2 *= s;  y2 *= s;
            x3 *= s;  y3 *= s;

            // then translate to top-left
            float tx = -0.5f;
            float ty = 0.4f;

            x1 += tx;  y1 += ty;
            x2 += tx;  y2 += ty;
            x3 += tx;  y3 += ty;
        }

        // Transform rectangle (index 1 & 2)
        if (tc == 1 || tc == 2)
        {
            float sx = 0.5f;
            float sy = 0.5f;

            x1 *= sx;  y1 *= sy;
            x2 *= sx;  y2 *= sy;
            x3 *= sx;  y3 *= sy;
            // TL stays at (0,0)
        }

        // final positions
        tris[tc].v1.pos.x = x1;
        tris[tc].v1.pos.y = y1;
        tris[tc].v1.pos.z = tc;     // store triangle index here

        tris[tc].v2.pos.x = x2;
        tris[tc].v2.pos.y = y2;
        tris[tc].v2.pos.z = tc;

        tris[tc].v3.pos.x = x3;
        tris[tc].v3.pos.y = y3;
        tris[tc].v3.pos.z = tc;
    }

    return tris;
}
void TransformToViewport(int width, int height, triangle* tri)
{
    
    tri->v1.pos.x = ((tri->v1.pos.x + 1) / 2) * width;
    tri->v1.pos.y = ((tri->v1.pos.y + 1) / 2) * height;

    tri->v2.pos.x = ((tri->v2.pos.x + 1) / 2) * width;
    tri->v2.pos.y = ((tri->v2.pos.y + 1) / 2) * height;

    tri->v3.pos.x = ((tri->v3.pos.x + 1) / 2) * width;
    tri->v3.pos.y = ((tri->v3.pos.y + 1) / 2) * height;


}
void ComputeBarycentricCoordinates(float x, float y, triangle t, float& alpha, float& beta, float& gamma)
{
    float BCP = (t.v3.pos.y - t.v2.pos.y) * x + (t.v2.pos.x - t.v3.pos.x) * y + (t.v3.pos.x * t.v2.pos.y) - (t.v2.pos.x * t.v3.pos.y);
    float BCA = (t.v3.pos.y - t.v2.pos.y) * t.v1.pos.x + (t.v2.pos.x - t.v3.pos.x) * t.v1.pos.y + (t.v3.pos.x * t.v2.pos.y) - (t.v2.pos.x * t.v3.pos.y);
    alpha = BCP / BCA;

    float ACP = (t.v3.pos.y - t.v1.pos.y) * x + (t.v1.pos.x - t.v3.pos.x) * y + (t.v3.pos.x * t.v1.pos.y) - (t.v1.pos.x * t.v3.pos.y);
    float ACB = (t.v3.pos.y - t.v1.pos.y) * t.v2.pos.x + (t.v1.pos.x - t.v3.pos.x) * t.v2.pos.y + (t.v3.pos.x * t.v1.pos.y) - (t.v1.pos.x * t.v3.pos.y);
    beta = ACP / ACB;

    float ABP = (t.v2.pos.y - t.v1.pos.y) * x + (t.v1.pos.x - t.v2.pos.x) * y + (t.v2.pos.x * t.v1.pos.y) - (t.v1.pos.x * t.v2.pos.y);
    float ABC = (t.v2.pos.y - t.v1.pos.y) * t.v3.pos.x + (t.v1.pos.x - t.v2.pos.x) * t.v3.pos.y + (t.v2.pos.x * t.v1.pos.y) - (t.v1.pos.x * t.v2.pos.y);
    gamma = ABP / ABC;
}

void ShadeFragment(float alpha, float beta, float gamma, triangle t, int& r, int& g, int& b)
{
    if ((int)t.v1.pos.z == 0)
    {
        r = 0;
        g = 255;
        b = 255;
    }
    else
    {
        r = 255;
        g = 204;
        b = 255;
    }
}



int main()
{
    COMP3069StartSDL();

    while (1)
    {
        if (COMP3069PressedEscape())
            break;

        ClearColourBuffer(255, 255, 255);

        int n_tris = 0;

        triangle* tris = AssembleTriangles(vertices, 9, &n_tris);

        for (int tc = 0; tc < n_tris; tc++)
        {
            TransformToViewport(PIXEL_W, PIXEL_H, &tris[tc]);
        }

        for (int py = 0; py < PIXEL_H; py++)
        {
            for (int px = 0; px < PIXEL_W; px++) 
            {
                for (int tc = 0; tc < n_tris; tc++) 
                {

                    float alpha, beta, gamma;
                    ComputeBarycentricCoordinates(px, py, tris[tc], alpha, beta, gamma);

                    if (alpha >= 0.f && alpha <= 1.0f &&
                        beta >= 0.f && beta <= 1.0f &&
                        gamma >= 0.f && gamma <= 1.0f)
                    {
                        int r, g, b;
                        
                        ShadeFragment(alpha, beta, gamma, tris[tc], r, g, b);

                        colour_buffer[px][PIXEL_H - py - 1].x = r;
                        colour_buffer[px][PIXEL_H - py - 1].y = g;
                        colour_buffer[px][PIXEL_H - py - 1].z = b;
                    }
                }
            }
        }
        COMP3069DisplayColourBuffer();
    }

    COMP3069StopSDL();

    return EXIT_SUCCESS;
}

