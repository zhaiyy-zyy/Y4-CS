#define _USE_MATH_DEFINES 
#include <iostream>
#include <SDL.h>
#include <Windows.h>
#undef main

#include "DoNotModify.h"


float vertices[] =
{
    //pos            
    -.5f, .5f, 0.f,         //tl 顶点1 (Top Left, tl)
    .5f, .5f, 0.f,          //tr 顶点2 (Top Right, tr)
    .0f, .0f, 0.f,          //b 顶点3 (Bottom, b)


    //pos            
    .5f, .0f, 0.f,          //t 顶点4 (Top)
    .25f, -.25f, 0.f,       //bl 顶点5 (Bottom Left)
    .75f, -.25f, 0.f,       //br 顶点6 (Bottom Right)

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
        tris[tc].v1.pos.x = verts[(tc * 9) + 0];
        tris[tc].v1.pos.y = verts[(tc * 9) + 1];
        tris[tc].v1.pos.z = verts[(tc * 9) + 2];

        tris[tc].v2.pos.x = verts[(tc * 9) + 3];
        tris[tc].v2.pos.y = verts[(tc * 9) + 4];
        tris[tc].v2.pos.z = verts[(tc * 9) + 5];

        tris[tc].v3.pos.x = verts[(tc * 9) + 6];
        tris[tc].v3.pos.y = verts[(tc * 9) + 7];
        tris[tc].v3.pos.z = verts[(tc * 9) + 8];
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
    r = 255; //红色
    g = 0; //不含绿色
    b = 0; //不含蓝色
}


int main()
{
    COMP3069StartSDL();
    ClearColourBuffer(0, 0, 0);
    while (1)
    {
        if (COMP3069PressedEscape())
            break;

        ClearColourBuffer(255, 255, 255);

        //定义三角形数量
        int n_tris = 0;

        //调用 AssembleTriangles() 函数，把顶点数组组装成三角形。
        triangle* tris = AssembleTriangles(vertices, 6, &n_tris);

        //对每个三角形调用 TransformToViewport()，把它的顶点从 NDC 坐标变换成屏幕坐标。
        for (int tc = 0; tc < n_tris; tc++)
        {
            TransformToViewport(PIXEL_W, PIXEL_H, &tris[tc]);
        }

        //光栅化
        for (int py = 0; py < PIXEL_H; py++) //扫描所有行（Y）
        {
            for (int px = 0; px < PIXEL_W; px++) //扫描所有列（X）
            {
                for (int tc = 0; tc < n_tris; tc++) //遍历每个三角形
                {
                    // 判断像素是否在三角形内部
                    float alpha, beta, gamma;
                    ComputeBarycentricCoordinates(px, py, tris[tc], alpha, beta, gamma);

                    if (alpha >= 0.f && alpha <= 1.0f &&
                        beta >= 0.f && beta <= 1.0f &&
                        gamma >= 0.f && gamma <= 1.0f)
                    {
                        int r, g, b;
                        ShadeFragment(alpha, beta, gamma, tris[tc], r, g, b);

                        // 写入颜色缓冲（注意Y翻转）
                        //colour_buffer[px][py].x = r;
                        //colour_buffer[px][py].y = g;
                        //colour_buffer[px][py].z = b;
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


