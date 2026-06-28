#define _USE_MATH_DEFINES 
#include <iostream>
#include <SDL.h>
#include <Windows.h>
#undef main

#include "DoNotModify.h"


//因为这是 2D 光栅化器，所以 z 一律是 0.f。
float vertices[] =
{
    //pos            
    -.5f, .5f, 0.f,         //tl 顶点1 (Top Left, tl)
    .5f, .5f, 0.f,          //tr 顶点2 (Top Right, tr)
    .0f, .0f, 0.f,          //b 顶点3 (Bottom, b)

    //pos
    -.5f, 0.f, 0.f,
    .5f, .25f, 0.f,
    1.f, .25f, 0.f,

};

//根据header文件完成的
/* 
colour_buffer 是一个 [PIXEL_W][PIXEL_H] 大小的 2D 数组，每个元素是 vec3,用来存每个像素的颜色。
vec3 的 .x/.y/.z 对应 R/G/B。
这段代码就是把屏幕上所有像素都填成你传进来的 (r,g,b) 颜色。

细节解释:
外层 for (y)：遍历每一行（从上到下）。
内层 for (x)：遍历每一列（从左到右）。
colour_buffer[x][y].x = r;:这个像素点的红色分量设为 r（0–255）。
这样清完以后，屏幕就是一个纯色背景。

对应渲染流水线里的 “clear framebuffer”。
考点:for 范围必须是 < PIXEL_H / PIXEL_W,绝不能写成 <=，否则数组越界。
*/
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
//将每三个顶点组成一个三角形
triangle* AssembleTriangles(float* verts, int n_verts, int* n_tris)
{
    //计算三角形数量并写回*n_tris，每三个顶点组成一个三角形
    *n_tris = n_verts / 3;

    //按三角形个数分配一块连续内存，存放triangle数组
    triangle* tris = (triangle*)malloc(sizeof(triangle) * (*n_tris));

    //遍历每个三角形，i是当前三角形的索引
    for (int i = 0; i < (*n_tris); i++)
    {
        //每个三角形消耗3个顶点*每个顶点3个分量 = 9个float，所以第 i 个三角形在 verts 里的起始下标是 i * 9。
        tris[i].v1.pos.x = verts[(i * 9) + 0];
        tris[i].v1.pos.y = verts[(i * 9) + 1];
        tris[i].v1.pos.z = verts[(i * 9) + 2];

        tris[i].v2.pos.x = verts[(i * 9) + 3];
        tris[i].v2.pos.y = verts[(i * 9) + 4];
        tris[i].v2.pos.z = verts[(i * 9) + 5];

        tris[i].v3.pos.x = verts[(i * 9) + 6];
        tris[i].v3.pos.y = verts[(i * 9) + 7];
        tris[i].v3.pos.z = verts[(i * 9) + 8];

    }
    
    return tris;
}

//把三角形三个顶点坐标从NDC sapce[-1,1],映射到Screen space[0, width-1]*[0, height-1]，公式（x+1）*width/2；（y+1）*height/2
void TransformToViewport(int width, int height, triangle* tri)
{
    tri->v1.pos.x = ((tri->v1.pos.x + 1) / 2) * width;
    tri->v1.pos.y = ((tri->v1.pos.y + 1) / 2) * height;

    tri->v2.pos.x = ((tri->v2.pos.x + 1) / 2) * width;
    tri->v2.pos.y = ((tri->v2.pos.y + 1) / 2) * height;

    tri->v3.pos.x = ((tri->v3.pos.x + 1) / 2) * width;
    tri->v3.pos.y = ((tri->v3.pos.y + 1) / 2) * height;
}

//计算重心坐标
void ComputeBarycentricCoordinates(float x, float y, triangle t, float& alpha, float& beta, float& gamma)
{
    float BCP = (t.v3.pos.y - t.v2.pos.y) * x + (t.v2.pos.x - t.v3.pos.x) * y + (t.v3.pos.x *t.v2.pos.y) - (t.v2.pos.x * t.v3.pos.y);
    float BCA = (t.v3.pos.y - t.v2.pos.y) * t.v1.pos.x + (t.v2.pos.x - t.v3.pos.x) * t.v1.pos.y + (t.v3.pos.x *t.v2.pos.y) - (t.v2.pos.x * t.v3.pos.y);
    alpha = BCP / BCA;

    float ACP = (t.v3.pos.y - t.v1.pos.y) * x + (t.v1.pos.x - t.v3.pos.x) * y + (t.v3.pos.x *t.v1.pos.y) - (t.v1.pos.x * t.v3.pos.y);
    float ACB = (t.v3.pos.y - t.v1.pos.y) * t.v2.pos.x + (t.v1.pos.x - t.v3.pos.x) * t.v2.pos.y + (t.v3.pos.x *t.v1.pos.y) - (t.v1.pos.x * t.v3.pos.y);
    beta = ACP / ACB;

    float ABP = (t.v2.pos.y - t.v1.pos.y) * x + (t.v1.pos.x - t.v2.pos.x) * y + (t.v2.pos.x *t.v1.pos.y) - (t.v1.pos.x * t.v2.pos.y);
    float ABC = (t.v2.pos.y - t.v1.pos.y) * t.v3.pos.x + (t.v1.pos.x - t.v2.pos.x) * t.v3.pos.y + (t.v2.pos.x *t.v1.pos.y) - (t.v1.pos.x * t.v2.pos.y);
    gamma = ABP / ABC;
}

//片段着色器
void ShadeFragment(float alpha, float beta, float gamma, triangle t, int& r, int& g, int& b)
{
    r = 255;
    g = 0;
    b = 0;
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

        //调用 AssembleTriangles() 函数，把顶点数组组装成三角形
        triangl* tris= AssembleTriangles(vertices, 6, &n_tris);

        //对每个三角形调用 TransformToViewport()，把它的顶点从 NDC 坐标变换成屏幕坐标。
        for (int i = 0; i < n_tris; i++)
        {
            TransformToViewport(PIXEL_W; PIXEL_H, &tris[i]);
        }

        //光栅化
        for (int py = 0; py < PIXEL_H; py++)
        {
            for (int px = 0; px < PIXEL_W; px++)
            {
                for (int i = 0; i < n_tris; i++)
                {
                    // 判断像素是否在三角形内部
                    float alpha, beta, gamma;
                    ComputeBarycentricCoordinates(px, py, tris[i], alpha, beta, gamma);

                    if(alpha >= 0.f && aplha <= 1.0f &&
                       beta >= 0.f && beta <= 1.0f &&
                       gamma >= 0.f && gemma <= 1.0f)
                       {
                        int r, g, b;
                        ShadeFragment(alpha, beta, gamma, tris[i], r, g, b);

                        //翻转
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
