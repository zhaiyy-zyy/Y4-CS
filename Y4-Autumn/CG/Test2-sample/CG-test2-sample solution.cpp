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
    .5f, .0f, 0.f,          //t 顶点4 (Top)
    .25f, -.25f, 0.f,       //bl 顶点5 (Bottom Left)
    .75f, -.25f, 0.f,       //br 顶点6 (Bottom Right)

};




//根据header文件完成的
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
    //计算三角形数量并写回*n_tris，每三个顶点，一个三角形
    *n_tris = n_verts / 3;
    //按三角形个数分配一块连续内存，存放triangle数组
    triangle* tris = (triangle*)malloc(sizeof(triangle) * (*n_tris));

    //遍历每一个三角形，tc是triangle count（当前三角形的索引）
    for (int tc = 0; tc < (*n_tris); tc++)
    {
        //每个三角形消耗3个顶点*每个顶点3个分量 = 9个float，所以第 tc 个三角形在 verts 里的起始下标是 tc * 9。
        
        //第一个顶点（x0,y0,z0）
        tris[tc].v1.pos.x = verts[(tc * 9) + 0];
        tris[tc].v1.pos.y = verts[(tc * 9) + 1];
        tris[tc].v1.pos.z = verts[(tc * 9) + 2];

        //第二个顶点（x1,y1,z1）
        tris[tc].v2.pos.x = verts[(tc * 9) + 3];
        tris[tc].v2.pos.y = verts[(tc * 9) + 4];
        tris[tc].v2.pos.z = verts[(tc * 9) + 5];

        //第三个顶点（x2,y2,z2）
        tris[tc].v3.pos.x = verts[(tc * 9) + 6];
        tris[tc].v3.pos.y = verts[(tc * 9) + 7];
        tris[tc].v3.pos.z = verts[(tc * 9) + 8];
    }

    //返回你刚分配并填好的三角形数组指针。
    return tris;
}

//把三角形三个顶点坐标从NDC sapce[-1,1],映射到Screen space[0, width-1]*[0, height-1]，公式（x+1）*width/2；（y+1）*height/2
void TransformToViewport(int width, int height, triangle* tri)
{
    //把 v1 顶点的 x 从 [-1,1] 转到 [0,width]
    tri->v1.pos.x = ((tri->v1.pos.x + 1) / 2) * width;
    //把 v1 顶点的 y 从 [-1,1] 转到 [0,height]
    tri->v1.pos.y = ((tri->v1.pos.y + 1) / 2) * height;

    tri->v2.pos.x = ((tri->v2.pos.x + 1) / 2) * width;
    tri->v2.pos.y = ((tri->v2.pos.y + 1) / 2) * height;

    tri->v3.pos.x = ((tri->v3.pos.x + 1) / 2) * width;
    tri->v3.pos.y = ((tri->v3.pos.y + 1) / 2) * height;
}

//计算重心坐标
void ComputeBarycentricCoordinates(float x, float y, triangle t, float& alpha, float& beta, float& gamma)
{
    //重心坐标的公式计算alpha， beta， gamma
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

//片段着色器
void ShadeFragment(float alpha, float beta, float gamma, triangle t, int& r, int& g, int& b)
{
    //设置为纯红色
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
