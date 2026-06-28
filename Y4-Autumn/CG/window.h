#pragma once

#include <GLFW/glfw3.h>

void framebuffer_size_callback(GLFWwindow* window, int w, int h)
{
	glViewport(0, 0, w, h);
}

GLFWwindow * CreateWindow(int w, int h, const char* title) 
{ 
	//Initialize the library, allocate resources
	glfwInit();

    	//Specify minimum OpenGL version required to run this program (Major = 3, Minor = 3, i.e., Version 3.3)
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    	//Use core profile (i.e., a subset of OpenGL features without backwards-compatible features we no longer need
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//Create a window with specified width, height, and title
	GLFWwindow* window = glfwCreateWindow(w, h, title, NULL, NULL);

	//Make the window our current context
	glfwMakeContextCurrent(window);

	//Specify the name of the callback function when an event on the window is detected
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	//Return the reference to the window object
	return window;
}
