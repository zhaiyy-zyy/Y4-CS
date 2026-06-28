#pragma once
#include <glad/glad.h>
#include "util.h"

//When 'LoadShader' function is called, 'vertexShaderFile' refers to "vertex.vert", 'fragmentShaderFile' refers to "fragment.frag"
unsigned int LoadShader(const char* vertexShaderFile, const char* fragmentShaderFile) 
{ 
	int success;
	char infoLog[512];

	//Create a vertex shader object called 'vertexShader'
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);

	//'vertexShaderSource' will be the pointer to the vertex shader source program
	char* vertexShaderSource = read_file(vertexShaderFile);

	//Load the vertex shader source program to the graphics memory, and set 'vertexShader' as the pointer
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);

	//Compile the vertex shader program
	glCompileShader(vertexShader);

	//Query compilation status of vertex shader program
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		//Get information log regarding shader's compilation
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}


	//Create a fragment shader object called 'fragmentShader'
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	//'fragmentShaderSource' will be the pointer to the fragment shader source program
	char* fragmentShaderSource = read_file(fragmentShaderFile);

	//Load the fragment shader source program to the graphics memory, and set 'fragmentShader' as the pointer
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);

	//Compile the fragment shader program
	glCompileShader(fragmentShader);

	//Query compilation status of fragment shader program
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		//Get information log regarding shader's compilation
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	//Create shader program object
	unsigned int shaderProgram = glCreateProgram();

	//Attach both vertex and fragment shaders to the shader program object
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	//Link the vertex and fragment shaders
	glLinkProgram(shaderProgram);

	//Free the memory allocated to the vertex and fragment shader source programs
	free(fragmentShaderSource);
	free(vertexShaderSource);

	//Delete the pointers to the vertex and fragment shaders
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	//Return the pointer to the linked shader program
	return(shaderProgram);

}