# COMP3069 Computer Graphics: Coursework (Assessment 3)
## 3D Interactive Boat Scene using OpenGL

---

## 1. Project Overview

This project is a 3D interactive graphics application developed using **OpenGL** as part of the COMP3069 coursework.

The program renders an animated boat scene surrounded by water, sky, islands, and trees.  
All major scene objects are created **procedurally in code**, without using any external 3D modelling software.

The application demonstrates:

- Procedural geometry generation  
- Hierarchical transformations  
- Multiple light sources (Phong lighting)  
- Bitmap texture mapping  
- Camera systems (model-viewer and fly-through)  
- Real-time animation and user interaction  
- Anti-aliasing and depth testing  

---

## 2. Keyboard Controls

| Key | Action |
|----|------|
| C | Toggle camera mode |
| W / A / S / D | Fly-through movement |
| Mouse | Fly-through look control |
| Arrow Keys | Orbit camera (model-viewer mode) |
| R / F | Zoom in / out (model-viewer mode) |
| A / D | Rotate boat (model-viewer mode) |
| 1 | Toggle day / night (directional light) |
| 2 | Toggle positional light |
| 3 | Toggle spotlight |
| Q | Reset fly camera |
| ESC | Exit program |

> **Note:**  
> If single-letter key inputs are not recognised correctly on some systems,
> the corresponding key combinations (e.g. `Ctrl + W`, `Ctrl + A`, etc.) can be
> used instead. This avoids the operating system interpreting single letters
> as text input. If an English input method is active, the keys can be used
> directly without any modifier.

---

## 3. Development Environment

- **Language:** C++  
- **Graphics API:** OpenGL   
- **Shader Language:** GLSL (vertex and fragment shaders)  

- **Libraries:**  
  - GLFW (window creation and input handling)  
  - GLAD (OpenGL function loader)  
  - GLM (matrix and vector mathematics)  

- **Platform:** Windows (the same environment as used in the lab sessions, where bitmap textures are loaded via the Windows GDI API)

---

## 4. File Structure

```text
/CW3
 ├── CW3.sln                 # Visual Studio solution file
 ├── CW3.vcxproj             # Project configuration
 ├── CW3.vcxproj.filters
 ├── CW3.cpp                 # Main program entry point
 ├── camera.h
 ├── ModelViewerCamera.h
 ├── shader.h
 ├── texture.h
 ├── bitmap.h
 ├── util.h
 ├── window.h
 ├── phong.vert              
 ├── phong.frag              
 ├── sky.vert                
 ├── sky.frag                
 ├── sky.bmp                 # Sky texture
 ├── water.bmp               # Water texture
 └── README.md               # Build and run instructions
```

---

## 5. Procedural Geometry

All geometry in the scene is generated procedurally in C++. Instead, all objects are constructed by explicitly defining vertex positions, computing surface normals, and assembling triangle-based meshes programmatically. This approach demonstrates direct control over geometric structure, surface continuity, and level of detail.

Generated objects include:

- Boat hull (slice–stack construction with smooth normals)  
- Front and rear decks  
- Cabin with arched windows  
- Roof and rim structures  
- Chimney, drain pipe, steering wheel  
- Cargo boxes  
- License plate and procedural text  
- Islands with height-based terrain variation  
- Trees composed of trunk and layered foliage  

---

## 6. Texture Implementation

Bitmap textures are loaded using a custom Windows GDI BMP loader.

- BMP headers (`BITMAPFILEHEADER` and `BITMAPINFOHEADER`) are read manually  
- Pixel data is loaded into a dynamic buffer  
- BMP colour channels are stored in **BGR** order and converted to **RGB**  
- Textures are uploaded using `glTexImage2D`  

### Texture Filtering

- **Magnification:** `GL_LINEAR`  
- **Minification:** `GL_LINEAR_MIPMAP_LINEAR`  
- **Mipmaps:** Generated using `glGenerateMipmap`  
- **Wrapping:** `GL_REPEAT`  

### Textures Applications

- **Water surface:** `water.bmp` texture
- **Sky background:** `sky.bmp` texture used as environment background

---

## 7. Lighting System

Phong lighting is implemented in the fragment shader with:

- Directional light (day/night switching)  
- Positional light with attenuation and animation  
- Spotlight attached to the camera  

All lights can be toggled using keyboard input.

### Lighting Controls

- **`1`** — Toggle **day / night** mode (directional light intensity)  
- **`2`** — Toggle **positional light** (animated point light with attenuation)  
- **`3`** — Toggle **spotlight** attached to the camera  

---

## 8. Camera System

Two camera modes are supported.

### Model-Viewer Camera

- Orbits around the boat  
- Controlled using arrow keys  
- Zoom using `R` and `F`  
- Boat rotation using `A` and `D`  

### Fly-Through Camera

- First-person navigation  
- Movement using `W`, `A`, `S`, `D`  
- Mouse controls orientation  
- Reset view using `Q`  

Press **`C`** to switch between camera modes.

> **Note:**  
> If single-letter key inputs are not recognised correctly on some systems,
> the corresponding key combinations (e.g. `Ctrl + W`, `Ctrl + A`, etc.) can be
> used instead. This avoids the operating system interpreting single letters
> as text input. If an English input method is active, the keys can be used
> directly without any modifier.
---

## 9. Animation

Real-time animations include:

- Boat bobbing, rolling, and pitching  
- Steering wheel rotation  
- Water wave simulation in vertex shader 
- Animated positional light  

Animations are time-based using `glfwGetTime()` and `deltaTime` for frame-rate independence.

---

## 10. Anti-Aliasing and Depth Testing

- Multisample Anti-Aliasing (MSAA) enabled using:
  ```cpp
  glEnable(GL_MULTISAMPLE);
  ```
- Depth testing enabled using:
  ```cpp
  glEnable(GL_DEPTH_TEST);
  ```