#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>

#include "Trajectory.h"
#include "ObstacleMap.h"
#include "collision_utils.h"

// Camera class for 3D navigation
class Camera3D {
public:
    glm::vec3 position;
    glm::vec3 target;
    glm::vec3 up;
    float yaw;
    float pitch;
    float distance;
    float fov;
    
    Camera3D() 
        : position(400.0f, 400.0f, 800.0f)
        , target(400.0f, 400.0f, 400.0f)
        , up(0.0f, 1.0f, 0.0f)
        , yaw(-90.0f)
        , pitch(0.0f)
        , distance(800.0f)
        , fov(45.0f) {}
    
    glm::mat4 getViewMatrix() const {
        return glm::lookAt(position, target, up);
    }
    
    glm::mat4 getProjectionMatrix(float width, float height) const {
        return glm::perspective(glm::radians(fov), width / height, 0.1f, 5000.0f);
    }
    
    void rotate(float dyaw, float dpitch) {
        yaw += dyaw;
        pitch += dpitch;
        
        // Constrain pitch
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
        
        updatePosition();
    }
    
    void zoom(float delta) {
        distance -= delta * 50.0f;
        if (distance < 100.0f) distance = 100.0f;
        if (distance > 3000.0f) distance = 3000.0f;
        updatePosition();
    }
    
    void pan(float dx, float dy) {
        glm::vec3 right = glm::normalize(glm::cross(position - target, up));
        glm::vec3 upDir = glm::normalize(glm::cross(right, position - target));
        
        target += right * dx * distance * 0.001f;
        target += upDir * dy * distance * 0.001f;
        
        updatePosition();
    }
    
    void updatePosition() {
        float x = distance * cos(glm::radians(pitch)) * cos(glm::radians(yaw));
        float y = distance * sin(glm::radians(pitch));
        float z = distance * cos(glm::radians(pitch)) * sin(glm::radians(yaw));
        position = target + glm::vec3(x, y, z);
    }
};

// Shader source code
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 objectColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float alpha;

void main() {
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, alpha);
}
)";

const char* lineVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
}
)";

const char* lineFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

uniform vec4 lineColor;

void main() {
    FragColor = lineColor;
}
)";

// Utility function to compile shaders
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
    }
    return shader;
}

GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

// Sphere mesh generation
class SphereMesh {
public:
    GLuint VAO, VBO, EBO;
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    
    SphereMesh(int sectors = 36, int stacks = 18) {
        generateSphere(sectors, stacks);
        setupMesh();
    }
    
    ~SphereMesh() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }
    
    void draw() const {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    
private:
    void generateSphere(int sectors, int stacks) {
        float radius = 1.0f;
        
        for (int i = 0; i <= stacks; ++i) {
            float stackAngle = M_PI / 2 - i * M_PI / stacks;
            float xy = radius * cosf(stackAngle);
            float z = radius * sinf(stackAngle);
            
            for (int j = 0; j <= sectors; ++j) {
                float sectorAngle = j * 2 * M_PI / sectors;
                float x = xy * cosf(sectorAngle);
                float y = xy * sinf(sectorAngle);
                
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
                
                // Normals (same as position for unit sphere)
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
            }
        }
        
        // Generate indices
        for (int i = 0; i < stacks; ++i) {
            int k1 = i * (sectors + 1);
            int k2 = k1 + sectors + 1;
            
            for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
                if (i != 0) {
                    indices.push_back(k1);
                    indices.push_back(k2);
                    indices.push_back(k1 + 1);
                }
                
                if (i != (stacks - 1)) {
                    indices.push_back(k1 + 1);
                    indices.push_back(k2);
                    indices.push_back(k2 + 1);
                }
            }
        }
    }
    
    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        
        glBindVertexArray(VAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), 
                     vertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                     indices.data(), GL_STATIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 
                            (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
    }
};

// Drawing functions for 3D visualization
class Renderer3D {
public:
    GLuint shaderProgram;
    GLuint lineShaderProgram;
    SphereMesh* sphereMesh;
    Camera3D camera;
    
    Renderer3D() {
        shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
        lineShaderProgram = createShaderProgram(lineVertexShaderSource, lineFragmentShaderSource);
        sphereMesh = new SphereMesh(36, 18);
    }
    
    ~Renderer3D() {
        delete sphereMesh;
        glDeleteProgram(shaderProgram);
        glDeleteProgram(lineShaderProgram);
    }
    
    void drawSphere(const glm::vec3& position, float radius, 
                   const glm::vec3& color, float alpha = 1.0f) {
        glUseProgram(shaderProgram);
        
        // Set up matrices
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::scale(model, glm::vec3(radius));
        
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix(800.0f, 600.0f);
        
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, 
                          glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, 
                          glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE,
                          glm::value_ptr(projection));
        
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, 
                    glm::value_ptr(color));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, 
                    glm::value_ptr(glm::vec3(400.0f, 1000.0f, 800.0f)));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1,
                    glm::value_ptr(camera.position));
        glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), alpha);
        
        if (alpha < 1.0f) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        
        sphereMesh->draw();
        
        if (alpha < 1.0f) {
            glDisable(GL_BLEND);
        }
    }
    
    void drawObstacles(const std::vector<ObstacleND>& obstacles) {
        for (const auto& obs : obstacles) {
            if (obs.center.size() >= 3) {
                glm::vec3 position(obs.center(0), obs.center(1), obs.center(2));
                drawSphere(position, obs.radius, glm::vec3(0.4f, 0.4f, 0.4f), 0.7f);
            }
        }
    }
    
    void drawObstacleMap(const ObstacleMap& obstacle_map) {
        drawObstacles(obstacle_map.getObstacles());
    }
    
    void drawNode(const PathNode& node, float radius, const glm::vec3& color) {
        if (node.position.size() >= 3) {
            glm::vec3 position(node.position(0), node.position(1), node.position(2));
            drawSphere(position, radius, color, 0.9f);
        }
    }
    
    void drawNodeCollisionRadius(const PathNode& node, bool in_collision) {
        if (node.position.size() >= 3) {
            glm::vec3 position(node.position(0), node.position(1), node.position(2));
            glm::vec3 color = in_collision ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
            drawSphere(position, node.radius, color, 0.15f);
        }
    }
    
    void drawTrajectorySegments(const Trajectory& trajectory, const glm::vec4& color) {
        if (trajectory.nodes.size() < 2) return;
        
        std::vector<float> vertices;
        for (const auto& node : trajectory.nodes) {
            if (node.position.size() >= 3) {
                vertices.push_back(node.position(0));
                vertices.push_back(node.position(1));
                vertices.push_back(node.position(2));
            }
        }
        
        GLuint VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), 
                     vertices.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glUseProgram(lineShaderProgram);
        
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix(800.0f, 600.0f);
        
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "view"), 1, GL_FALSE,
                          glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "projection"), 1, GL_FALSE,
                          glm::value_ptr(projection));
        glUniform4fv(glGetUniformLocation(lineShaderProgram, "lineColor"), 1,
                    glm::value_ptr(color));
        
        glLineWidth(2.0f);
        glDrawArrays(GL_LINE_STRIP, 0, vertices.size() / 3);
        
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
    
    void drawTrajectoryWithCollisionRadius(const Trajectory& trajectory,
                                          const std::vector<ObstacleND>& obstacles,
                                          bool show_collision_radius = true) {
        // Draw collision radii first (translucent)
        if (show_collision_radius) {
            for (const auto& node : trajectory.nodes) {
                bool in_collision = isNodeInCollision(node, obstacles);
                drawNodeCollisionRadius(node, in_collision);
            }
        }
        
        // Draw trajectory line
        drawTrajectorySegments(trajectory, glm::vec4(0.2f, 0.2f, 0.8f, 0.8f));
        
        // Draw nodes
        for (const auto& node : trajectory.nodes) {
            drawNode(node, 3.0f, glm::vec3(0.4f, 0.4f, 1.0f));
        }
        
        // Draw start and goal
        if (!trajectory.nodes.empty()) {
            drawNode(trajectory.nodes[trajectory.start_index], 8.0f, glm::vec3(0.0f, 1.0f, 0.0f));
            drawNode(trajectory.nodes[trajectory.goal_index], 8.0f, glm::vec3(1.0f, 0.0f, 0.0f));
        }
    }
    
    void drawGrid(float size = 800.0f, int divisions = 10) {
        std::vector<float> vertices;
        float step = size / divisions;
        
        // Grid lines parallel to X-axis
        for (int i = 0; i <= divisions; ++i) {
            float z = i * step;
            vertices.push_back(0.0f);
            vertices.push_back(0.0f);
            vertices.push_back(z);
            
            vertices.push_back(size);
            vertices.push_back(0.0f);
            vertices.push_back(z);
        }
        
        // Grid lines parallel to Z-axis
        for (int i = 0; i <= divisions; ++i) {
            float x = i * step;
            vertices.push_back(x);
            vertices.push_back(0.0f);
            vertices.push_back(0.0f);
            
            vertices.push_back(x);
            vertices.push_back(0.0f);
            vertices.push_back(size);
        }
        
        GLuint VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                     vertices.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glUseProgram(lineShaderProgram);
        
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix(800.0f, 600.0f);
        
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "view"), 1, GL_FALSE,
                          glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "projection"), 1, GL_FALSE,
                          glm::value_ptr(projection));
        glUniform4f(glGetUniformLocation(lineShaderProgram, "lineColor"), 
                   0.7f, 0.7f, 0.7f, 0.3f);
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glLineWidth(1.0f);
        glDrawArrays(GL_LINES, 0, vertices.size() / 3);
        glDisable(GL_BLEND);
        
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
};

/**
 * @brief Mouse state structure to track button presses and cursor position
 */
struct MouseState {
    bool leftPressed = false;
    bool rightPressed = false;
    bool middlePressed = false;
    double lastX = 0.0;
    double lastY = 0.0;
    bool firstMouse = true;
};

// Global mouse state
static MouseState g_mouseState;

/**
 * @brief Mouse button callback for handling button press/release events
 * 
 * This function tracks the state of left, right, and middle mouse buttons.
 * It also resets the firstMouse flag when a button is pressed to avoid
 * sudden camera jumps.
 * 
 * @param window GLFW window pointer
 * @param button Mouse button identifier (GLFW_MOUSE_BUTTON_*)
 * @param action Action type (GLFW_PRESS or GLFW_RELEASE)
 * @param mods Modifier keys (not currently used)
 */
inline void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_mouseState.leftPressed = true;
            g_mouseState.firstMouse = true;
            glfwGetCursorPos(window, &g_mouseState.lastX, &g_mouseState.lastY);
        } else if (action == GLFW_RELEASE) {
            g_mouseState.leftPressed = false;
        }
    }
    
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            g_mouseState.rightPressed = true;
            g_mouseState.firstMouse = true;
            glfwGetCursorPos(window, &g_mouseState.lastX, &g_mouseState.lastY);
        } else if (action == GLFW_RELEASE) {
            g_mouseState.rightPressed = false;
        }
    }
    
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            g_mouseState.middlePressed = true;
            g_mouseState.firstMouse = true;
            glfwGetCursorPos(window, &g_mouseState.lastX, &g_mouseState.lastY);
        } else if (action == GLFW_RELEASE) {
            g_mouseState.middlePressed = false;
        }
    }
}

/**
 * @brief Cursor position callback for handling mouse movement
 * 
 * This function handles camera rotation (left mouse) and panning (right/middle mouse).
 * It retrieves the Renderer3D instance from the window user pointer.
 * The firstMouse flag prevents sudden camera jumps when starting a drag operation.
 * 
 * Usage:
 *   - Left mouse drag: Rotate camera
 *   - Right/Middle mouse drag: Pan camera
 * 
 * @param window GLFW window pointer
 * @param xpos Current X position of cursor
 * @param ypos Current Y position of cursor
 */
inline void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    // Get renderer from window user pointer
    Renderer3D* renderer = static_cast<Renderer3D*>(glfwGetWindowUserPointer(window));
    
    // Handle first mouse movement to avoid jump
    if (g_mouseState.firstMouse) {
        g_mouseState.lastX = xpos;
        g_mouseState.lastY = ypos;
        g_mouseState.firstMouse = false;
        return;
    }
    
    // Calculate mouse offset
    double xoffset = xpos - g_mouseState.lastX;
    double yoffset = g_mouseState.lastY - ypos;  // Reversed since y-coordinates go from bottom to top
    
    g_mouseState.lastX = xpos;
    g_mouseState.lastY = ypos;
    
    // Apply camera transformations if renderer is available
    if (renderer) {
        if (g_mouseState.leftPressed) {
            // Rotate camera with left mouse button
            renderer->camera.rotate(static_cast<float>(xoffset) * 0.5f, 
                                   static_cast<float>(yoffset) * 0.5f);
        }
        else if (g_mouseState.middlePressed || g_mouseState.rightPressed) {
            // Pan camera with middle or right mouse button
            renderer->camera.pan(static_cast<float>(-xoffset), 
                               static_cast<float>(yoffset));
        }
    }
}

/**
 * @brief Scroll callback for handling mouse wheel zoom
 * 
 * This function zooms the camera in/out based on scroll wheel input.
 * It retrieves the Renderer3D instance from the window user pointer.
 * 
 * @param window GLFW window pointer
 * @param xoffset Horizontal scroll offset (not used)
 * @param yoffset Vertical scroll offset (zoom direction)
 */
inline void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    Renderer3D* renderer = static_cast<Renderer3D*>(glfwGetWindowUserPointer(window));
    if (renderer) {
        renderer->camera.zoom(static_cast<float>(yoffset));
    }
}

/**
 * @brief Framebuffer size callback for handling window resize
 * 
 * This function updates the OpenGL viewport when the window is resized.
 * 
 * @param window GLFW window pointer
 * @param width New framebuffer width
 * @param height New framebuffer height
 */
inline void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
