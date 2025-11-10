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


/**
 * @brief Visualizes initial trajectory state in 3D
 * @param obstacles List of obstacles in the environment
 * @param trajectory Initial trajectory to visualize
 * @param window_title Title for the visualization window
 */
inline void visualizeInitialState3D(const std::vector<ObstacleND>& obstacles,
                                   const Trajectory& trajectory,
                                   const std::string& window_title) {
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    
    GLFWwindow* window = glfwCreateWindow(1200, 900, window_title.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        return;
    }
    
    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    
    // Create renderer
    Renderer3D renderer;
    glfwSetWindowUserPointer(window, &renderer);
    
    // Adjust camera to center of workspace
    renderer.camera.target = glm::vec3(400.0f, 300.0f, 400.0f);
    renderer.camera.distance = 1200.0f;
    renderer.camera.yaw = -45.0f;
    renderer.camera.pitch = 30.0f;
    renderer.camera.rotate(0.0f, 0.0f);  // Update position
    
    std::cout << "\n=== 3D Controls ===\n";
    std::cout << "Left Mouse Drag:  Rotate camera\n";
    std::cout << "Right Mouse Drag: Pan camera\n";
    std::cout << "Mouse Scroll:     Zoom in/out\n";
    std::cout << "G:                Toggle grid\n";
    std::cout << "SPACE/ENTER:      Continue to optimization\n";
    std::cout << "ESC:              Exit\n";
    std::cout << "===================\n\n";
    
    bool showGrid = true;
    bool gKeyWasPressed = false;
    bool shouldContinue = false;
    
    // Main render loop
    while (!glfwWindowShouldClose(window) && !shouldContinue) {
        // Process keyboard input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS || 
            glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
            shouldContinue = true;
            std::cout << "Continuing to optimization...\n";
        }
        
        // Toggle grid with G key (with debouncing)
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            if (!gKeyWasPressed) {
                showGrid = !showGrid;
                std::cout << "Grid: " << (showGrid ? "ON" : "OFF") << "\n";
                gKeyWasPressed = true;
            }
        } else {
            gKeyWasPressed = false;
        }
        
        // Clear screen
        glClearColor(0.94f, 0.94f, 0.94f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // 1. Draw grid (optional)
        if (showGrid) {
            renderer.drawGrid(800.0f, 10);
        }
        
        // 2. Draw obstacles
        renderer.drawObstacles(obstacles);
        
        // 3. Draw initial trajectory (blue/cyan)
        renderer.drawTrajectorySegments(trajectory, glm::vec4(0.2f, 0.6f, 1.0f, 0.8f));
        
        // 4. Draw nodes
        for (const auto& node : trajectory.nodes) {
            renderer.drawNode(node, 3.0f, glm::vec3(0.3f, 0.7f, 1.0f));
        }
        
        // 5. Draw Start/Goal
        if (!trajectory.nodes.empty()) {
            renderer.drawNode(trajectory.nodes[trajectory.start_index],
                            10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
            renderer.drawNode(trajectory.nodes[trajectory.goal_index],
                            10.0f, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwTerminate();
}

/**
 * @brief Opens an interactive window to display 3D trajectory history
 * @param obstacles List of obstacles in the environment
 * @param trajectory_history Complete history of trajectories through optimization
 * @param window_title Title for the visualization window
 * @param width Window width in pixels
 * @param height Window height in pixels
 */
inline void showTrajectoryHistoryWindow3D(
    const std::vector<ObstacleND>& obstacles,
    const std::vector<Trajectory>& trajectory_history,
    const std::string& window_title = "3D Optimization History",
    unsigned int width = 1200,
    unsigned int height = 900)
{
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    
    GLFWwindow* window = glfwCreateWindow(width, height, window_title.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        return;
    }
    
    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    
    // Create renderer
    Renderer3D renderer;
    glfwSetWindowUserPointer(window, &renderer);
    
    // Setup camera
    renderer.camera.target = glm::vec3(400.0f, 300.0f, 400.0f);
    renderer.camera.distance = 1200.0f;
    renderer.camera.yaw = -45.0f;
    renderer.camera.pitch = 30.0f;
    renderer.camera.rotate(0.0f, 0.0f);
    
    size_t current_iteration = trajectory_history.empty() ? 0 : trajectory_history.size() - 1;
    bool showGrid = true;
    bool showHistory = true;
    bool gKeyWasPressed = false;
    bool hKeyWasPressed = false;
    
    std::cout << "\n=== 3D Controls ===\n";
    std::cout << "Left Mouse Drag:  Rotate camera\n";
    std::cout << "Right Mouse Drag: Pan camera\n";
    std::cout << "Mouse Scroll:     Zoom in/out\n";
    std::cout << "Left/Right Arrow: Navigate trajectory history\n";
    std::cout << "G:                Toggle grid\n";
    std::cout << "H:                Toggle history\n";
    std::cout << "ESC:              Close window\n";
    std::cout << "===================\n\n";
    
    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Keyboard input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        // Navigate history
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS && !trajectory_history.empty() && current_iteration > 0) {
            current_iteration--;
            std::cout << "Iteration: " << current_iteration << "/" << (trajectory_history.size() - 1) << "\n";
            glfwWaitEventsTimeout(0.1);  // Debounce
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS && !trajectory_history.empty() && current_iteration < trajectory_history.size() - 1) {
            current_iteration++;
            std::cout << "Iteration: " << current_iteration << "/" << (trajectory_history.size() - 1) << "\n";
            glfwWaitEventsTimeout(0.1);  // Debounce
        }
        
        // Toggle grid
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            if (!gKeyWasPressed) {
                showGrid = !showGrid;
                std::cout << "Grid: " << (showGrid ? "ON" : "OFF") << "\n";
                gKeyWasPressed = true;
            }
        } else {
            gKeyWasPressed = false;
        }
        
        // Toggle history
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
            if (!hKeyWasPressed) {
                showHistory = !showHistory;
                std::cout << "History: " << (showHistory ? "ON" : "OFF") << "\n";
                hKeyWasPressed = true;
            }
        } else {
            hKeyWasPressed = false;
        }
        
        // Clear screen
        glClearColor(0.94f, 0.94f, 0.94f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Draw scene
        if (showGrid) {
            renderer.drawGrid(800.0f, 10);
        }
        
        renderer.drawObstacles(obstacles);
        
        // Draw trajectory history (semi-transparent)
        if (showHistory && !trajectory_history.empty()) {
            for (size_t i = 0; i < current_iteration; ++i) {
                float alpha = 0.1f + 0.3f * (float(i) / std::max(1.0f, float(current_iteration)));
                renderer.drawTrajectorySegments(trajectory_history[i], 
                    glm::vec4(0.5f, 0.5f, 0.5f, alpha));
            }
        }
        
        // Draw current trajectory
        if (!trajectory_history.empty()) {
            const Trajectory& current_traj = trajectory_history[current_iteration];
            renderer.drawTrajectorySegments(current_traj, glm::vec4(0.0f, 0.8f, 0.2f, 0.9f));
            
            // Draw nodes
            for (const auto& node : current_traj.nodes) {
                renderer.drawNode(node, 3.0f, glm::vec3(0.2f, 0.9f, 0.3f));
            }
            
            // Draw start/goal
            if (!current_traj.nodes.empty()) {
                renderer.drawNode(current_traj.nodes[current_traj.start_index],
                                10.0f, glm::vec3(0.0f, 1.0f, 0.0f));
                renderer.drawNode(current_traj.nodes[current_traj.goal_index],
                                10.0f, glm::vec3(1.0f, 0.0f, 0.0f));
            }
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwTerminate();
}