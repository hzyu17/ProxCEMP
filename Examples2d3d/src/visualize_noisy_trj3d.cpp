#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "../include/PCEMotionPlanner.h"
#include "../examples/CollisionAvoidanceTask.h"
#include "../examples/visualization.h"

// --- Constants ---
namespace {
    constexpr unsigned int WINDOW_WIDTH = 1000;
    constexpr unsigned int WINDOW_HEIGHT = 800;
    constexpr float SAVE_SCALE = 4.0f;
    
    // Font sizes
    constexpr unsigned int TITLE_FONT_SIZE = 18;
    constexpr unsigned int LABEL_FONT_SIZE = 13;
    constexpr unsigned int HINT_FONT_SIZE = 11;
    constexpr unsigned int LEGEND_FONT_SIZE = 12;
    
    // 3D rendering parameters
    constexpr float DEFAULT_DISTANCE = 1500.0f;
    constexpr float ROTATION_SPEED = 0.01f;
    constexpr float ZOOM_SPEED = 50.0f;
    constexpr float AUTO_ROTATE_SPEED = 0.005f;
    
    // Lighting parameters
    constexpr float AMBIENT_INTENSITY = 0.3f;
    constexpr float DIFFUSE_INTENSITY = 0.6f;
    constexpr float SPECULAR_INTENSITY = 0.4f;
    constexpr float SPECULAR_POWER = 32.0f;
    constexpr float DEPTH_FOG_NEAR = 0.3f;
    constexpr float DEPTH_FOG_FAR = 1.0f;
    
    // Publication colors
    const sf::Color BG_COLOR_TOP(240, 248, 255);      // Light blue gradient top
    const sf::Color BG_COLOR_BOTTOM(255, 255, 255);   // White bottom
    const sf::Color SAMPLE_COLOR(31, 119, 180, 35);
    const sf::Color BASE_TRAJ_COLOR(214, 39, 40);
    const sf::Color START_COLOR(44, 160, 44);
    const sf::Color GOAL_COLOR(255, 127, 14);
    const sf::Color OBSTACLE_BASE_COLOR(100, 140, 180);  // Blue-gray for 3D look
    const sf::Color AXIS_COLOR(80, 80, 80);
    const sf::Color GRID_COLOR(200, 210, 220);
    const sf::Color TEXT_COLOR(30, 30, 30);
    const sf::Color SHADOW_COLOR(0, 0, 0, 40);
}

// --- 3D Math Structures ---
struct Vec3 {
    float x, y, z;
    
    Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const { 
        float len = length();
        return len > 0 ? Vec3(x/len, y/len, z/len) : Vec3(0,0,0);
    }
    
    static Vec3 cross(const Vec3& a, const Vec3& b) {
        return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
    }
    
    static float dot(const Vec3& a, const Vec3& b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }
};

struct Camera3D {
    float theta = 0.4f;      // Azimuth angle (rotation around Y axis)
    float phi = 0.3f;        // Elevation angle
    float distance = DEFAULT_DISTANCE;
    Vec3 center;             // Look-at point
    float fov = 50.0f;       // Field of view in degrees (narrower = larger appearance)
    
    // Rotate camera
    void rotate(float dtheta, float dphi) {
        theta += dtheta;
        phi += dphi;
        // Clamp phi to avoid gimbal lock
        phi = std::max(-1.5f, std::min(1.5f, phi));
    }
    
    void zoom(float dz) {
        distance = std::max(100.0f, distance + dz);
    }
    
    // Get camera position
    Vec3 getPosition() const {
        return {
            center.x + distance * std::cos(phi) * std::sin(theta),
            center.y + distance * std::sin(phi),
            center.z + distance * std::cos(phi) * std::cos(theta)
        };
    }
    
    // Project 3D point to 2D screen coordinates
    sf::Vector2f project(const Vec3& point, float screenWidth, float screenHeight) const {
        Vec3 camPos = getPosition();
        
        // View direction vectors
        Vec3 forward = (center - camPos).normalized();
        Vec3 right = Vec3::cross(forward, Vec3(0, 1, 0)).normalized();
        Vec3 up = Vec3::cross(right, forward).normalized();
        
        // Transform point to camera space
        Vec3 rel = point - camPos;
        float camX = Vec3::dot(rel, right);
        float camY = Vec3::dot(rel, up);
        float camZ = Vec3::dot(rel, forward);
        
        // Perspective projection
        if (camZ <= 0.1f) camZ = 0.1f;  // Clip near plane
        
        float fovRad = fov * 3.14159f / 180.0f;
        float scale = (screenHeight / 2.0f) / std::tan(fovRad / 2.0f);
        
        float screenX = screenWidth / 2.0f + (camX / camZ) * scale;
        float screenY = screenHeight / 2.0f - (camY / camZ) * scale;  // Flip Y
        
        return {screenX, screenY};
    }
    
    // Get depth for sorting (distance from camera)
    float getDepth(const Vec3& point) const {
        Vec3 camPos = getPosition();
        return (point - camPos).length();
    }
};

// --- Statistics Struct ---
struct NoiseStats3D {
    float avg_perturbation = 0.0f;
    float max_perturbation = 0.0f;
    size_t num_samples = 0;
    size_t num_nodes = 0;
};

// --- Drawable 3D Elements ---
struct Drawable3D {
    enum Type { LINE, SPHERE, POINT };
    Type type;
    Vec3 p1, p2;      // For lines: start/end; For spheres/points: p1 = center
    float radius;      // For spheres
    sf::Color color;
    float depth;       // For sorting
    
    static Drawable3D makeLine(const Vec3& start, const Vec3& end, sf::Color col) {
        Drawable3D d;
        d.type = LINE;
        d.p1 = start;
        d.p2 = end;
        d.color = col;
        d.depth = 0;
        return d;
    }
    
    static Drawable3D makeSphere(const Vec3& center, float r, sf::Color col) {
        Drawable3D d;
        d.type = SPHERE;
        d.p1 = center;
        d.radius = r;
        d.color = col;
        d.depth = 0;
        return d;
    }
    
    static Drawable3D makePoint(const Vec3& pos, float r, sf::Color col) {
        Drawable3D d;
        d.type = POINT;
        d.p1 = pos;
        d.radius = r;
        d.color = col;
        d.depth = 0;
        return d;
    }
};

// --- Helper Functions ---
namespace {

bool tryLoadFont(sf::Font& font) {
    const std::vector<std::string> paths = {
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
    };
    for (const auto& p : paths) {
        if (font.openFromFile(p)) return true;
    }
    return false;
}

// Light direction (normalized) - from upper-right-front
Vec3 getLightDirection() {
    return Vec3(0.5f, 0.8f, 0.3f).normalized();
}

// Calculate Phong lighting for a point on a sphere
sf::Color calculateLighting(const Vec3& surfaceNormal, const Vec3& viewDir, 
                            sf::Color baseColor, float depth_factor = 1.0f) {
    Vec3 lightDir = getLightDirection();
    
    // Ambient
    float ambient = AMBIENT_INTENSITY;
    
    // Diffuse (Lambert)
    float diffuse = std::max(0.0f, Vec3::dot(surfaceNormal, lightDir)) * DIFFUSE_INTENSITY;
    
    // Specular (Blinn-Phong)
    Vec3 halfVec = (lightDir + viewDir).normalized();
    float specular = std::pow(std::max(0.0f, Vec3::dot(surfaceNormal, halfVec)), SPECULAR_POWER) * SPECULAR_INTENSITY;
    
    // Combine lighting
    float intensity = std::min(1.0f, ambient + diffuse);
    
    // Apply depth fog (objects further away fade slightly)
    float fog = 1.0f - (1.0f - depth_factor) * 0.3f;
    
    // Calculate final color
    int r = std::min(255, static_cast<int>(baseColor.r * intensity * fog + specular * 255));
    int g = std::min(255, static_cast<int>(baseColor.g * intensity * fog + specular * 255));
    int b = std::min(255, static_cast<int>(baseColor.b * intensity * fog + specular * 255));
    
    return sf::Color(r, g, b, baseColor.a);
}

// Draw a gradient background
void drawGradientBackground(sf::RenderTarget& target, unsigned int width, unsigned int height) {
    sf::VertexArray gradient(sf::PrimitiveType::TriangleStrip, 4);
    
    gradient[0].position = {0.f, 0.f};
    gradient[0].color = BG_COLOR_TOP;
    
    gradient[1].position = {static_cast<float>(width), 0.f};
    gradient[1].color = BG_COLOR_TOP;
    
    gradient[2].position = {0.f, static_cast<float>(height)};
    gradient[2].color = BG_COLOR_BOTTOM;
    
    gradient[3].position = {static_cast<float>(width), static_cast<float>(height)};
    gradient[3].color = BG_COLOR_BOTTOM;
    
    target.draw(gradient);
}

// Draw a shaded 3D sphere using concentric circles
void drawShadedSphere(sf::RenderTarget& target, sf::Vector2f screenPos, float screenRadius,
                      sf::Color baseColor, const Vec3& viewDir, float depth_factor) {
    if (screenRadius < 2.0f) {
        // Too small, just draw a point
        sf::CircleShape dot(screenRadius);
        dot.setPosition({screenPos.x - screenRadius, screenPos.y - screenRadius});
        dot.setFillColor(baseColor);
        target.draw(dot);
        return;
    }
    
    // Draw sphere with radial gradient for 3D effect
    int num_rings = std::min(static_cast<int>(screenRadius / 2), 20);
    num_rings = std::max(num_rings, 5);
    
    Vec3 lightDir = getLightDirection();
    
    // Light offset for highlight (in screen space)
    float highlightOffsetX = -screenRadius * 0.3f;
    float highlightOffsetY = -screenRadius * 0.3f;
    
    for (int i = num_rings; i >= 0; --i) {
        float t = static_cast<float>(i) / num_rings;
        float r = screenRadius * t;
        
        // Calculate surface normal at this ring (approximation)
        // Center is pointing at viewer, edge is perpendicular
        float normalZ = std::sqrt(1.0f - t * t);  // z component of normal
        Vec3 surfaceNormal(0, 0, normalZ);
        
        // Adjust normal based on position for more realistic shading
        float nx = -t * 0.5f;  // tilt toward light
        float ny = t * 0.3f;
        surfaceNormal = Vec3(nx, ny, normalZ).normalized();
        
        // Calculate lighting for this ring
        float diffuse = std::max(0.0f, Vec3::dot(surfaceNormal, lightDir));
        float ambient = AMBIENT_INTENSITY;
        float intensity = ambient + diffuse * DIFFUSE_INTENSITY;
        
        // Add specular highlight near center
        Vec3 halfVec = (lightDir + viewDir).normalized();
        float specular = std::pow(std::max(0.0f, Vec3::dot(surfaceNormal, halfVec)), SPECULAR_POWER);
        
        // Darken edges (rim darkening)
        float rimDark = 1.0f - (1.0f - normalZ) * 0.4f;
        intensity *= rimDark;
        
        // Apply depth fog
        float fog = 1.0f - (1.0f - depth_factor) * 0.25f;
        intensity *= fog;
        
        int cr = std::min(255, static_cast<int>(baseColor.r * intensity + specular * 80));
        int cg = std::min(255, static_cast<int>(baseColor.g * intensity + specular * 80));
        int cb = std::min(255, static_cast<int>(baseColor.b * intensity + specular * 80));
        
        sf::CircleShape ring(r);
        ring.setPosition({screenPos.x - r + highlightOffsetX * (1.0f - t) * 0.3f,
                         screenPos.y - r + highlightOffsetY * (1.0f - t) * 0.3f});
        ring.setFillColor(sf::Color(cr, cg, cb, baseColor.a));
        target.draw(ring);
    }
    
    // Add specular highlight
    float highlightRadius = screenRadius * 0.25f;
    sf::CircleShape highlight(highlightRadius);
    highlight.setPosition({screenPos.x + highlightOffsetX - highlightRadius,
                          screenPos.y + highlightOffsetY - highlightRadius});
    highlight.setFillColor(sf::Color(255, 255, 255, 100));
    target.draw(highlight);
}

// Draw a shadow ellipse on the ground plane
void drawShadow(sf::RenderTarget& target, const Camera3D& cam,
                const Vec3& center, float radius, float groundY,
                unsigned int screenW, unsigned int screenH) {
    // Project shadow position (directly below object)
    Vec3 shadowPos(center.x, groundY, center.z);
    sf::Vector2f screenShadow = cam.project(shadowPos, screenW, screenH);
    
    // Shadow size based on height above ground and perspective
    float height = center.y - groundY;
    float shadowScale = 1.0f + height * 0.001f;  // Larger shadow when higher
    
    float depth = cam.getDepth(shadowPos);
    float fovRad = cam.fov * 3.14159f / 180.0f;
    float screenRadius = (radius * shadowScale / depth) * (screenH / 2.0f) / std::tan(fovRad / 2.0f);
    
    // Draw elliptical shadow (compressed in Y for perspective)
    sf::CircleShape shadow(screenRadius);
    shadow.setScale({1.0f, 0.3f});  // Flatten for ground plane
    shadow.setPosition({screenShadow.x - screenRadius, screenShadow.y - screenRadius * 0.3f});
    
    // Fade shadow based on height
    int alpha = std::max(0, static_cast<int>(60 - height * 0.05f));
    shadow.setFillColor(sf::Color(0, 0, 0, alpha));
    target.draw(shadow);
}

void computeBounds(const Trajectory& base_trajectory,
                   const std::vector<Trajectory>& noisy_samples,
                   Vec3& minBound, Vec3& maxBound) {
    minBound = Vec3(1e10f, 1e10f, 1e10f);
    maxBound = Vec3(-1e10f, -1e10f, -1e10f);
    
    auto updateBounds = [&](const Eigen::VectorXf& pos) {
        if (pos.size() >= 3) {
            minBound.x = std::min(minBound.x, pos(0));
            minBound.y = std::min(minBound.y, pos(1));
            minBound.z = std::min(minBound.z, pos(2));
            maxBound.x = std::max(maxBound.x, pos(0));
            maxBound.y = std::max(maxBound.y, pos(1));
            maxBound.z = std::max(maxBound.z, pos(2));
        }
    };
    
    for (const auto& node : base_trajectory.nodes) {
        updateBounds(node.position);
    }
    for (const auto& sample : noisy_samples) {
        for (const auto& node : sample.nodes) {
            updateBounds(node.position);
        }
    }
    
    // Add padding
    Vec3 range = maxBound - minBound;
    float padding = 0.1f;
    minBound = minBound - range * padding;
    maxBound = maxBound + range * padding;
}

void drawLine3D(sf::RenderTarget& target, const Camera3D& cam,
                const Vec3& p1, const Vec3& p2, float width, sf::Color color,
                unsigned int screenW, unsigned int screenH,
                bool depth_vary_width = false, float minDepth = 0, float maxDepth = 1) {
    sf::Vector2f s1 = cam.project(p1, screenW, screenH);
    sf::Vector2f s2 = cam.project(p2, screenW, screenH);
    
    sf::Vector2f d = s2 - s1;
    float len = std::sqrt(d.x * d.x + d.y * d.y);
    if (len < 0.5f) return;
    
    // Optionally vary width based on depth
    float finalWidth = width;
    if (depth_vary_width && maxDepth > minDepth) {
        float avgDepth = (cam.getDepth(p1) + cam.getDepth(p2)) * 0.5f;
        float depthFactor = 1.0f - (avgDepth - minDepth) / (maxDepth - minDepth);
        finalWidth = width * (0.5f + 0.5f * depthFactor);
    }
    
    sf::RectangleShape line({len, finalWidth});
    line.setPosition(s1);
    line.setFillColor(color);
    line.setRotation(sf::radians(std::atan2(d.y, d.x)));
    line.setOrigin({0, finalWidth / 2});
    target.draw(line);
}

void drawCircle3D(sf::RenderTarget& target, const Camera3D& cam,
                  const Vec3& center, float radius, sf::Color fillColor, sf::Color outlineColor,
                  unsigned int screenW, unsigned int screenH, bool use_shading = true,
                  float minDepth = 0.0f, float maxDepth = 1.0f) {
    sf::Vector2f screenPos = cam.project(center, screenW, screenH);
    
    // Approximate screen-space radius based on depth
    float depth = cam.getDepth(center);
    float fovRad = cam.fov * 3.14159f / 180.0f;
    float screenRadius = (radius / depth) * (screenH / 2.0f) / std::tan(fovRad / 2.0f);
    
    if (screenRadius < 1.0f) screenRadius = 1.0f;
    if (screenRadius > 500.0f) return;  // Too close, skip
    
    // Calculate depth factor for fog (0 = far, 1 = near)
    float depthRange = maxDepth - minDepth;
    float depth_factor = (depthRange > 0) ? 1.0f - (depth - minDepth) / depthRange : 1.0f;
    depth_factor = std::max(0.0f, std::min(1.0f, depth_factor));
    
    // View direction (approximation - from camera to object)
    Vec3 camPos = cam.getPosition();
    Vec3 viewDir = (camPos - center).normalized();
    
    if (use_shading && screenRadius > 3.0f) {
        drawShadedSphere(target, screenPos, screenRadius, fillColor, viewDir, depth_factor);
    } else {
        // Simple circle for small objects or when shading disabled
        sf::CircleShape circle(screenRadius);
        circle.setPosition({screenPos.x - screenRadius, screenPos.y - screenRadius});
        circle.setFillColor(fillColor);
        if (outlineColor.a > 0) {
            circle.setOutlineColor(outlineColor);
            circle.setOutlineThickness(1.0f);
        }
        target.draw(circle);
    }
}

void drawAxes(sf::RenderTarget& target, const Camera3D& cam,
              const Vec3& origin, float length,
              unsigned int screenW, unsigned int screenH,
              const sf::Font* font) {
    // Colors with gradients for 3D effect
    sf::Color xColor(220, 60, 60);
    sf::Color yColor(60, 180, 60);
    sf::Color zColor(60, 100, 220);
    
    // Draw axis lines with glow effect
    auto drawAxisWithGlow = [&](const Vec3& end, sf::Color color) {
        // Glow
        sf::Color glowCol = color;
        glowCol.a = 40;
        drawLine3D(target, cam, origin, end, 5.0f, glowCol, screenW, screenH);
        // Main line
        drawLine3D(target, cam, origin, end, 2.5f, color, screenW, screenH);
    };
    
    drawAxisWithGlow(origin + Vec3(length, 0, 0), xColor);
    drawAxisWithGlow(origin + Vec3(0, length, 0), yColor);
    drawAxisWithGlow(origin + Vec3(0, 0, length), zColor);
    
    // Draw arrow heads as small spheres at the end
    float arrowSize = length * 0.08f;
    drawCircle3D(target, cam, origin + Vec3(length, 0, 0), arrowSize, xColor, sf::Color::Transparent, screenW, screenH, true, 0, 1);
    drawCircle3D(target, cam, origin + Vec3(0, length, 0), arrowSize, yColor, sf::Color::Transparent, screenW, screenH, true, 0, 1);
    drawCircle3D(target, cam, origin + Vec3(0, 0, length), arrowSize, zColor, sf::Color::Transparent, screenW, screenH, true, 0, 1);
    
    // Draw origin sphere
    drawCircle3D(target, cam, origin, arrowSize * 0.8f, sf::Color(80, 80, 80), sf::Color::Transparent, screenW, screenH, true, 0, 1);
    
    if (font) {
        auto drawAxisLabel = [&](const Vec3& pos, const std::string& label, sf::Color color) {
            sf::Vector2f screenPos = cam.project(pos, screenW, screenH);
            
            // Shadow
            sf::Text shadow(*font, label, LABEL_FONT_SIZE + 1);
            shadow.setFillColor(sf::Color(255, 255, 255, 200));
            shadow.setStyle(sf::Text::Bold);
            shadow.setPosition({screenPos.x + 1.0f, screenPos.y + 1.0f});
            target.draw(shadow);
            
            sf::Text text(*font, label, LABEL_FONT_SIZE + 1);
            text.setFillColor(color);
            text.setStyle(sf::Text::Bold);
            text.setPosition(screenPos);
            target.draw(text);
        };
        
        drawAxisLabel(origin + Vec3(length * 1.15f, 0, 0), "X", xColor);
        drawAxisLabel(origin + Vec3(0, length * 1.15f, 0), "Y", yColor);
        drawAxisLabel(origin + Vec3(0, 0, length * 1.15f), "Z", zColor);
    }
}

void drawGrid(sf::RenderTarget& target, const Camera3D& cam,
              const Vec3& minB, const Vec3& maxB, int divisions,
              unsigned int screenW, unsigned int screenH) {
    // Draw grid on XZ plane at Y = minB.y
    float y = minB.y;
    float stepX = (maxB.x - minB.x) / divisions;
    float stepZ = (maxB.z - minB.z) / divisions;
    
    // Draw filled ground plane with subtle gradient
    std::vector<sf::Vertex> groundQuad;
    groundQuad.reserve(4);
    
    sf::Vector2f corners[4];
    corners[0] = cam.project(Vec3(minB.x, y, minB.z), screenW, screenH);
    corners[1] = cam.project(Vec3(maxB.x, y, minB.z), screenW, screenH);
    corners[2] = cam.project(Vec3(maxB.x, y, maxB.z), screenW, screenH);
    corners[3] = cam.project(Vec3(minB.x, y, maxB.z), screenW, screenH);
    
    sf::Color groundNear(245, 248, 250, 200);
    sf::Color groundFar(235, 240, 245, 150);
    
    sf::VertexArray ground(sf::PrimitiveType::TriangleFan, 4);
    ground[0].position = corners[0]; ground[0].color = groundFar;
    ground[1].position = corners[1]; ground[1].color = groundFar;
    ground[2].position = corners[2]; ground[2].color = groundNear;
    ground[3].position = corners[3]; ground[3].color = groundNear;
    target.draw(ground);
    
    // Draw grid lines with depth-based fading
    for (int i = 0; i <= divisions; ++i) {
        float x = minB.x + i * stepX;
        float z = minB.z + i * stepZ;
        
        // Calculate line opacity based on distance from center
        float tx = static_cast<float>(i) / divisions;
        float edgeFade = 1.0f - std::abs(tx - 0.5f) * 0.6f;
        
        sf::Color lineColor = GRID_COLOR;
        lineColor.a = static_cast<std::uint8_t>(lineColor.a * edgeFade);
        
        // Lines parallel to Z (vary thickness by depth)
        Vec3 p1(x, y, minB.z);
        Vec3 p2(x, y, maxB.z);
        float lineWidth = (i % 4 == 0) ? 1.5f : 0.8f;  // Thicker major lines
        drawLine3D(target, cam, p1, p2, lineWidth, lineColor, screenW, screenH);
        
        // Lines parallel to X
        p1 = Vec3(minB.x, y, z);
        p2 = Vec3(maxB.x, y, z);
        drawLine3D(target, cam, p1, p2, lineWidth, lineColor, screenW, screenH);
    }
}

void drawScene3D(sf::RenderTarget& target,
                 const std::vector<ObstacleND>& obstacles,
                 const Trajectory& base_trajectory,
                 const std::vector<Trajectory>& noisy_samples,
                 const Camera3D& cam,
                 const sf::Font* font,
                 bool show_legend,
                 bool show_axes,
                 bool show_grid,
                 const NoiseStats3D& stats,
                 unsigned int screenW, unsigned int screenH) {
    
    // Compute bounds
    Vec3 minB, maxB;
    computeBounds(base_trajectory, noisy_samples, minB, maxB);
    Vec3 center = (minB + maxB) * 0.5f;
    
    // Compute depth range for fog
    float minDepth = 1e10f, maxDepth = 0.0f;
    for (const auto& node : base_trajectory.nodes) {
        if (node.position.size() >= 3) {
            Vec3 pos(node.position(0), node.position(1), node.position(2));
            float d = cam.getDepth(pos);
            minDepth = std::min(minDepth, d);
            maxDepth = std::max(maxDepth, d);
        }
    }
    
    // Draw gradient background
    drawGradientBackground(target, screenW, screenH);
    
    // Draw grid first (background)
    if (show_grid) {
        drawGrid(target, cam, minB, maxB, 10, screenW, screenH);
    }
    
    // Collect all drawables for depth sorting
    std::vector<Drawable3D> drawables;
    
    // Add obstacles with nicer color
    for (const auto& obs : obstacles) {
        if (obs.dimensions() >= 3) {
            Vec3 pos(obs.center(0), obs.center(1), obs.center(2));
            auto d = Drawable3D::makeSphere(pos, obs.radius, OBSTACLE_BASE_COLOR);
            d.depth = cam.getDepth(pos);
            drawables.push_back(d);
        }
    }
    
    // Add noisy sample lines with depth-based alpha
    for (const auto& sample : noisy_samples) {
        for (size_t i = 0; i + 1 < sample.nodes.size(); ++i) {
            if (sample.nodes[i].position.size() >= 3) {
                Vec3 p1(sample.nodes[i].position(0), sample.nodes[i].position(1), sample.nodes[i].position(2));
                Vec3 p2(sample.nodes[i+1].position(0), sample.nodes[i+1].position(1), sample.nodes[i+1].position(2));
                
                // Vary alpha based on depth
                float avgDepth = cam.getDepth((p1 + p2) * 0.5f);
                float depthFactor = (maxDepth > minDepth) ? (avgDepth - minDepth) / (maxDepth - minDepth) : 0.5f;
                int alpha = static_cast<int>(25 + 20 * (1.0f - depthFactor));  // Brighter when closer
                
                sf::Color sampleCol = SAMPLE_COLOR;
                sampleCol.a = alpha;
                
                auto d = Drawable3D::makeLine(p1, p2, sampleCol);
                d.depth = avgDepth;
                drawables.push_back(d);
            }
        }
    }
    
    // Add base trajectory lines with glow effect
    for (size_t i = 0; i + 1 < base_trajectory.nodes.size(); ++i) {
        if (base_trajectory.nodes[i].position.size() >= 3) {
            Vec3 p1(base_trajectory.nodes[i].position(0), base_trajectory.nodes[i].position(1), base_trajectory.nodes[i].position(2));
            Vec3 p2(base_trajectory.nodes[i+1].position(0), base_trajectory.nodes[i+1].position(1), base_trajectory.nodes[i+1].position(2));
            auto d = Drawable3D::makeLine(p1, p2, BASE_TRAJ_COLOR);
            d.depth = cam.getDepth((p1 + p2) * 0.5f);
            drawables.push_back(d);
        }
    }
    
    // Add waypoints
    for (size_t i = 1; i + 1 < base_trajectory.nodes.size(); ++i) {
        if (base_trajectory.nodes[i].position.size() >= 3) {
            Vec3 pos(base_trajectory.nodes[i].position(0), 
                     base_trajectory.nodes[i].position(1), 
                     base_trajectory.nodes[i].position(2));
            auto d = Drawable3D::makePoint(pos, 5.0f, BASE_TRAJ_COLOR);
            d.depth = cam.getDepth(pos);
            drawables.push_back(d);
        }
    }
    
    // Add start/goal markers
    if (!base_trajectory.nodes.empty() && base_trajectory.nodes[0].position.size() >= 3) {
        Vec3 startPos(base_trajectory.nodes[base_trajectory.start_index].position(0),
                      base_trajectory.nodes[base_trajectory.start_index].position(1),
                      base_trajectory.nodes[base_trajectory.start_index].position(2));
        Vec3 goalPos(base_trajectory.nodes[base_trajectory.goal_index].position(0),
                     base_trajectory.nodes[base_trajectory.goal_index].position(1),
                     base_trajectory.nodes[base_trajectory.goal_index].position(2));
        
        auto ds = Drawable3D::makePoint(startPos, 15.0f, START_COLOR);
        ds.depth = cam.getDepth(startPos) - 0.1f;
        drawables.push_back(ds);
        
        auto dg = Drawable3D::makePoint(goalPos, 15.0f, GOAL_COLOR);
        dg.depth = cam.getDepth(goalPos) - 0.1f;
        drawables.push_back(dg);
    }
    
    // Sort by depth (far to near for correct occlusion)
    std::sort(drawables.begin(), drawables.end(), 
              [](const Drawable3D& a, const Drawable3D& b) { return a.depth > b.depth; });
    
    // Render all drawables
    for (const auto& d : drawables) {
        switch (d.type) {
            case Drawable3D::LINE: {
                // Draw line with slight glow for trajectory
                if (d.color.r > 200) {  // Red = trajectory line
                    // Glow effect
                    sf::Color glowCol = d.color;
                    glowCol.a = 50;
                    drawLine3D(target, cam, d.p1, d.p2, 4.0f, glowCol, screenW, screenH);
                }
                drawLine3D(target, cam, d.p1, d.p2, 2.0f, d.color, screenW, screenH);
                break;
            }
            case Drawable3D::SPHERE:
                drawCircle3D(target, cam, d.p1, d.radius, d.color, sf::Color::Transparent,
                            screenW, screenH, true, minDepth, maxDepth);
                break;
            case Drawable3D::POINT:
                drawCircle3D(target, cam, d.p1, d.radius, d.color, sf::Color(0, 0, 0, 80),
                            screenW, screenH, true, minDepth, maxDepth);
                break;
        }
    }
    
    // Draw axes
    if (show_axes) {
        float axisLen = std::max({maxB.x - minB.x, maxB.y - minB.y, maxB.z - minB.z}) * 0.18f;
        drawAxes(target, cam, minB, axisLen, screenW, screenH, font);
    }
    
    // Draw title with shadow
    if (font) {
        sf::Text titleShadow(*font, "3D Noise Distribution N(0, R^{-1})", TITLE_FONT_SIZE);
        titleShadow.setFillColor(sf::Color(255, 255, 255, 180));
        titleShadow.setStyle(sf::Text::Bold);
        titleShadow.setPosition({16.0f, 11.0f});
        target.draw(titleShadow);
        
        sf::Text title(*font, "3D Noise Distribution N(0, R^{-1})", TITLE_FONT_SIZE);
        title.setFillColor(TEXT_COLOR);
        title.setStyle(sf::Text::Bold);
        title.setPosition({15.0f, 10.0f});
        target.draw(title);
    }
    
    // Draw legend with improved styling
    if (font && show_legend) {
        float lx = screenW - 185.0f;
        float ly = 15.0f;
        
        // Shadow
        sf::RectangleShape legendShadow({170.0f, 130.0f});
        legendShadow.setPosition({lx + 3.0f, ly + 3.0f});
        legendShadow.setFillColor(sf::Color(0, 0, 0, 30));
        target.draw(legendShadow);
        
        // Background with rounded appearance (using gradient)
        sf::RectangleShape legendBg({170.0f, 130.0f});
        legendBg.setPosition({lx, ly});
        legendBg.setFillColor(sf::Color(255, 255, 255, 245));
        legendBg.setOutlineColor(sf::Color(180, 190, 200));
        legendBg.setOutlineThickness(1.0f);
        target.draw(legendBg);
        
        float itemY = ly + 12.0f;
        float itemH = 22.0f;
        
        auto drawLegendItem = [&](const std::string& label, sf::Color color, bool isLine) {
            if (isLine) {
                // Draw line with glow
                sf::RectangleShape glow({28.0f, 6.0f});
                glow.setPosition({lx + 8.0f, itemY + 5.0f});
                glow.setFillColor(sf::Color(color.r, color.g, color.b, 50));
                target.draw(glow);
                
                sf::RectangleShape line({26.0f, 3.0f});
                line.setPosition({lx + 9.0f, itemY + 6.5f});
                line.setFillColor(color);
                target.draw(line);
            } else {
                // Draw 3D-ish sphere marker
                sf::CircleShape marker(7.0f);
                marker.setPosition({lx + 14.0f, itemY + 2.0f});
                marker.setFillColor(color);
                target.draw(marker);
                
                sf::CircleShape highlight(2.5f);
                highlight.setPosition({lx + 16.0f, itemY + 4.0f});
                highlight.setFillColor(sf::Color(255, 255, 255, 120));
                target.draw(highlight);
            }
            
            sf::Text text(*font, label, LEGEND_FONT_SIZE);
            text.setFillColor(TEXT_COLOR);
            text.setPosition({lx + 42.0f, itemY});
            target.draw(text);
            
            itemY += itemH;
        };
        
        drawLegendItem("Noise samples", sf::Color(31, 119, 180, 200), true);
        drawLegendItem("Mean trajectory", BASE_TRAJ_COLOR, true);
        drawLegendItem("Start", START_COLOR, false);
        drawLegendItem("Goal", GOAL_COLOR, false);
        drawLegendItem("Obstacles", OBSTACLE_BASE_COLOR, false);
    }
    
    // Draw stats with improved styling
    if (font && show_legend) {
        float sx = 15.0f;
        float sy = screenH - 80.0f;
        
        // Shadow
        sf::RectangleShape statsShadow({225.0f, 65.0f});
        statsShadow.setPosition({sx + 3.0f, sy + 3.0f});
        statsShadow.setFillColor(sf::Color(0, 0, 0, 25));
        target.draw(statsShadow);
        
        sf::RectangleShape statsBg({225.0f, 65.0f});
        statsBg.setPosition({sx, sy});
        statsBg.setFillColor(sf::Color(248, 250, 255, 248));
        statsBg.setOutlineColor(sf::Color(180, 190, 200));
        statsBg.setOutlineThickness(1.0f);
        target.draw(statsBg);
        
        char buf[128];
        snprintf(buf, sizeof(buf), "Samples: %zu  |  Nodes: %zu", stats.num_samples, stats.num_nodes);
        sf::Text t1(*font, buf, 11);
        t1.setFillColor(TEXT_COLOR);
        t1.setPosition({sx + 12.0f, sy + 12.0f});
        target.draw(t1);
        
        snprintf(buf, sizeof(buf), "Avg perturbation: %.2f", stats.avg_perturbation);
        sf::Text t2(*font, buf, 11);
        t2.setFillColor(TEXT_COLOR);
        t2.setPosition({sx + 12.0f, sy + 32.0f});
        target.draw(t2);
        
        snprintf(buf, sizeof(buf), "Max perturbation: %.2f", stats.max_perturbation);
        sf::Text t3(*font, buf, 11);
        t3.setFillColor(TEXT_COLOR);
        t3.setPosition({sx + 12.0f, sy + 50.0f});
        target.draw(t3);
    }
}

bool saveImage3D(const std::vector<ObstacleND>& obstacles,
                 const Trajectory& base_trajectory,
                 const std::vector<Trajectory>& noisy_samples,
                 const Camera3D& cam,
                 const sf::Font* font,
                 bool show_legend,
                 bool show_axes,
                 bool show_grid,
                 const NoiseStats3D& stats,
                 int& save_counter,
                 float scale) {
    unsigned int w = static_cast<unsigned int>(WINDOW_WIDTH * scale);
    unsigned int h = static_cast<unsigned int>(WINDOW_HEIGHT * scale);
    
    sf::RenderTexture rt;
    if (!rt.resize({w, h})) {
        std::cerr << "Failed to create render texture!\n";
        return false;
    }
    
    rt.clear(sf::Color::White);
    
    sf::View view(sf::FloatRect({0.f, 0.f}, {(float)WINDOW_WIDTH, (float)WINDOW_HEIGHT}));
    view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
    rt.setView(view);
    
    drawScene3D(rt, obstacles, base_trajectory, noisy_samples, cam, font,
                show_legend, show_axes, show_grid, stats, WINDOW_WIDTH, WINDOW_HEIGHT);
    
    rt.display();
    
    std::string suffix = (scale > 1.0f) ? "_highres" : "";
    std::string filename = "noise_3d_" + std::to_string(save_counter++) + suffix + ".png";
    
    if (rt.getTexture().copyToImage().saveToFile(filename)) {
        std::cout << "Saved: " << filename << " (" << w << "x" << h << ")\n";
        return true;
    }
    return false;
}

} // anonymous namespace

/**
 * @brief Interactive 3D noise visualization with rotation controls
 */
void visualizeNoise3D(const std::vector<ObstacleND>& obstacles,
                      const Trajectory& base_trajectory, 
                      const std::vector<Trajectory>& noisy_samples,
                      const NoiseStats3D& stats = NoiseStats3D()) {
    
    sf::RenderWindow window(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}), 
                           "3D Noise Visualization (Drag to rotate)", 
                           sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    sf::Font font;
    bool font_loaded = tryLoadFont(font);

    // Compute bounds and center camera
    Vec3 minB, maxB;
    computeBounds(base_trajectory, noisy_samples, minB, maxB);
    
    Camera3D cam;
    cam.center = (minB + maxB) * 0.5f;
    cam.distance = (maxB - minB).length() * 0.9f;  // Closer for larger view
    cam.theta = 0.5f;
    cam.phi = 0.4f;
    cam.fov = 50.0f;  // Narrower FOV for less distortion and larger appearance

    std::cout << "\n=== 3D Visualization Controls ===\n"
              << "Mouse drag:     Rotate view\n"
              << "Scroll wheel:   Zoom in/out\n"
              << "+/-:            Zoom in/out\n"
              << "Arrow keys:     Rotate view\n"
              << "R:              Reset view\n"
              << "A:              Toggle auto-rotate\n"
              << "X:              Toggle axes\n"
              << "G:              Toggle grid\n"
              << "L:              Toggle legend\n"
              << "S:              Save PNG (1x)\n"
              << "P:              Save PNG (4x high-res)\n"
              << "ESC:            Exit\n"
              << "=================================\n\n";

    bool show_legend = true;
    bool show_axes = true;
    bool show_grid = true;
    bool auto_rotate = false;
    bool dragging = false;
    sf::Vector2i lastMouse;
    int save_counter = 0;

    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            
            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                switch (key->code) {
                    case sf::Keyboard::Key::Escape: window.close(); break;
                    case sf::Keyboard::Key::L: show_legend = !show_legend; break;
                    case sf::Keyboard::Key::X: show_axes = !show_axes; break;
                    case sf::Keyboard::Key::G: show_grid = !show_grid; break;
                    case sf::Keyboard::Key::A: auto_rotate = !auto_rotate; break;
                    case sf::Keyboard::Key::R:
                        cam.theta = 0.5f;
                        cam.phi = 0.4f;
                        cam.distance = (maxB - minB).length() * 0.9f;
                        cam.fov = 50.0f;
                        break;
                    case sf::Keyboard::Key::Left: cam.rotate(-0.1f, 0); break;
                    case sf::Keyboard::Key::Right: cam.rotate(0.1f, 0); break;
                    case sf::Keyboard::Key::Up: cam.rotate(0, 0.1f); break;
                    case sf::Keyboard::Key::Down: cam.rotate(0, -0.1f); break;
                    case sf::Keyboard::Key::Equal: cam.zoom(-ZOOM_SPEED); break;  // + key (zoom in)
                    case sf::Keyboard::Key::Hyphen: cam.zoom(ZOOM_SPEED); break;  // - key (zoom out)
                    case sf::Keyboard::Key::S:
                        saveImage3D(obstacles, base_trajectory, noisy_samples, cam,
                                   font_loaded ? &font : nullptr, show_legend, show_axes, show_grid,
                                   stats, save_counter, 1.0f);
                        break;
                    case sf::Keyboard::Key::P:
                        saveImage3D(obstacles, base_trajectory, noisy_samples, cam,
                                   font_loaded ? &font : nullptr, show_legend, show_axes, show_grid,
                                   stats, save_counter, SAVE_SCALE);
                        break;
                    default: break;
                }
            }
            
            if (const auto* mouseBtn = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseBtn->button == sf::Mouse::Button::Left) {
                    dragging = true;
                    lastMouse = {mouseBtn->position.x, mouseBtn->position.y};
                }
            }
            
            if (event->is<sf::Event::MouseButtonReleased>()) {
                dragging = false;
            }
            
            if (const auto* mouseMove = event->getIf<sf::Event::MouseMoved>()) {
                if (dragging) {
                    int dx = mouseMove->position.x - lastMouse.x;
                    int dy = mouseMove->position.y - lastMouse.y;
                    cam.rotate(dx * ROTATION_SPEED, -dy * ROTATION_SPEED);
                    lastMouse = {mouseMove->position.x, mouseMove->position.y};
                }
            }
            
            if (const auto* scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
                cam.zoom(-scroll->delta * ZOOM_SPEED);
            }
        }
        
        // Auto-rotate
        if (auto_rotate) {
            cam.rotate(AUTO_ROTATE_SPEED, 0);
        }
        
        window.clear(sf::Color::White);
        drawScene3D(window, obstacles, base_trajectory, noisy_samples, cam,
                   font_loaded ? &font : nullptr, show_legend, show_axes, show_grid,
                   stats, WINDOW_WIDTH, WINDOW_HEIGHT);
        
        // Keyboard hint
        if (font_loaded) {
            std::string hint = auto_rotate ? "[Auto-rotating] " : "";
            hint += "Drag: Rotate | Scroll/+/-: Zoom | S/P: Save | ESC: Exit";
            sf::Text hintText(font, hint, HINT_FONT_SIZE);
            hintText.setFillColor(sf::Color(100, 100, 100));
            hintText.setPosition({WINDOW_WIDTH - 450.0f, WINDOW_HEIGHT - 22.0f});
            window.draw(hintText);
        }

        window.display();
    }
}

int main() {
    std::cout << "==========================================\n"
              << "  3D Trajectory Noise Visualization\n"
              << "  (Interactive 3D View)\n"
              << "==========================================\n\n";

    // Use source file location to find config (same as main.cpp)
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../Examples2d3d/configs/config_3d.yaml";
    
    std::string config_file;
    try {
        config_file = std::filesystem::canonical(config_path).string();
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: Config file not found at: " << config_path << "\n";
        std::cerr << "Please ensure config_3d.yaml exists in the configs/ directory.\n";
        return 1;
    }
    
    YAML::Node config;
    
    try {
        config = YAML::LoadFile(config_file);
        std::cout << "Loaded: " << config_file << "\n";
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    size_t num_dims = config["motion_planning"]["num_dimensions"].as<size_t>(2);
    if (num_dims != 3) {
        std::cerr << "Error: Expected 3D configuration (num_dimensions=3)\n";
        return 1;
    }

    std::cout << "\n=== Creating 3D Task ===\n";
    auto task = std::make_shared<pce::CollisionAvoidanceTask>(config);
    std::cout << "Obstacles: " << task->getObstacleMap()->size() << "\n";

    std::cout << "\n=== Initializing Planner ===\n";
    PCEConfig pce_config;
    pce_config.loadFromFile(config_file);
    
    ProximalCrossEntropyMotionPlanner planner(task);
    if (!planner.initialize(pce_config)) {
        std::cerr << "Planner initialization failed\n";
        return 1;
    }
    
    const Trajectory& traj = planner.getCurrentTrajectory();
    std::cout << "Trajectory: " << traj.nodes.size() << " nodes, " << traj.dimensions() << "D\n";
    
    if (traj.dimensions() < 3) {
        std::cerr << "Error: Trajectory is not 3D\n";
        return 1;
    }

    const int num_samples = pce_config.num_samples;
    std::cout << "\nGenerating " << num_samples << " noise samples...\n";
    
    auto epsilon = planner.sampleNoiseMatrices(num_samples, traj.nodes.size(), traj.dimensions());
    
    Eigen::MatrixXf Y_base(traj.dimensions(), traj.nodes.size());
    for (size_t i = 0; i < traj.nodes.size(); ++i) {
        Y_base.col(i) = traj.nodes[i].position;
    }
    
    std::vector<Trajectory> noisy_samples;
    noisy_samples.reserve(num_samples);
    
    for (int m = 0; m < num_samples; ++m) {
        Trajectory perturbed = traj;
        Eigen::MatrixXf Y_perturbed = Y_base + epsilon[m];
        for (size_t i = 0; i < traj.nodes.size(); ++i) {
            perturbed.nodes[i].position = Y_perturbed.col(i);
        }
        noisy_samples.push_back(std::move(perturbed));
    }

    // Compute stats
    NoiseStats3D stats;
    stats.num_samples = num_samples;
    stats.num_nodes = traj.nodes.size();
    
    float total = 0.0f;
    for (const auto& sample : noisy_samples) {
        for (size_t i = 0; i < traj.nodes.size(); ++i) {
            float p = (sample.nodes[i].position - traj.nodes[i].position).norm();
            total += p;
            stats.max_perturbation = std::max(stats.max_perturbation, p);
        }
    }
    stats.avg_perturbation = total / (num_samples * traj.nodes.size());
    
    std::cout << "Avg perturbation: " << stats.avg_perturbation << "\n";
    std::cout << "Max perturbation: " << stats.max_perturbation << "\n\n";

    visualizeNoise3D(task->getObstacles(), traj, noisy_samples, stats);

    std::cout << "Visualization closed.\n";
    return 0;
}