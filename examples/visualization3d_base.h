/**
 * @file visualization3d_base.h
 * @brief Shared 3D visualization primitives
 * 
 * Contains Vec3, Camera3D, and drawing functions used by both
 * visualize_collision3d.cpp and visualization3d_comparison.h
 */
#pragma once

#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Trajectory.h"
#include "ObstacleMap.h"

namespace viz3d {

// ============================================================================
// Constants
// ============================================================================

namespace constants {
    constexpr float DEFAULT_DISTANCE = 1500.0f;
    constexpr float ROTATION_SPEED = 0.01f;
    constexpr float ZOOM_SPEED = 50.0f;
    constexpr float AUTO_ROTATE_SPEED = 0.005f;
    
    constexpr float AMBIENT_INTENSITY = 0.35f;
    constexpr float DIFFUSE_INTENSITY = 0.55f;
    constexpr float SPECULAR_INTENSITY = 0.3f;
    constexpr float SPECULAR_POWER = 32.0f;
    
    constexpr unsigned int TITLE_FONT_SIZE = 18;
    constexpr unsigned int LABEL_FONT_SIZE = 13;
    constexpr unsigned int HINT_FONT_SIZE = 11;
    constexpr unsigned int LEGEND_FONT_SIZE = 12;
    constexpr unsigned int STATUS_FONT_SIZE = 12;
}

namespace colors {
    inline sf::Color bgTop() { return sf::Color(240, 248, 255); }
    inline sf::Color bgBottom() { return sf::Color(255, 255, 255); }
    inline sf::Color trajectory() { return sf::Color(50, 100, 200); }
    inline sf::Color start() { return sf::Color(44, 160, 44); }
    inline sf::Color goal() { return sf::Color(255, 127, 14); }
    inline sf::Color obstacle() { return sf::Color(100, 140, 180); }
    inline sf::Color grid() { return sf::Color(200, 210, 220); }
    inline sf::Color text() { return sf::Color(30, 30, 30); }
    inline sf::Color safe() { return sf::Color(100, 255, 100); }
    inline sf::Color near() { return sf::Color(255, 200, 50); }
    inline sf::Color collision() { return sf::Color(255, 80, 80); }
}

// ============================================================================
// 3D Math Structures
// ============================================================================

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
    float theta = 0.4f;
    float phi = 0.3f;
    float distance = constants::DEFAULT_DISTANCE;
    Vec3 center;
    float fov = 50.0f;
    
    void rotate(float dtheta, float dphi) {
        theta += dtheta;
        phi += dphi;
        phi = std::max(-1.5f, std::min(1.5f, phi));
    }
    
    void zoom(float dz) {
        distance = std::max(100.0f, distance + dz);
    }
    
    Vec3 getPosition() const {
        return {
            center.x + distance * std::cos(phi) * std::sin(theta),
            center.y + distance * std::sin(phi),
            center.z + distance * std::cos(phi) * std::cos(theta)
        };
    }
    
    sf::Vector2f project(const Vec3& point, float screenWidth, float screenHeight) const {
        Vec3 camPos = getPosition();
        Vec3 forward = (center - camPos).normalized();
        Vec3 right = Vec3::cross(forward, Vec3(0, 1, 0)).normalized();
        Vec3 up = Vec3::cross(right, forward).normalized();
        
        Vec3 rel = point - camPos;
        float camX = Vec3::dot(rel, right);
        float camY = Vec3::dot(rel, up);
        float camZ = Vec3::dot(rel, forward);
        
        if (camZ <= 0.1f) camZ = 0.1f;
        
        float fovRad = fov * 3.14159f / 180.0f;
        float scale = (screenHeight / 2.0f) / std::tan(fovRad / 2.0f);
        
        return {screenWidth / 2.0f + (camX / camZ) * scale,
                screenHeight / 2.0f - (camY / camZ) * scale};
    }
    
    float getDepth(const Vec3& point) const {
        Vec3 camPos = getPosition();
        return (point - camPos).length();
    }
    
    void setupFromBounds(const Vec3& minB, const Vec3& maxB) {
        center = (minB + maxB) * 0.5f;
        distance = (maxB - minB).length() * 0.9f;
        theta = 0.5f;
        phi = 0.4f;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

inline bool tryLoadFont(sf::Font& font) {
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

inline Vec3 getLightDirection() {
    return Vec3(0.5f, 0.8f, 0.3f).normalized();
}

inline float calculateSDF3D(const Vec3& pos, const std::vector<ObstacleND>& obstacles) {
    float min_sdf = std::numeric_limits<float>::max();
    for (const auto& obs : obstacles) {
        if (obs.dimensions() < 3) continue;
        float dx = pos.x - obs.center(0);
        float dy = pos.y - obs.center(1);
        float dz = pos.z - obs.center(2);
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        min_sdf = std::min(min_sdf, dist - obs.radius);
    }
    return min_sdf;
}

// ============================================================================
// Drawing Primitives
// ============================================================================

inline void drawGradientBackground(sf::RenderTarget& target, unsigned int w, unsigned int h) {
    sf::VertexArray gradient(sf::PrimitiveType::TriangleStrip, 4);
    gradient[0].position = {0.f, 0.f};
    gradient[0].color = colors::bgTop();
    gradient[1].position = {static_cast<float>(w), 0.f};
    gradient[1].color = colors::bgTop();
    gradient[2].position = {0.f, static_cast<float>(h)};
    gradient[2].color = colors::bgBottom();
    gradient[3].position = {static_cast<float>(w), static_cast<float>(h)};
    gradient[3].color = colors::bgBottom();
    target.draw(gradient);
}

inline void drawShadedSphere(sf::RenderTarget& target, sf::Vector2f screenPos, 
                             float screenRadius, sf::Color baseColor, 
                             const Vec3& viewDir, float depth_factor) {
    if (screenRadius < 2.0f) {
        sf::CircleShape dot(screenRadius);
        dot.setPosition({screenPos.x - screenRadius, screenPos.y - screenRadius});
        dot.setFillColor(baseColor);
        target.draw(dot);
        return;
    }
    
    int num_rings = std::min(static_cast<int>(screenRadius / 2), 15);
    num_rings = std::max(num_rings, 4);
    
    Vec3 lightDir = getLightDirection();
    float highlightOffsetX = -screenRadius * 0.3f;
    float highlightOffsetY = -screenRadius * 0.3f;
    
    for (int i = num_rings; i >= 0; --i) {
        float t = static_cast<float>(i) / num_rings;
        float r = screenRadius * t;
        
        float normalZ = std::sqrt(1.0f - t * t);
        Vec3 surfaceNormal = Vec3(-t * 0.5f, t * 0.3f, normalZ).normalized();
        
        float diffuse = std::max(0.0f, Vec3::dot(surfaceNormal, lightDir));
        float intensity = constants::AMBIENT_INTENSITY + diffuse * constants::DIFFUSE_INTENSITY;
        
        Vec3 halfVec = (lightDir + viewDir).normalized();
        float specular = std::pow(std::max(0.0f, Vec3::dot(surfaceNormal, halfVec)), 
                                  constants::SPECULAR_POWER);
        
        float rimDark = 1.0f - (1.0f - normalZ) * 0.4f;
        intensity *= rimDark;
        intensity *= 1.0f - (1.0f - depth_factor) * 0.2f;
        
        int cr = std::min(255, static_cast<int>(baseColor.r * intensity + specular * 60));
        int cg = std::min(255, static_cast<int>(baseColor.g * intensity + specular * 60));
        int cb = std::min(255, static_cast<int>(baseColor.b * intensity + specular * 60));
        
        sf::CircleShape ring(r);
        ring.setPosition({screenPos.x - r + highlightOffsetX * (1.0f - t) * 0.3f,
                         screenPos.y - r + highlightOffsetY * (1.0f - t) * 0.3f});
        ring.setFillColor(sf::Color(cr, cg, cb, baseColor.a));
        target.draw(ring);
    }
    
    float highlightRadius = screenRadius * 0.2f;
    sf::CircleShape highlight(highlightRadius);
    highlight.setPosition({screenPos.x + highlightOffsetX - highlightRadius,
                          screenPos.y + highlightOffsetY - highlightRadius});
    highlight.setFillColor(sf::Color(255, 255, 255, 80));
    target.draw(highlight);
}

inline void drawLine3D(sf::RenderTarget& target, const Camera3D& cam,
                       const Vec3& p1, const Vec3& p2, float width, sf::Color color,
                       unsigned int screenW, unsigned int screenH) {
    sf::Vector2f s1 = cam.project(p1, screenW, screenH);
    sf::Vector2f s2 = cam.project(p2, screenW, screenH);
    
    sf::Vector2f d = s2 - s1;
    float len = std::sqrt(d.x * d.x + d.y * d.y);
    if (len < 0.5f) return;
    
    sf::RectangleShape line({len, width});
    line.setPosition(s1);
    line.setFillColor(color);
    line.setRotation(sf::radians(std::atan2(d.y, d.x)));
    line.setOrigin({0, width / 2});
    target.draw(line);
}

inline void drawCircle3D(sf::RenderTarget& target, const Camera3D& cam,
                         const Vec3& center, float radius, sf::Color fillColor,
                         unsigned int screenW, unsigned int screenH, 
                         bool use_shading = true,
                         float minDepth = 0.0f, float maxDepth = 1.0f,
                         bool with_outline = false) {
    sf::Vector2f screenPos = cam.project(center, screenW, screenH);
    
    float depth = cam.getDepth(center);
    float fovRad = cam.fov * 3.14159f / 180.0f;
    float screenRadius = (radius / depth) * (screenH / 2.0f) / std::tan(fovRad / 2.0f);
    
    if (screenRadius < 1.0f) screenRadius = 1.0f;
    if (screenRadius > 500.0f) return;
    
    float depthRange = maxDepth - minDepth;
    float depth_factor = (depthRange > 0) ? 1.0f - (depth - minDepth) / depthRange : 1.0f;
    depth_factor = std::max(0.0f, std::min(1.0f, depth_factor));
    
    Vec3 camPos = cam.getPosition();
    Vec3 viewDir = (camPos - center).normalized();
    
    if (use_shading && screenRadius > 3.0f) {
        drawShadedSphere(target, screenPos, screenRadius, fillColor, viewDir, depth_factor);
    } else {
        sf::CircleShape circle(screenRadius);
        circle.setPosition({screenPos.x - screenRadius, screenPos.y - screenRadius});
        circle.setFillColor(fillColor);
        if (with_outline) {
            sf::Color outlineCol(fillColor.r * 0.6f, fillColor.g * 0.6f, fillColor.b * 0.6f, 255);
            circle.setOutlineColor(outlineCol);
            circle.setOutlineThickness(1.5f);
        }
        target.draw(circle);
    }
}

inline void drawGrid(sf::RenderTarget& target, const Camera3D& cam,
                     const Vec3& minB, const Vec3& maxB, int divisions,
                     unsigned int screenW, unsigned int screenH) {
    float y = minB.y;
    float stepX = (maxB.x - minB.x) / divisions;
    float stepZ = (maxB.z - minB.z) / divisions;
    
    // Ground plane
    sf::Vector2f corners[4];
    corners[0] = cam.project(Vec3(minB.x, y, minB.z), screenW, screenH);
    corners[1] = cam.project(Vec3(maxB.x, y, minB.z), screenW, screenH);
    corners[2] = cam.project(Vec3(maxB.x, y, maxB.z), screenW, screenH);
    corners[3] = cam.project(Vec3(minB.x, y, maxB.z), screenW, screenH);
    
    sf::VertexArray ground(sf::PrimitiveType::TriangleFan, 4);
    ground[0].position = corners[0]; ground[0].color = sf::Color(245, 248, 250, 180);
    ground[1].position = corners[1]; ground[1].color = sf::Color(245, 248, 250, 180);
    ground[2].position = corners[2]; ground[2].color = sf::Color(240, 245, 248, 150);
    ground[3].position = corners[3]; ground[3].color = sf::Color(240, 245, 248, 150);
    target.draw(ground);
    
    sf::Color gridColor = colors::grid();
    for (int i = 0; i <= divisions; ++i) {
        float x = minB.x + i * stepX;
        float z = minB.z + i * stepZ;
        float tx = static_cast<float>(i) / divisions;
        float edgeFade = 1.0f - std::abs(tx - 0.5f) * 0.5f;
        
        sf::Color lineColor = gridColor;
        lineColor.a = static_cast<std::uint8_t>(gridColor.a * edgeFade);
        float lineWidth = (i % 4 == 0) ? 1.2f : 0.6f;
        
        drawLine3D(target, cam, Vec3(x, y, minB.z), Vec3(x, y, maxB.z), lineWidth, lineColor, screenW, screenH);
        drawLine3D(target, cam, Vec3(minB.x, y, z), Vec3(maxB.x, y, z), lineWidth, lineColor, screenW, screenH);
    }
}

inline void drawAxes(sf::RenderTarget& target, const Camera3D& cam,
                     const Vec3& origin, float length,
                     unsigned int screenW, unsigned int screenH,
                     const sf::Font* font) {
    sf::Color xColor(220, 60, 60);
    sf::Color yColor(60, 180, 60);
    sf::Color zColor(60, 100, 220);
    
    auto drawAxisWithGlow = [&](const Vec3& end, sf::Color color) {
        sf::Color glowCol = color; glowCol.a = 40;
        drawLine3D(target, cam, origin, end, 4.0f, glowCol, screenW, screenH);
        drawLine3D(target, cam, origin, end, 2.0f, color, screenW, screenH);
    };
    
    drawAxisWithGlow(origin + Vec3(length, 0, 0), xColor);
    drawAxisWithGlow(origin + Vec3(0, length, 0), yColor);
    drawAxisWithGlow(origin + Vec3(0, 0, length), zColor);
    
    float arrowSize = length * 0.07f;
    drawCircle3D(target, cam, origin + Vec3(length, 0, 0), arrowSize, xColor, screenW, screenH);
    drawCircle3D(target, cam, origin + Vec3(0, length, 0), arrowSize, yColor, screenW, screenH);
    drawCircle3D(target, cam, origin + Vec3(0, 0, length), arrowSize, zColor, screenW, screenH);
    
    if (font) {
        auto drawLabel = [&](const Vec3& pos, const std::string& label, sf::Color color) {
            sf::Vector2f screenPos = cam.project(pos, screenW, screenH);
            sf::Text shadow(*font, label, constants::LABEL_FONT_SIZE);
            shadow.setFillColor(sf::Color(255, 255, 255, 180));
            shadow.setStyle(sf::Text::Bold);
            shadow.setPosition({screenPos.x + 1.0f, screenPos.y + 1.0f});
            target.draw(shadow);
            
            sf::Text text(*font, label, constants::LABEL_FONT_SIZE);
            text.setFillColor(color);
            text.setStyle(sf::Text::Bold);
            text.setPosition(screenPos);
            target.draw(text);
        };
        
        drawLabel(origin + Vec3(length * 1.12f, 0, 0), "X", xColor);
        drawLabel(origin + Vec3(0, length * 1.12f, 0), "Y", yColor);
        drawLabel(origin + Vec3(0, 0, length * 1.12f), "Z", zColor);
    }
}

// ============================================================================
// Bounds Computation
// ============================================================================

inline void computeBounds(Vec3& minB, Vec3& maxB, 
                          const std::vector<ObstacleND>& obstacles,
                          const Trajectory& trajectory) {
    minB = Vec3(1e10f, 1e10f, 1e10f);
    maxB = Vec3(-1e10f, -1e10f, -1e10f);
    
    for (const auto& node : trajectory.nodes) {
        if (node.position.size() >= 3) {
            minB.x = std::min(minB.x, node.position(0));
            minB.y = std::min(minB.y, node.position(1));
            minB.z = std::min(minB.z, node.position(2));
            maxB.x = std::max(maxB.x, node.position(0));
            maxB.y = std::max(maxB.y, node.position(1));
            maxB.z = std::max(maxB.z, node.position(2));
        }
    }
    for (const auto& obs : obstacles) {
        if (obs.dimensions() >= 3) {
            minB.x = std::min(minB.x, obs.center(0) - obs.radius);
            minB.y = std::min(minB.y, obs.center(1) - obs.radius);
            minB.z = std::min(minB.z, obs.center(2) - obs.radius);
            maxB.x = std::max(maxB.x, obs.center(0) + obs.radius);
            maxB.y = std::max(maxB.y, obs.center(1) + obs.radius);
            maxB.z = std::max(maxB.z, obs.center(2) + obs.radius);
        }
    }
    
    Vec3 range = maxB - minB;
    minB = minB - range * 0.1f;
    maxB = maxB + range * 0.1f;
}

} // namespace viz3d