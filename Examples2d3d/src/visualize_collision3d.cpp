#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>

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
    constexpr unsigned int STATUS_FONT_SIZE = 12;
    
    // 3D rendering parameters
    constexpr float DEFAULT_DISTANCE = 1500.0f;
    constexpr float ROTATION_SPEED = 0.01f;
    constexpr float ZOOM_SPEED = 50.0f;
    constexpr float AUTO_ROTATE_SPEED = 0.005f;
    
    // Lighting parameters
    constexpr float AMBIENT_INTENSITY = 0.35f;
    constexpr float DIFFUSE_INTENSITY = 0.55f;
    constexpr float SPECULAR_INTENSITY = 0.3f;
    constexpr float SPECULAR_POWER = 32.0f;
    
    // Colors
    const sf::Color BG_COLOR_TOP(240, 248, 255);
    const sf::Color BG_COLOR_BOTTOM(255, 255, 255);
    const sf::Color TRAJECTORY_COLOR(50, 100, 200);
    const sf::Color START_COLOR(44, 160, 44);
    const sf::Color GOAL_COLOR(255, 127, 14);
    const sf::Color OBSTACLE_BASE_COLOR(100, 140, 180);
    const sf::Color GRID_COLOR(200, 210, 220);
    const sf::Color TEXT_COLOR(30, 30, 30);
    
    // Collision status colors - bright and visible
    const sf::Color SAFE_COLOR(100, 255, 100);       // Bright green
    const sf::Color NEAR_COLOR(255, 200, 50);        // Bright yellow-orange
    const sf::Color COLLISION_COLOR(255, 80, 80);    // Bright red
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
    float theta = 0.4f;
    float phi = 0.3f;
    float distance = DEFAULT_DISTANCE;
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

Vec3 getLightDirection() {
    return Vec3(0.5f, 0.8f, 0.3f).normalized();
}

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

void drawShadedSphere(sf::RenderTarget& target, sf::Vector2f screenPos, float screenRadius,
                      sf::Color baseColor, const Vec3& viewDir, float depth_factor) {
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
        float intensity = AMBIENT_INTENSITY + diffuse * DIFFUSE_INTENSITY;
        
        Vec3 halfVec = (lightDir + viewDir).normalized();
        float specular = std::pow(std::max(0.0f, Vec3::dot(surfaceNormal, halfVec)), SPECULAR_POWER);
        
        float rimDark = 1.0f - (1.0f - normalZ) * 0.4f;
        intensity *= rimDark;
        
        float fog = 1.0f - (1.0f - depth_factor) * 0.2f;
        intensity *= fog;
        
        int cr = std::min(255, static_cast<int>(baseColor.r * intensity + specular * 60));
        int cg = std::min(255, static_cast<int>(baseColor.g * intensity + specular * 60));
        int cb = std::min(255, static_cast<int>(baseColor.b * intensity + specular * 60));
        
        sf::CircleShape ring(r);
        ring.setPosition({screenPos.x - r + highlightOffsetX * (1.0f - t) * 0.3f,
                         screenPos.y - r + highlightOffsetY * (1.0f - t) * 0.3f});
        ring.setFillColor(sf::Color(cr, cg, cb, baseColor.a));
        target.draw(ring);
    }
    
    // Specular highlight
    float highlightRadius = screenRadius * 0.2f;
    sf::CircleShape highlight(highlightRadius);
    highlight.setPosition({screenPos.x + highlightOffsetX - highlightRadius,
                          screenPos.y + highlightOffsetY - highlightRadius});
    highlight.setFillColor(sf::Color(255, 255, 255, 80));
    target.draw(highlight);
}

void drawLine3D(sf::RenderTarget& target, const Camera3D& cam,
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

void drawCircle3D(sf::RenderTarget& target, const Camera3D& cam,
                  const Vec3& center, float radius, sf::Color fillColor,
                  unsigned int screenW, unsigned int screenH, bool use_shading = true,
                  float minDepth = 0.0f, float maxDepth = 1.0f, bool with_outline = false) {
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
            // Darker outline for contrast
            sf::Color outlineCol(fillColor.r * 0.6f, fillColor.g * 0.6f, fillColor.b * 0.6f, 255);
            circle.setOutlineColor(outlineCol);
            circle.setOutlineThickness(1.5f);
        }
        target.draw(circle);
    }
}

void drawGrid(sf::RenderTarget& target, const Camera3D& cam,
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
    
    for (int i = 0; i <= divisions; ++i) {
        float x = minB.x + i * stepX;
        float z = minB.z + i * stepZ;
        
        float tx = static_cast<float>(i) / divisions;
        float edgeFade = 1.0f - std::abs(tx - 0.5f) * 0.5f;
        
        sf::Color lineColor = GRID_COLOR;
        lineColor.a = static_cast<std::uint8_t>(lineColor.a * edgeFade);
        
        float lineWidth = (i % 4 == 0) ? 1.2f : 0.6f;
        drawLine3D(target, cam, Vec3(x, y, minB.z), Vec3(x, y, maxB.z), lineWidth, lineColor, screenW, screenH);
        drawLine3D(target, cam, Vec3(minB.x, y, z), Vec3(maxB.x, y, z), lineWidth, lineColor, screenW, screenH);
    }
}

void drawAxes(sf::RenderTarget& target, const Camera3D& cam,
              const Vec3& origin, float length,
              unsigned int screenW, unsigned int screenH,
              const sf::Font* font) {
    sf::Color xColor(220, 60, 60);
    sf::Color yColor(60, 180, 60);
    sf::Color zColor(60, 100, 220);
    
    // Axis lines with glow
    auto drawAxisWithGlow = [&](const Vec3& end, sf::Color color) {
        sf::Color glowCol = color; glowCol.a = 40;
        drawLine3D(target, cam, origin, end, 4.0f, glowCol, screenW, screenH);
        drawLine3D(target, cam, origin, end, 2.0f, color, screenW, screenH);
    };
    
    drawAxisWithGlow(origin + Vec3(length, 0, 0), xColor);
    drawAxisWithGlow(origin + Vec3(0, length, 0), yColor);
    drawAxisWithGlow(origin + Vec3(0, 0, length), zColor);
    
    // Arrow tips
    float arrowSize = length * 0.07f;
    drawCircle3D(target, cam, origin + Vec3(length, 0, 0), arrowSize, xColor, screenW, screenH, true, 0, 1);
    drawCircle3D(target, cam, origin + Vec3(0, length, 0), arrowSize, yColor, screenW, screenH, true, 0, 1);
    drawCircle3D(target, cam, origin + Vec3(0, 0, length), arrowSize, zColor, screenW, screenH, true, 0, 1);
    
    if (font) {
        auto drawLabel = [&](const Vec3& pos, const std::string& label, sf::Color color) {
            sf::Vector2f screenPos = cam.project(pos, screenW, screenH);
            sf::Text shadow(*font, label, LABEL_FONT_SIZE);
            shadow.setFillColor(sf::Color(255, 255, 255, 180));
            shadow.setStyle(sf::Text::Bold);
            shadow.setPosition({screenPos.x + 1.0f, screenPos.y + 1.0f});
            target.draw(shadow);
            
            sf::Text text(*font, label, LABEL_FONT_SIZE);
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

// Calculate 3D SDF
float calculateSDF3D(const Vec3& pos, const std::vector<ObstacleND>& obstacles) {
    float min_sdf = std::numeric_limits<float>::max();
    
    for (const auto& obs : obstacles) {
        if (obs.dimensions() < 3) continue;
        
        float dx = pos.x - obs.center(0);
        float dy = pos.y - obs.center(1);
        float dz = pos.z - obs.center(2);
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        float sdf = dist - obs.radius;
        
        min_sdf = std::min(min_sdf, sdf);
    }
    
    return min_sdf;
}

} // anonymous namespace

// --- Drawable with collision info ---
struct Drawable3D {
    enum Type { LINE, SPHERE, COLLISION_SPHERE };
    Type type;
    Vec3 p1, p2;
    float radius;
    sf::Color color;
    float depth;
    int collision_status;  // 0=safe, 1=near, 2=collision
    
    static Drawable3D makeLine(const Vec3& start, const Vec3& end, sf::Color col) {
        Drawable3D d;
        d.type = LINE;
        d.p1 = start; d.p2 = end;
        d.color = col;
        d.collision_status = 0;
        return d;
    }
    
    static Drawable3D makeSphere(const Vec3& center, float r, sf::Color col) {
        Drawable3D d;
        d.type = SPHERE;
        d.p1 = center;
        d.radius = r;
        d.color = col;
        d.collision_status = 0;
        return d;
    }
    
    static Drawable3D makeCollisionSphere(const Vec3& center, float r, int status) {
        Drawable3D d;
        d.type = COLLISION_SPHERE;
        d.p1 = center;
        d.radius = r;
        d.collision_status = status;
        // Color based on status
        if (status == 2) d.color = COLLISION_COLOR;
        else if (status == 1) d.color = NEAR_COLOR;
        else d.color = SAFE_COLOR;
        return d;
    }
};

// --- Main Visualization ---
void drawScene3D(sf::RenderTarget& target,
                 const std::vector<ObstacleND>& obstacles,
                 const Trajectory& trajectory,
                 const Camera3D& cam,
                 const sf::Font* font,
                 bool show_collision_spheres,
                 bool show_legend,
                 bool show_axes,
                 bool show_grid,
                 float collision_threshold,
                 unsigned int screenW, unsigned int screenH,
                 int collision_count = 0, int near_count = 0, int safe_count = 0) {
    
    // Compute bounds
    Vec3 minB(1e10f, 1e10f, 1e10f), maxB(-1e10f, -1e10f, -1e10f);
    
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
    
    // Add padding
    Vec3 range = maxB - minB;
    minB = minB - range * 0.1f;
    maxB = maxB + range * 0.1f;
    
    // Depth range
    float minDepth = 1e10f, maxDepth = 0.0f;
    for (const auto& node : trajectory.nodes) {
        if (node.position.size() >= 3) {
            Vec3 pos(node.position(0), node.position(1), node.position(2));
            float d = cam.getDepth(pos);
            minDepth = std::min(minDepth, d);
            maxDepth = std::max(maxDepth, d);
        }
    }
    
    // Draw background
    drawGradientBackground(target, screenW, screenH);
    
    // Draw grid
    if (show_grid) {
        drawGrid(target, cam, minB, maxB, 10, screenW, screenH);
    }
    
    // Collect drawables
    std::vector<Drawable3D> drawables;
    
    // Obstacles
    for (const auto& obs : obstacles) {
        if (obs.dimensions() >= 3) {
            Vec3 pos(obs.center(0), obs.center(1), obs.center(2));
            auto d = Drawable3D::makeSphere(pos, obs.radius, OBSTACLE_BASE_COLOR);
            d.depth = cam.getDepth(pos);
            drawables.push_back(d);
        }
    }
    
    // Trajectory lines
    for (size_t i = 0; i + 1 < trajectory.nodes.size(); ++i) {
        if (trajectory.nodes[i].position.size() >= 3) {
            Vec3 p1(trajectory.nodes[i].position(0), trajectory.nodes[i].position(1), trajectory.nodes[i].position(2));
            Vec3 p2(trajectory.nodes[i+1].position(0), trajectory.nodes[i+1].position(1), trajectory.nodes[i+1].position(2));
            auto d = Drawable3D::makeLine(p1, p2, TRAJECTORY_COLOR);
            d.depth = cam.getDepth((p1 + p2) * 0.5f);
            drawables.push_back(d);
        }
    }
    
    // Collision spheres for each node
    if (show_collision_spheres) {
        for (size_t i = 0; i < trajectory.nodes.size(); ++i) {
            const auto& node = trajectory.nodes[i];
            if (node.position.size() < 3) continue;
            
            Vec3 pos(node.position(0), node.position(1), node.position(2));
            float sdf = calculateSDF3D(pos, obstacles);
            float effective_sdf = sdf - node.radius;
            
            int status = 0;  // safe
            if (effective_sdf < 0.0f) status = 2;  // collision
            else if (effective_sdf < collision_threshold) status = 1;  // near
            
            auto d = Drawable3D::makeCollisionSphere(pos, node.radius, status);
            d.depth = cam.getDepth(pos);
            drawables.push_back(d);
        }
    }
    
    // Start/Goal markers
    if (!trajectory.nodes.empty() && trajectory.nodes[0].position.size() >= 3) {
        Vec3 startPos(trajectory.nodes[trajectory.start_index].position(0),
                      trajectory.nodes[trajectory.start_index].position(1),
                      trajectory.nodes[trajectory.start_index].position(2));
        Vec3 goalPos(trajectory.nodes[trajectory.goal_index].position(0),
                     trajectory.nodes[trajectory.goal_index].position(1),
                     trajectory.nodes[trajectory.goal_index].position(2));
        
        auto ds = Drawable3D::makeSphere(startPos, 18.0f, START_COLOR);
        ds.depth = cam.getDepth(startPos) - 0.1f;
        drawables.push_back(ds);
        
        auto dg = Drawable3D::makeSphere(goalPos, 18.0f, GOAL_COLOR);
        dg.depth = cam.getDepth(goalPos) - 0.1f;
        drawables.push_back(dg);
    }
    
    // Sort by depth
    std::sort(drawables.begin(), drawables.end(),
              [](const Drawable3D& a, const Drawable3D& b) { return a.depth > b.depth; });
    
    // Render
    for (const auto& d : drawables) {
        switch (d.type) {
            case Drawable3D::LINE: {
                sf::Color glowCol = d.color; glowCol.a = 50;
                drawLine3D(target, cam, d.p1, d.p2, 4.0f, glowCol, screenW, screenH);
                drawLine3D(target, cam, d.p1, d.p2, 2.0f, d.color, screenW, screenH);
                break;
            }
            case Drawable3D::SPHERE:
                drawCircle3D(target, cam, d.p1, d.radius, d.color, screenW, screenH, true, minDepth, maxDepth);
                break;
            case Drawable3D::COLLISION_SPHERE: {
                sf::Color fillCol = d.color;
                // Higher alpha for better visibility
                fillCol.a = (d.collision_status == 2) ? 230 : (d.collision_status == 1) ? 200 : 180;
                drawCircle3D(target, cam, d.p1, d.radius, fillCol, screenW, screenH, false, minDepth, maxDepth, true);
                break;
            }
        }
    }
    
    // Draw axes
    if (show_axes) {
        float axisLen = std::max({maxB.x - minB.x, maxB.y - minB.y, maxB.z - minB.z}) * 0.15f;
        drawAxes(target, cam, minB, axisLen, screenW, screenH, font);
    }
    
    // Title
    if (font) {
        sf::Text title(*font, "3D Collision Visualization", TITLE_FONT_SIZE);
        title.setFillColor(TEXT_COLOR);
        title.setStyle(sf::Text::Bold);
        title.setPosition({15.0f, 10.0f});
        target.draw(title);
    }
    
    // Legend
    if (font && show_legend) {
        float lx = screenW - 190.0f;
        float ly = 15.0f;
        
        sf::RectangleShape shadow({175.0f, 155.0f});
        shadow.setPosition({lx + 3.0f, ly + 3.0f});
        shadow.setFillColor(sf::Color(0, 0, 0, 25));
        target.draw(shadow);
        
        sf::RectangleShape bg({175.0f, 155.0f});
        bg.setPosition({lx, ly});
        bg.setFillColor(sf::Color(255, 255, 255, 245));
        bg.setOutlineColor(sf::Color(180, 190, 200));
        bg.setOutlineThickness(1.0f);
        target.draw(bg);
        
        float itemY = ly + 12.0f;
        float itemH = 22.0f;
        
        auto drawItem = [&](const std::string& label, sf::Color color) {
            sf::CircleShape marker(7.0f);
            marker.setPosition({lx + 12.0f, itemY + 2.0f});
            marker.setFillColor(color);
            target.draw(marker);
            
            sf::CircleShape hl(2.5f);
            hl.setPosition({lx + 14.0f, itemY + 4.0f});
            hl.setFillColor(sf::Color(255, 255, 255, 100));
            target.draw(hl);
            
            sf::Text text(*font, label, LEGEND_FONT_SIZE);
            text.setFillColor(TEXT_COLOR);
            text.setPosition({lx + 35.0f, itemY});
            target.draw(text);
            
            itemY += itemH;
        };
        
        drawItem("Safe (no collision)", SAFE_COLOR);
        drawItem("Near collision", NEAR_COLOR);
        drawItem("In collision", COLLISION_COLOR);
        drawItem("Obstacles", OBSTACLE_BASE_COLOR);
        drawItem("Start", START_COLOR);
        drawItem("Goal", GOAL_COLOR);
    }
    
    // Status box
    if (font && show_collision_spheres) {
        float sx = 15.0f;
        float sy = screenH - 85.0f;
        
        sf::RectangleShape statusBg({200.0f, 70.0f});
        statusBg.setPosition({sx, sy});
        statusBg.setFillColor(sf::Color(248, 250, 255, 248));
        statusBg.setOutlineColor(sf::Color(180, 190, 200));
        statusBg.setOutlineThickness(1.0f);
        target.draw(statusBg);
        
        char buf[64];
        snprintf(buf, sizeof(buf), "Collision: %d", collision_count);
        sf::Text t1(*font, buf, STATUS_FONT_SIZE);
        t1.setFillColor(collision_count > 0 ? COLLISION_COLOR : TEXT_COLOR);
        t1.setPosition({sx + 12.0f, sy + 10.0f});
        target.draw(t1);
        
        snprintf(buf, sizeof(buf), "Near: %d", near_count);
        sf::Text t2(*font, buf, STATUS_FONT_SIZE);
        t2.setFillColor(near_count > 0 ? NEAR_COLOR : TEXT_COLOR);
        t2.setPosition({sx + 12.0f, sy + 30.0f});
        target.draw(t2);
        
        snprintf(buf, sizeof(buf), "Safe: %d", safe_count);
        sf::Text t3(*font, buf, STATUS_FONT_SIZE);
        t3.setFillColor(SAFE_COLOR);
        t3.setPosition({sx + 12.0f, sy + 50.0f});
        target.draw(t3);
    }
}

int main() {
    std::cout << "==========================================\n"
              << "  3D Collision Visualization\n"
              << "==========================================\n\n";

    // Load config
    std::filesystem::path source_path(__FILE__);
    std::filesystem::path source_dir = source_path.parent_path();
    std::filesystem::path config_path = source_dir / "../Examples2d3d/configs/config_3d.yaml"; // Running from the root directory of the project
    
    std::string config_file;
    try {
        config_file = std::filesystem::canonical(config_path).string();
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: Config file not found: " << config_path << "\n";
        return 1;
    }
    
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_file);
        std::cout << "Loaded: " << config_file << "\n";
    } catch (const YAML::Exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Create task
    std::cout << "\n=== Creating 3D Task ===\n";
    auto task = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto obstacle_map = task->getObstacleMap();
    std::cout << "Obstacles: " << obstacle_map->size() << "\n";

    // Create planner
    std::cout << "\n=== Creating Planner ===\n";
    PCEConfig pce_config;
    pce_config.loadFromFile(config_file);
    
    ProximalCrossEntropyMotionPlanner planner(task);
    if (!planner.initialize(pce_config)) {
        std::cerr << "Planner initialization failed\n";
        return 1;
    }
    
    std::cout << "Trajectory: " << planner.getCurrentTrajectory().nodes.size() << " nodes\n";

    // Setup window
    sf::RenderWindow window(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}),
                           "3D Collision Visualization",
                           sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    sf::Font font;
    bool font_loaded = tryLoadFont(font);

    // Camera setup
    const Trajectory& traj = planner.getCurrentTrajectory();
    Vec3 minB(1e10f, 1e10f, 1e10f), maxB(-1e10f, -1e10f, -1e10f);
    for (const auto& node : traj.nodes) {
        if (node.position.size() >= 3) {
            minB.x = std::min(minB.x, node.position(0));
            minB.y = std::min(minB.y, node.position(1));
            minB.z = std::min(minB.z, node.position(2));
            maxB.x = std::max(maxB.x, node.position(0));
            maxB.y = std::max(maxB.y, node.position(1));
            maxB.z = std::max(maxB.z, node.position(2));
        }
    }
    
    Camera3D cam;
    cam.center = (minB + maxB) * 0.5f;
    cam.distance = (maxB - minB).length() * 0.9f;
    cam.theta = 0.5f;
    cam.phi = 0.4f;

    std::cout << "\n=== Controls ===\n"
              << "SPACE:       Start optimization\n"
              << "R:           Reset trajectory\n"
              << "C:           Toggle collision spheres\n"
              << "Mouse drag:  Rotate view\n"
              << "Scroll:      Zoom\n"
              << "A:           Toggle auto-rotate\n"
              << "X:           Toggle axes\n"
              << "G:           Toggle grid\n"
              << "L:           Toggle legend\n"
              << "S:           Save PNG\n"
              << "ESC:         Quit\n"
              << "================\n\n"
              << "Collision colors:\n"
              << "  GREEN  = Safe\n"
              << "  ORANGE = Near collision\n"
              << "  RED    = In collision\n\n";

    bool show_collision_spheres = true;
    bool show_legend = true;
    bool show_axes = true;
    bool show_grid = true;
    bool auto_rotate = false;
    bool dragging = false;
    sf::Vector2i lastMouse;
    float collision_threshold = 10.0f;
    int save_counter = 0;

    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            
            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                switch (key->code) {
                    case sf::Keyboard::Key::Escape: window.close(); break;
                    case sf::Keyboard::Key::C:
                        show_collision_spheres = !show_collision_spheres;
                        std::cout << "Collision spheres: " << (show_collision_spheres ? "ON" : "OFF") << "\n";
                        break;
                    case sf::Keyboard::Key::L: show_legend = !show_legend; break;
                    case sf::Keyboard::Key::X: show_axes = !show_axes; break;
                    case sf::Keyboard::Key::G: show_grid = !show_grid; break;
                    case sf::Keyboard::Key::A: auto_rotate = !auto_rotate; break;
                    case sf::Keyboard::Key::R:
                        std::cout << "\n--- Resetting Trajectory ---\n";
                        if (planner.initialize(pce_config)) {
                            std::cout << "Reset complete\n";
                        }
                        break;
                    case sf::Keyboard::Key::Space:
                        std::cout << "\n--- Starting Optimization ---\n";
                        if (planner.optimize()) {
                            std::cout << "Optimization completed\n";
                        } else {
                            std::cout << "Optimization failed\n";
                        }
                        break;
                    case sf::Keyboard::Key::Left: cam.rotate(-0.1f, 0); break;
                    case sf::Keyboard::Key::Right: cam.rotate(0.1f, 0); break;
                    case sf::Keyboard::Key::Up: cam.rotate(0, 0.1f); break;
                    case sf::Keyboard::Key::Down: cam.rotate(0, -0.1f); break;
                    case sf::Keyboard::Key::S: {
                        sf::RenderTexture rt;
                        if (rt.resize({WINDOW_WIDTH, WINDOW_HEIGHT})) {
                            rt.clear(sf::Color::White);
                            const Trajectory& current = planner.getCurrentTrajectory();
                            drawScene3D(rt, task->getObstacles(), current, cam,
                                       font_loaded ? &font : nullptr,
                                       show_collision_spheres, show_legend, show_axes, show_grid,
                                       collision_threshold, WINDOW_WIDTH, WINDOW_HEIGHT);
                            rt.display();
                            std::string filename = "collision_3d_" + std::to_string(save_counter++) + ".png";
                            if (rt.getTexture().copyToImage().saveToFile(filename)) {
                                std::cout << "Saved: " << filename << "\n";
                            }
                        }
                        break;
                    }
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
        
        if (auto_rotate) {
            cam.rotate(AUTO_ROTATE_SPEED, 0);
        }
        
        // Count collision status
        const Trajectory& current_traj = planner.getCurrentTrajectory();
        int collision_count = 0, near_count = 0, safe_count = 0;
        
        for (const auto& node : current_traj.nodes) {
            if (node.position.size() < 3) continue;
            Vec3 pos(node.position(0), node.position(1), node.position(2));
            float sdf = calculateSDF3D(pos, task->getObstacles());
            float effective_sdf = sdf - node.radius;
            
            if (effective_sdf < 0.0f) collision_count++;
            else if (effective_sdf < collision_threshold) near_count++;
            else safe_count++;
        }
        
        // Render
        window.clear(sf::Color::White);
        drawScene3D(window, task->getObstacles(), current_traj, cam,
                   font_loaded ? &font : nullptr,
                   show_collision_spheres, show_legend, show_axes, show_grid,
                   collision_threshold, WINDOW_WIDTH, WINDOW_HEIGHT,
                   collision_count, near_count, safe_count);
        
        // Hint
        if (font_loaded) {
            std::string hint = auto_rotate ? "[Auto-rotating] " : "";
            hint += "SPACE: Optimize | C: Toggle collision | Drag: Rotate";
            sf::Text hintText(font, hint, HINT_FONT_SIZE);
            hintText.setFillColor(sf::Color(100, 100, 100));
            hintText.setPosition({WINDOW_WIDTH - 450.0f, WINDOW_HEIGHT - 22.0f});
            window.draw(hintText);
        }
        
        window.display();
    }

    // Save obstacle map
    unsigned int seed = 999;
    if (config["experiment"] && config["experiment"]["random_seed"]) {
        seed = config["experiment"]["random_seed"].as<unsigned int>();
    }
    obstacle_map->saveToJSON("obstacle_map_3d_seed_" + std::to_string(seed) + ".json");
    std::cout << "\nSaved obstacle map.\n";

    return 0;
}