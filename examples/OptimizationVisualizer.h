#pragma once

#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include "IterationData.h"
#include "ObstacleMap.h"
#include "visualization_base.h"    // Reuse basic 2D drawing functions
#include "visualization3d_base.h"  // Reuse 3D drawing primitives

/**
 * @brief Publication-quality visualization for optimization results
 * 
 * Designed for tier-1 robotics journal submissions (IEEE Transactions on Robotics,
 * Robotics and Automation Letters, etc.)
 * 
 * This class extends the basic visualization functions from visualization_base.h
 * with publication-quality styling including:
 * - Thick trajectory lines for print clarity (vs thin LineStrip in visualization_base.h)
 * - matplotlib-compatible color palette
 * - Axis labels, tick marks, and legends
 * - High-resolution export options
 * 
 * Color scheme follows matplotlib default for consistency with Python plots:
 * - Current trajectory: #0000ff (pure blue)
 * - Previous trajectories: #aec7e8 (light blue) with alpha gradient
 * - Start marker: #2ca02c (green)
 * - Goal marker: #ff7f0e (orange)
 */
class OptimizationVisualizer {
public:
    OptimizationVisualizer(unsigned int width = 800, unsigned int height = 600)
        : window_width_(width), window_height_(height), output_prefix_("optimization") {}

    void setOutputPrefix(const std::string& prefix) { output_prefix_ = prefix; }

    /**
     * @brief Non-interactive batch save of final trajectory state
     */
    void saveStaticPlot(const ObstacleMap& obstacle_map,
                        const Trajectory& final_trajectory,
                        const std::string& filename) {
        OptimizationHistory history;
        IterationData data;
        data.iteration = 0;
        data.mean_trajectory = final_trajectory;
        data.total_cost = 0.0f;
        history.addIteration(data);

        sf::Font font;
        bool font_loaded = loadFont(font);

        sf::RenderTexture rt;
        if (!createRenderTexture(rt, STATIC_SAVE_SCALE)) return;

        rt.clear(sf::Color::White);
        drawTrajectoryFigure(rt, obstacle_map, history, 0, false, false, font, font_loaded, false,
                            window_width_, window_height_);
        rt.display();

        if (!rt.getTexture().copyToImage().saveToFile(filename)) {
            std::cerr << "Failed to save image: " << filename << std::endl;
        }
    }

    /**
     * @brief Interactive trajectory evolution viewer
     */
    void showTrajectoryEvolution(const ObstacleMap& obstacle_map,
                                  const OptimizationHistory& history,
                                  const std::string& title = "Trajectory Evolution") {
        unsigned int current_width = window_width_;
        unsigned int current_height = window_height_;
        
        sf::RenderWindow window(sf::VideoMode({current_width, current_height}), title,
                                sf::Style::Titlebar | sf::Style::Close | sf::Style::Resize);
        window.setFramerateLimit(60);

        sf::Font font;
        bool font_loaded = loadFont(font);

        int current_iter = history.iterations.size() - 1;
        bool show_samples = true;
        bool show_all_means = true;
        bool playing = false;
        int play_speed = 5;
        int frame_counter = 0;
        int save_counter = 0;

        printTrajectoryControls();

        while (window.isOpen()) {
            while (const auto event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>()) window.close();

                // Handle window resize
                if (const auto* resized = event->getIf<sf::Event::Resized>()) {
                    current_width = resized->size.x;
                    current_height = resized->size.y;
                    sf::View view(sf::FloatRect({0.f, 0.f}, 
                                  {static_cast<float>(current_width), static_cast<float>(current_height)}));
                    window.setView(view);
                }

                if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                    switch (key->code) {
                        case sf::Keyboard::Key::Escape: window.close(); break;
                        case sf::Keyboard::Key::Left:
                            current_iter = std::max(0, current_iter - 1); break;
                        case sf::Keyboard::Key::Right:
                            current_iter = std::min((int)history.iterations.size() - 1, current_iter + 1); break;
                        case sf::Keyboard::Key::Home: current_iter = 0; break;
                        case sf::Keyboard::Key::End: current_iter = history.iterations.size() - 1; break;
                        case sf::Keyboard::Key::Space: playing = !playing; frame_counter = 0; break;
                        case sf::Keyboard::Key::A: show_samples = !show_samples; break;
                        case sf::Keyboard::Key::M: show_all_means = !show_all_means; break;
                        case sf::Keyboard::Key::S:
                            saveTrajectoryImage(obstacle_map, history, current_iter,
                                               show_samples, show_all_means, font, font_loaded,
                                               save_counter++, 1.0f);
                            break;
                        case sf::Keyboard::Key::P:
                            saveTrajectoryImage(obstacle_map, history, current_iter,
                                               show_samples, show_all_means, font, font_loaded,
                                               save_counter++, HIGHRES_SCALE);
                            break;
                        case sf::Keyboard::Key::G:
                            saveAnimationGIF(obstacle_map, history, show_samples, show_all_means, font, font_loaded);
                            break;
                        default: break;
                    }
                }
            }

            if (playing && ++frame_counter >= play_speed) {
                frame_counter = 0;
                current_iter = (current_iter + 1) % history.iterations.size();
            }

            window.clear(sf::Color::White);
            drawTrajectoryFigure(window, obstacle_map, history, current_iter,
                                show_samples, show_all_means, font, font_loaded, true,
                                current_width, current_height);
            window.display();
        }
    }

    /**
     * @brief Interactive cost convergence plot viewer
     */
    void showCostPlot(const OptimizationHistory& history,
                      const std::string& title = "Cost Convergence") {
        unsigned int current_width = window_width_;
        unsigned int current_height = window_height_;
        
        sf::RenderWindow window(sf::VideoMode({current_width, current_height}), title,
                                sf::Style::Titlebar | sf::Style::Close | sf::Style::Resize);
        window.setFramerateLimit(60);

        sf::Font font;
        bool font_loaded = loadFont(font);

        int save_counter = 0;

        printCostControls();

        while (window.isOpen()) {
            while (const auto event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>()) window.close();

                // Handle window resize
                if (const auto* resized = event->getIf<sf::Event::Resized>()) {
                    current_width = resized->size.x;
                    current_height = resized->size.y;
                    sf::View view(sf::FloatRect({0.f, 0.f}, 
                                  {static_cast<float>(current_width), static_cast<float>(current_height)}));
                    window.setView(view);
                }

                if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                    switch (key->code) {
                        case sf::Keyboard::Key::Escape: window.close(); break;
                        case sf::Keyboard::Key::S:
                            saveCostImage(history, font, font_loaded,
                                         save_counter++, 1.0f);
                            break;
                        case sf::Keyboard::Key::P:
                            saveCostImage(history, font, font_loaded,
                                         save_counter++, HIGHRES_SCALE);
                            break;
                        default: break;
                    }
                }
            }

            window.clear(sf::Color::White);
            drawCostFigure(window, history, font, font_loaded, true,
                          current_width, current_height);
            window.display();
        }
    }

    void showAll(const ObstacleMap& obstacle_map, const OptimizationHistory& history,
                 const std::string& name = "") {
        if (!name.empty()) setOutputPrefix(name);
        showTrajectoryEvolution(obstacle_map, history,
                               name.empty() ? "Trajectory Evolution" : name + " - Trajectory");
        showCostPlot(history, name.empty() ? "Cost Convergence" : name + " - Cost");
    }

    /**
     * @brief Interactive 3D trajectory evolution viewer
     * 
     * Uses primitives from visualization3d_base.h for rendering.
     * Supports mouse rotation, keyboard navigation, and high-res export.
     */
    void showTrajectoryEvolution3D(const ObstacleMap& obstacle_map,
                                    const OptimizationHistory& history,
                                    const std::string& title = "3D Trajectory Evolution") {
        if (history.iterations.empty()) return;

        // Use wider window for 3D (initial size)
        unsigned int width_3d = window_width_ * 3 / 2;
        unsigned int height_3d = window_height_;

        sf::RenderWindow window(sf::VideoMode({width_3d, height_3d}), title,
                                sf::Style::Titlebar | sf::Style::Close | sf::Style::Resize);
        window.setFramerateLimit(60);

        sf::Font font;
        bool font_loaded = loadFont(font);

        // Compute bounds from all iterations
        viz3d::Vec3 minB, maxB;
        computeHistoryBounds3D(history, obstacle_map.getObstacles(), minB, maxB);

        // Setup camera
        viz3d::Camera3D cam;
        cam.setupFromBounds(minB, maxB);
        
        // Zoom in and adjust view for better framing
        cam.distance *= 0.65f;   // Zoom in (65% of default distance)
        cam.fov = 42.0f;         // Tighter field of view
        cam.theta = 0.6f;        // Horizontal viewing angle
        cam.phi = 0.45f;         // Slightly elevated view

        int current_iter = history.iterations.size() - 1;
        bool show_samples = true;
        bool show_all_means = true;
        bool playing = false;
        bool auto_rotate = false;
        int play_speed = 5;
        int frame_counter = 0;
        int save_counter = 0;

        // Mouse state
        bool dragging = false;
        sf::Vector2i lastMouse;

        print3DControls();

        while (window.isOpen()) {
            while (const auto event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>()) window.close();

                // Handle window resize
                if (const auto* resized = event->getIf<sf::Event::Resized>()) {
                    width_3d = resized->size.x;
                    height_3d = resized->size.y;
                    sf::View view(sf::FloatRect({0.f, 0.f}, 
                                  {static_cast<float>(width_3d), static_cast<float>(height_3d)}));
                    window.setView(view);
                }

                if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                    switch (key->code) {
                        case sf::Keyboard::Key::Escape: window.close(); break;
                        case sf::Keyboard::Key::Left:
                            current_iter = std::max(0, current_iter - 1); break;
                        case sf::Keyboard::Key::Right:
                            current_iter = std::min((int)history.iterations.size() - 1, current_iter + 1); break;
                        case sf::Keyboard::Key::Home: current_iter = 0; break;
                        case sf::Keyboard::Key::End: current_iter = history.iterations.size() - 1; break;
                        case sf::Keyboard::Key::Space: playing = !playing; frame_counter = 0; break;
                        case sf::Keyboard::Key::A: show_samples = !show_samples; break;
                        case sf::Keyboard::Key::M: show_all_means = !show_all_means; break;
                        case sf::Keyboard::Key::R: auto_rotate = !auto_rotate; break;
                        case sf::Keyboard::Key::S:
                            saveTrajectory3DImage(obstacle_map, history, current_iter,
                                                  show_samples, show_all_means, cam,
                                                  font, font_loaded, save_counter++, 1.0f);
                            break;
                        case sf::Keyboard::Key::P:
                            saveTrajectory3DImage(obstacle_map, history, current_iter,
                                                  show_samples, show_all_means, cam,
                                                  font, font_loaded, save_counter++, HIGHRES_SCALE);
                            break;
                        case sf::Keyboard::Key::G:
                            saveAnimation3DGIF(obstacle_map, history, show_samples, show_all_means,
                                               cam, font, font_loaded);
                            break;
                        default: break;
                    }
                }

                if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                    if (mb->button == sf::Mouse::Button::Left) {
                        dragging = true;
                        lastMouse = {mb->position.x, mb->position.y};
                    }
                }
                if (event->is<sf::Event::MouseButtonReleased>()) {
                    dragging = false;
                }
                if (const auto* mm = event->getIf<sf::Event::MouseMoved>()) {
                    if (dragging) {
                        float dx = static_cast<float>(mm->position.x - lastMouse.x);
                        float dy = static_cast<float>(mm->position.y - lastMouse.y);
                        cam.rotate(dx * viz3d::constants::ROTATION_SPEED,
                                   -dy * viz3d::constants::ROTATION_SPEED);
                        lastMouse = {mm->position.x, mm->position.y};
                    }
                }
                if (const auto* scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
                    // Proportional zoom - feels more natural
                    float zoom_amount = cam.distance * 0.1f * scroll->delta;
                    cam.zoom(-zoom_amount);
                }
            }

            if (playing && ++frame_counter >= play_speed) {
                frame_counter = 0;
                current_iter = (current_iter + 1) % history.iterations.size();
            }

            if (auto_rotate) {
                cam.rotate(viz3d::constants::AUTO_ROTATE_SPEED, 0);
            }

            window.clear(sf::Color::White);
            draw3DTrajectoryFigure(window, obstacle_map, history, current_iter,
                                   show_samples, show_all_means, cam,
                                   font, font_loaded, true, width_3d, height_3d,
                                   minB, maxB);
            window.display();
        }
    }

    /**
     * @brief Non-interactive batch save of 3D trajectory
     */
    void saveStatic3DPlot(const ObstacleMap& obstacle_map,
                          const Trajectory& final_trajectory,
                          const std::string& filename) {
        OptimizationHistory history;
        IterationData data;
        data.iteration = 0;
        data.mean_trajectory = final_trajectory;
        data.total_cost = 0.0f;
        history.addIteration(data);

        sf::Font font;
        bool font_loaded = loadFont(font);

        unsigned int width_3d = window_width_ * 3 / 2;
        unsigned int height_3d = window_height_;

        viz3d::Vec3 minB, maxB;
        computeHistoryBounds3D(history, obstacle_map.getObstacles(), minB, maxB);

        viz3d::Camera3D cam;
        cam.setupFromBounds(minB, maxB);
        
        // Zoom in and adjust view for better framing
        cam.distance *= 0.65f;   // Zoom in (65% of default distance)
        cam.fov = 42.0f;         // Tighter field of view
        cam.theta = 0.6f;        // Horizontal viewing angle
        cam.phi = 0.45f;         // Slightly elevated view

        sf::RenderTexture rt;
        unsigned int w = static_cast<unsigned int>(width_3d * STATIC_SAVE_SCALE);
        unsigned int h = static_cast<unsigned int>(height_3d * STATIC_SAVE_SCALE);

        if (!rt.resize({w, h})) {
            std::cerr << "Failed to create render texture\n";
            return;
        }

        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)width_3d, (float)height_3d}));
        view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
        rt.setView(view);

        rt.clear(sf::Color::White);
        draw3DTrajectoryFigure(rt, obstacle_map, history, 0, false, false, cam,
                               font, font_loaded, false, width_3d, height_3d,
                               minB, maxB);
        rt.display();

        if (!rt.getTexture().copyToImage().saveToFile(filename)) {
            std::cerr << "Failed to save image: " << filename << std::endl;
        }
    }

    /**
     * @brief Save animation as GIF (requires ImageMagick)
     */
    void saveAnimationGIF(const ObstacleMap& obstacle_map,
                          const OptimizationHistory& history,
                          bool show_samples, bool show_all_means,
                          const sf::Font& font, bool font_loaded,
                          int delay_ms = 100,
                          int final_frames = 8) {
        if (history.iterations.empty()) return;

        std::cout << "Saving animation frames...\n";

        sf::RenderTexture rt;
        if (!createRenderTexture(rt, GIF_SCALE)) return;

        std::string frame_dir = output_prefix_ + "_frames";
        std::system(("mkdir -p " + frame_dir).c_str());

        int frame_num = 0;
        
        // Phase 1: Optimization iterations
        for (size_t i = 0; i < history.iterations.size(); ++i) {
            rt.clear(sf::Color::White);
            drawTrajectoryFigure(rt, obstacle_map, history, i, show_samples, show_all_means,
                                font, font_loaded, false, window_width_, window_height_);
            rt.display();

            std::ostringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << frame_num++ << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());

            if ((i + 1) % 10 == 0 || i == history.iterations.size() - 1) {
                std::cout << "  Frame " << (i + 1) << "/" << history.iterations.size() << "\n";
            }
        }

        // Phase 2: Final frames with green optimized trajectory
        std::cout << "  Adding " << final_frames << " final frames (green trajectory)...\n";
        for (int f = 0; f < final_frames; ++f) {
            rt.clear(sf::Color::White);
            drawFinalTrajectoryFrame(rt, obstacle_map, history, font, font_loaded,
                                     window_width_, window_height_);
            rt.display();

            std::ostringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << frame_num++ << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());
        }

        std::string gif_name = output_prefix_ + "_animation.gif";
        
        // Try multiple methods to create GIF
        bool success = false;
        
        // Method 1: ImageMagick 7+ (magick)
        std::ostringstream cmd1;
        cmd1 << "magick -delay " << (delay_ms / 10) << " -loop 0 "
             << frame_dir << "/frame_*.png " << gif_name << " 2>/dev/null";
        
        std::cout << "Creating GIF (trying magick)...\n";
        if (std::system(cmd1.str().c_str()) == 0) {
            success = true;
        } else {
            // Method 2: ImageMagick 6 (convert)
            std::ostringstream cmd2;
            cmd2 << "convert -delay " << (delay_ms / 10) << " -loop 0 "
                 << frame_dir << "/frame_*.png " << gif_name << " 2>/dev/null";
            
            std::cout << "Trying convert...\n";
            if (std::system(cmd2.str().c_str()) == 0) {
                success = true;
            } else {
                // Method 3: ffmpeg
                std::ostringstream cmd3;
                cmd3 << "ffmpeg -y -framerate " << (1000 / delay_ms) << " -i "
                     << frame_dir << "/frame_%04d.png "
                     << "-vf \"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" "
                     << gif_name << " 2>/dev/null";
                
                std::cout << "Trying ffmpeg...\n";
                if (std::system(cmd3.str().c_str()) == 0) {
                    success = true;
                }
            }
        }

        if (success) {
            std::cout << "Saved: " << gif_name << "\n";
            std::system(("rm -rf " + frame_dir).c_str());
        } else {
            std::cerr << "GIF creation failed. Install one of:\n"
                      << "  sudo apt install imagemagick\n"
                      << "  sudo apt install ffmpeg\n"
                      << "Frames saved in: " << frame_dir << "/\n";
        }
    }

private:
    // === Configuration ===
    unsigned int window_width_;
    unsigned int window_height_;
    std::string output_prefix_;

    // Scale factors for different output types
    static constexpr float HIGHRES_SCALE = 4.0f;    // ~400 DPI for print
    static constexpr float STATIC_SAVE_SCALE = 2.0f;
    static constexpr float GIF_SCALE = 2.0f;

    // === Publication Color Palette (optimized for tier-1 conference submissions) ===
    // Note: Different from visualization_base.h colors which are designed for screen viewing
    struct Colors {
        // Trajectory colors - Pure blue for maximum visibility in print
        static constexpr auto current()     { return sf::Color(0, 0, 255); }         // Pure blue (#0000ff)
        static constexpr auto previous()    { return sf::Color(150, 180, 255); }     // Light blue for history
        static constexpr auto samples()     { return sf::Color(0, 0, 255, 40); }     // Pure blue, semi-transparent
        
        // Marker colors
        static constexpr auto start()       { return sf::Color(44, 160, 44); }       // #2ca02c
        static constexpr auto goal()        { return sf::Color(255, 127, 14); }      // #ff7f0e
        
        // Cost curve colors
        static constexpr auto total()       { return sf::Color(0, 0, 255); }         // Pure blue
        static constexpr auto collision()   { return sf::Color(255, 127, 14); }      // #ff7f0e
        static constexpr auto smoothness()  { return sf::Color(44, 160, 44); }       // #2ca02c
        
        // UI colors
        static constexpr auto obstacle()    { return sf::Color(200, 200, 200); }
        static constexpr auto obstacleBorder() { return sf::Color(140, 140, 140); }
        static constexpr auto axis()        { return sf::Color(40, 40, 40); }
        static constexpr auto grid()        { return sf::Color(230, 230, 230); }
        static constexpr auto text()        { return sf::Color(25, 25, 25); }
        static constexpr auto legendBg()    { return sf::Color(255, 255, 255, 245); }
        static constexpr auto legendBorder(){ return sf::Color(160, 160, 160); }
        static constexpr auto hint()        { return sf::Color(130, 130, 130); }
        
        // Collision status colors (for 3D visualization)
        static constexpr auto safe()        { return sf::Color(100, 255, 100); }      // Bright green
        static constexpr auto nearCollision() { return sf::Color(255, 200, 50); }     // Bright yellow-orange
        static constexpr auto inCollision() { return sf::Color(255, 80, 80); }        // Bright red
    };

    // === Typography (publication-quality sizes for tier-1 conferences) ===
    struct FontSize {
        static constexpr unsigned int title = 22;
        static constexpr unsigned int axisLabel = 22;      // Increased for visibility
        static constexpr unsigned int tickLabel = 16;      // Increased for readability
        static constexpr unsigned int legend = 16;
        static constexpr unsigned int stats = 14;
        static constexpr unsigned int hint = 12;
    };

    // === Layout (margins and spacing - adjusted for larger fonts) ===
    struct Layout {
        static constexpr float marginLeft = 85.0f;         // Increased for larger Y labels
        static constexpr float marginRight = 35.0f;
        static constexpr float marginTop = 60.0f;
        static constexpr float marginBottom = 75.0f;       // Increased for larger X labels
        static constexpr float legendPadding = 12.0f;
        static constexpr float itemSpacing = 24.0f;
    };

    // === Line/Marker Sizes (thicker for publication clarity) ===
    struct Sizes {
        static constexpr float currentLine = 3.5f;         // Slightly thicker
        static constexpr float previousLine = 2.0f;
        static constexpr float sampleLine = 1.0f;
        static constexpr float costLine = 3.0f;            // Thicker for visibility
        static constexpr float costLineSecondary = 2.5f;
        static constexpr float waypoint = 5.0f;
        static constexpr float startMarker = 12.0f;
        static constexpr float goalMarker = 14.0f;
        static constexpr float axisBorder = 2.0f;          // Thicker axis border
        static constexpr float legendBorder = 1.5f;
    };

    // === Helper Functions ===
    
    bool loadFont(sf::Font& font) {
        const std::vector<std::string> paths = {
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        };
        for (const auto& p : paths) {
            if (font.openFromFile(p)) return true;
        }
        std::cerr << "Warning: Could not load font\n";
        return false;
    }

    bool createRenderTexture(sf::RenderTexture& rt, float scale) {
        unsigned int w = static_cast<unsigned int>(window_width_ * scale);
        unsigned int h = static_cast<unsigned int>(window_height_ * scale);
        
        if (!rt.resize({w, h})) {
            std::cerr << "Failed to create render texture\n";
            return false;
        }
        
        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)window_width_, (float)window_height_}));
        view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
        rt.setView(view);
        return true;
    }

    void printTrajectoryControls() {
        std::cout << "\n=== Trajectory Viewer Controls ===\n"
                  << "Left/Right: Navigate iterations\n"
                  << "Home/End:   Jump to first/last\n"
                  << "Space:      Play/Pause animation\n"
                  << "A:          Toggle samples\n"
                  << "M:          Toggle previous means\n"
                  << "S:          Save PNG (1x)\n"
                  << "P:          Save PNG (4x high-res)\n"
                  << "G:          Save GIF animation\n"
                  << "ESC:        Exit\n"
                  << "==================================\n\n";
    }

    void printCostControls() {
        std::cout << "\n=== Cost Plot Controls ===\n"
                  << "S:   Save PNG (1x)\n"
                  << "P:   Save PNG (4x high-res)\n"
                  << "ESC: Exit\n"
                  << "==========================\n\n";
    }

    void print3DControls() {
        std::cout << "\n=== 3D Trajectory Viewer Controls ===\n"
                  << "Mouse drag:  Rotate view\n"
                  << "Scroll:      Zoom in/out\n"
                  << "Left/Right:  Navigate iterations\n"
                  << "Home/End:    Jump to first/last\n"
                  << "Space:       Play/Pause animation\n"
                  << "R:           Toggle auto-rotation\n"
                  << "A:           Toggle samples\n"
                  << "M:           Toggle previous means\n"
                  << "S:           Save PNG (1x)\n"
                  << "P:           Save PNG (4x high-res)\n"
                  << "G:           Save GIF animation\n"
                  << "ESC:         Exit\n"
                  << "======================================\n\n";
    }

    // === Publication-Quality Drawing (thick lines for print) ===
    // Note: visualization_base.h uses thin LineStrip via drawTrajectorySegments().
    // For publication figures, thick lines (RectangleShapes) are required for print clarity.

    template<typename RenderTarget>
    void drawThickLine(RenderTarget& target, sf::Vector2f p1, sf::Vector2f p2,
                       float width, sf::Color color) {
        sf::Vector2f d = p2 - p1;
        float len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len < 0.001f) return;
        
        sf::RectangleShape line({len, width});
        line.setPosition(p1);
        line.setFillColor(color);
        line.setRotation(sf::radians(std::atan2(d.y, d.x)));
        line.setOrigin({0, width / 2});
        target.draw(line);
    }

    template<typename RenderTarget>
    void drawLegendLine(RenderTarget& target, float x, float y, float width,
                        sf::Color color, float lineWidth = 3.0f) {
        drawThickLine(target, {x, y + 8.0f}, {x + width, y + 8.0f}, lineWidth, color);
    }

    template<typename RenderTarget>
    void drawLegendMarker(RenderTarget& target, const sf::Font& font,
                          float x, float y, const std::string& label, sf::Color color,
                          bool isSquare = false) {
        if (isSquare) {
            sf::RectangleShape marker({12.0f, 12.0f});
            marker.setPosition({x + 6.0f, y + 2.0f});
            marker.setFillColor(color);
            marker.setOutlineColor(sf::Color(color.r * 0.7f, color.g * 0.7f, color.b * 0.7f));
            marker.setOutlineThickness(1.0f);
            target.draw(marker);
        } else {
            sf::CircleShape marker(6.0f);
            marker.setPosition({x + 6.0f, y + 2.0f});
            marker.setFillColor(color);
            marker.setOutlineColor(sf::Color(color.r * 0.7f, color.g * 0.7f, color.b * 0.7f));
            marker.setOutlineThickness(1.0f);
            target.draw(marker);
        }
        
        sf::Text text(font, label, FontSize::legend);
        text.setFillColor(Colors::text());
        text.setPosition({x + 28.0f, y});
        target.draw(text);
    }

    template<typename RenderTarget>
    void drawLegendEntry(RenderTarget& target, const sf::Font& font,
                         float x, float y, const std::string& label, sf::Color color,
                         float lineWidth = 3.0f) {
        drawLegendLine(target, x, y, 24.0f, color, lineWidth);
        sf::Text text(font, label, FontSize::legend);
        text.setFillColor(Colors::text());
        text.setPosition({x + 32.0f, y});
        target.draw(text);
    }

    sf::RectangleShape createBox(float x, float y, float w, float h,
                                  sf::Color fill, sf::Color outline, float borderWidth = 1.0f) {
        sf::RectangleShape box({w, h});
        box.setPosition({x, y});
        box.setFillColor(fill);
        box.setOutlineColor(outline);
        box.setOutlineThickness(borderWidth);
        return box;
    }

    // === Publication-Quality Obstacle Drawing ===
    // Note: visualization_base.h drawObstacles() uses different colors (100,100,100,180).
    // This version uses lighter colors with borders for print clarity.
    template<typename RenderTarget>
    void drawObstaclesStyled(RenderTarget& target, const std::vector<ObstacleND>& obstacles,
                             float scale, float offset_x, float offset_y) {
        for (const auto& obs : obstacles) {
            if (obs.dimensions() >= 2) {
                float px = offset_x + obs.center(0) * scale;
                float py = offset_y + obs.center(1) * scale;
                float r = obs.radius * scale;
                
                sf::CircleShape circle(r);
                circle.setPosition({px - r, py - r});
                circle.setFillColor(Colors::obstacle());
                circle.setOutlineColor(Colors::obstacleBorder());
                circle.setOutlineThickness(1.5f);
                target.draw(circle);
            }
        }
    }

    // === Trajectory Validation ===
    // Check if a trajectory has valid 2D positions for drawing
    static bool isValidTrajectory2D(const Trajectory& trajectory) {
        if (trajectory.nodes.empty()) return false;
        for (const auto& node : trajectory.nodes) {
            if (node.position.size() < 2) return false;
        }
        return true;
    }
    
    // Check if a trajectory has valid 3D positions for drawing
    static bool isValidTrajectory3D(const Trajectory& trajectory) {
        if (trajectory.nodes.empty()) return false;
        for (const auto& node : trajectory.nodes) {
            if (node.position.size() < 3) return false;
        }
        return true;
    }

    // === Publication-Quality Trajectory Drawing ===
    // Note: visualization_base.h drawTrajectorySegments() uses thin LineStrip.
    // This version uses thick rectangle lines for publication figures.
    template<typename RenderTarget>
    void drawThickTrajectory(RenderTarget& target, const Trajectory& trajectory,
                             float scale, float offset_x, float offset_y,
                             sf::Color color, float lineWidth) {
        // Validate trajectory before drawing
        if (!isValidTrajectory2D(trajectory)) return;
        
        for (size_t i = 0; i + 1 < trajectory.nodes.size(); ++i) {
            sf::Vector2f p1 = {offset_x + trajectory.nodes[i].position(0) * scale,
                               offset_y + trajectory.nodes[i].position(1) * scale};
            sf::Vector2f p2 = {offset_x + trajectory.nodes[i+1].position(0) * scale,
                               offset_y + trajectory.nodes[i+1].position(1) * scale};
            drawThickLine(target, p1, p2, lineWidth, color);
        }
    }

    // === Main Drawing Functions ===

    // Green color for final optimized trajectory
    static constexpr sf::Color optimizedGreen() { return sf::Color(0, 180, 0); }

    /**
     * @brief Draw final frame with green optimized trajectory
     */
    template<typename RenderTarget>
    void drawFinalTrajectoryFrame(RenderTarget& target,
                                   const ObstacleMap& obstacle_map,
                                   const OptimizationHistory& history,
                                   const sf::Font& font, bool font_loaded,
                                   unsigned int width, unsigned int height) {
        if (history.iterations.empty()) return;

        const auto& final_iter = history.iterations.back();
        float map_w = obstacle_map.getMapWidth();
        float map_h = obstacle_map.getMapHeight();

        // Compute plot area
        float plot_w = width - Layout::marginLeft - Layout::marginRight;
        float plot_h = height - Layout::marginTop - Layout::marginBottom;
        float scale = std::min(plot_w / map_w, plot_h / map_h);
        
        float plot_actual_w = map_w * scale;
        float plot_actual_h = map_h * scale;
        float offset_x = Layout::marginLeft + (plot_w - plot_actual_w) / 2;
        float offset_y = Layout::marginTop + (plot_h - plot_actual_h) / 2;

        auto transform = [&](float x, float y) -> sf::Vector2f {
            return {offset_x + x * scale, offset_y + y * scale};
        };

        // Plot background
        target.draw(createBox(offset_x, offset_y, plot_actual_w, plot_actual_h,
                              sf::Color::White, Colors::axis(), Sizes::axisBorder));

        // Obstacles
        drawObstaclesStyled(target, obstacle_map.getObstacles(), scale, offset_x, offset_y);

        // Draw optimized trajectory in GREEN
        const auto& mean_traj = final_iter.mean_trajectory;
        sf::Color green = optimizedGreen();
        drawThickTrajectory(target, mean_traj, scale, offset_x, offset_y,
                           green, Sizes::currentLine + 0.5f);  // Slightly thicker

        // Waypoints in green
        if (isValidTrajectory2D(mean_traj)) {
            for (size_t i = 1; i + 1 < mean_traj.nodes.size(); ++i) {
                sf::Vector2f pos = transform(mean_traj.nodes[i].position(0), mean_traj.nodes[i].position(1));
                sf::CircleShape marker(Sizes::waypoint);
                marker.setPosition({pos.x - Sizes::waypoint, pos.y - Sizes::waypoint});
                marker.setFillColor(green);
                target.draw(marker);
            }
        }

        // Start and Goal markers
        if (isValidTrajectory2D(mean_traj) && !mean_traj.nodes.empty()) {
            size_t start_idx = std::min(mean_traj.start_index, mean_traj.nodes.size() - 1);
            size_t goal_idx = std::min(mean_traj.goal_index, mean_traj.nodes.size() - 1);
            
            sf::Vector2f start_pos = transform(mean_traj.nodes[start_idx].position(0),
                                               mean_traj.nodes[start_idx].position(1));
            sf::Vector2f goal_pos = transform(mean_traj.nodes[goal_idx].position(0),
                                              mean_traj.nodes[goal_idx].position(1));

            // Start (circle)
            sf::CircleShape start_marker(Sizes::startMarker);
            start_marker.setPosition({start_pos.x - Sizes::startMarker, start_pos.y - Sizes::startMarker});
            start_marker.setFillColor(Colors::start());
            start_marker.setOutlineColor(sf::Color(25, 100, 25));
            start_marker.setOutlineThickness(2.0f);
            target.draw(start_marker);

            // Goal (square)
            float gs = Sizes::goalMarker;
            sf::RectangleShape goal_marker({gs, gs});
            goal_marker.setPosition({goal_pos.x - gs / 2, goal_pos.y - gs / 2});
            goal_marker.setFillColor(Colors::goal());
            goal_marker.setOutlineColor(sf::Color(180, 90, 10));
            goal_marker.setOutlineThickness(2.0f);
            target.draw(goal_marker);
        }

        // Labels
        if (font_loaded) {
            // X-axis label
            sf::Text xlabel(font, "X Position", FontSize::axisLabel);
            xlabel.setFillColor(Colors::text());
            xlabel.setStyle(sf::Text::Bold);
            sf::FloatRect xb = xlabel.getLocalBounds();
            xlabel.setPosition({offset_x + plot_actual_w / 2 - xb.size.x / 2, offset_y + plot_actual_h + 45.0f});
            target.draw(xlabel);

            // Y-axis label
            sf::Text ylabel(font, "Y Position", FontSize::axisLabel);
            ylabel.setFillColor(Colors::text());
            ylabel.setStyle(sf::Text::Bold);
            ylabel.setRotation(sf::degrees(-90.0f));
            ylabel.setPosition({offset_x - 65.0f, offset_y + plot_actual_h / 2 + 40.0f});
            target.draw(ylabel);

            // Tick labels
            for (int i = 0; i <= 4; ++i) {
                float data_x = i * map_w / 4;
                float data_y = i * map_h / 4;
                float sx = offset_x + i * plot_actual_w / 4;
                float sy = offset_y + i * plot_actual_h / 4;

                sf::Text xt(font, std::to_string((int)data_x), FontSize::tickLabel);
                xt.setFillColor(Colors::text());
                sf::FloatRect xtb = xt.getLocalBounds();
                xt.setPosition({sx - xtb.size.x / 2, offset_y + plot_actual_h + 12.0f});
                target.draw(xt);

                sf::Text yt(font, std::to_string((int)data_y), FontSize::tickLabel);
                yt.setFillColor(Colors::text());
                sf::FloatRect ytb = yt.getLocalBounds();
                yt.setPosition({offset_x - ytb.size.x - 12.0f, sy - 9.0f});
                target.draw(yt);
            }

            // Title - "Optimized Trajectory"
            sf::Text title(font, "Optimized Trajectory", FontSize::title);
            title.setFillColor(green);
            title.setStyle(sf::Text::Bold);
            title.setPosition({offset_x, 15.0f});
            target.draw(title);

            // Cost display
            std::ostringstream cost_ss;
            cost_ss << "Final Cost: " << std::fixed << std::setprecision(2) << final_iter.total_cost;
            sf::Text cost_text(font, cost_ss.str(), FontSize::axisLabel);
            cost_text.setFillColor(Colors::text());
            cost_text.setPosition({offset_x + plot_actual_w - 200.0f, 18.0f});
            target.draw(cost_text);

            // Legend
            float lx = offset_x + plot_actual_w - 165.0f;
            float ly = offset_y + 12.0f;
            
            target.draw(createBox(lx, ly, 155.0f, 90.0f, Colors::legendBg(), Colors::legendBorder()));

            float item_y = ly + 10.0f;
            drawLegendEntry(target, font, lx + 8.0f, item_y, "Optimized", green, Sizes::currentLine);
            item_y += Layout::itemSpacing;
            drawLegendMarker(target, font, lx + 8.0f, item_y, "Start", Colors::start(), false);
            item_y += Layout::itemSpacing;
            drawLegendMarker(target, font, lx + 8.0f, item_y, "Goal", Colors::goal(), true);
        }
    }

    template<typename RenderTarget>
    void drawTrajectoryFigure(RenderTarget& target,
                               const ObstacleMap& obstacle_map,
                               const OptimizationHistory& history,
                               int current_iter,
                               bool show_samples, bool show_all_means,
                               const sf::Font& font, bool font_loaded,
                               bool show_hints,
                               unsigned int width, unsigned int height) {
        if (history.iterations.empty()) return;

        const auto& current = history.iterations[current_iter];
        float map_w = obstacle_map.getMapWidth();
        float map_h = obstacle_map.getMapHeight();

        // Compute plot area
        float plot_w = width - Layout::marginLeft - Layout::marginRight;
        float plot_h = height - Layout::marginTop - Layout::marginBottom;
        float scale = std::min(plot_w / map_w, plot_h / map_h);
        
        float plot_actual_w = map_w * scale;
        float plot_actual_h = map_h * scale;
        float offset_x = Layout::marginLeft + (plot_w - plot_actual_w) / 2;
        float offset_y = Layout::marginTop + (plot_h - plot_actual_h) / 2;

        auto transform = [&](float x, float y) -> sf::Vector2f {
            return {offset_x + x * scale, offset_y + y * scale};
        };

        // Plot background
        target.draw(createBox(offset_x, offset_y, plot_actual_w, plot_actual_h,
                              sf::Color::White, Colors::axis(), Sizes::axisBorder));

        // Obstacles (publication styling)
        drawObstaclesStyled(target, obstacle_map.getObstacles(), scale, offset_x, offset_y);

        // Samples
        if (show_samples && !current.samples.empty()) {
            for (const auto& sample : current.samples) {
                drawThickTrajectory(target, sample, scale, offset_x, offset_y,
                                   Colors::samples(), Sizes::sampleLine);
            }
        }

        // Previous trajectories with gradient
        if (show_all_means && current_iter > 0) {
            for (int i = 0; i < current_iter; ++i) {
                const auto& traj = history.iterations[i].mean_trajectory;
                float t = (float)i / current_iter;
                uint8_t alpha = static_cast<uint8_t>(50 + 150 * t);
                sf::Color col(150, 180, 255, alpha);  // Light blue gradient
                drawThickTrajectory(target, traj, scale, offset_x, offset_y,
                                   col, Sizes::previousLine);
            }
        }

        // Current trajectory
        const auto& mean_traj = current.mean_trajectory;
        drawThickTrajectory(target, mean_traj, scale, offset_x, offset_y,
                           Colors::current(), Sizes::currentLine);

        // Waypoints (only if trajectory is valid for 2D)
        if (isValidTrajectory2D(mean_traj)) {
            for (size_t i = 1; i + 1 < mean_traj.nodes.size(); ++i) {
                sf::Vector2f pos = transform(mean_traj.nodes[i].position(0), mean_traj.nodes[i].position(1));
                sf::CircleShape marker(Sizes::waypoint);
                marker.setPosition({pos.x - Sizes::waypoint, pos.y - Sizes::waypoint});
                marker.setFillColor(Colors::current());
                target.draw(marker);
            }
        }

        // Start and Goal markers (only if trajectory is valid for 2D)
        if (isValidTrajectory2D(mean_traj) && !mean_traj.nodes.empty()) {
            size_t start_idx = std::min(mean_traj.start_index, mean_traj.nodes.size() - 1);
            size_t goal_idx = std::min(mean_traj.goal_index, mean_traj.nodes.size() - 1);
            
            sf::Vector2f start_pos = transform(mean_traj.nodes[start_idx].position(0),
                                               mean_traj.nodes[start_idx].position(1));
            sf::Vector2f goal_pos = transform(mean_traj.nodes[goal_idx].position(0),
                                              mean_traj.nodes[goal_idx].position(1));

            // Start (circle)
            sf::CircleShape start_marker(Sizes::startMarker);
            start_marker.setPosition({start_pos.x - Sizes::startMarker, start_pos.y - Sizes::startMarker});
            start_marker.setFillColor(Colors::start());
            start_marker.setOutlineColor(sf::Color(25, 100, 25));
            start_marker.setOutlineThickness(2.0f);
            target.draw(start_marker);

            // Goal (square)
            float gs = Sizes::goalMarker;
            sf::RectangleShape goal_marker({gs, gs});
            goal_marker.setPosition({goal_pos.x - gs / 2, goal_pos.y - gs / 2});
            goal_marker.setFillColor(Colors::goal());
            goal_marker.setOutlineColor(sf::Color(180, 90, 10));
            goal_marker.setOutlineThickness(2.0f);
            target.draw(goal_marker);
        }

        if (font_loaded) {
            drawTrajectoryLabels(target, font, offset_x, offset_y, plot_actual_w, plot_actual_h,
                                 map_w, map_h, current_iter, history.iterations.size(),
                                 current.total_cost, show_hints, height);
        }
    }

    template<typename RenderTarget>
    void drawTrajectoryLabels(RenderTarget& target, const sf::Font& font,
                               float offset_x, float offset_y, float plot_w, float plot_h,
                               float map_w, float map_h, int current_iter, size_t total_iters,
                               float cost, bool show_hints, unsigned int window_h) {
        // X-axis label (larger, bold for publication)
        sf::Text xlabel(font, "X Position", FontSize::axisLabel);
        xlabel.setFillColor(Colors::text());
        xlabel.setStyle(sf::Text::Bold);
        sf::FloatRect xb = xlabel.getLocalBounds();
        xlabel.setPosition({offset_x + plot_w / 2 - xb.size.x / 2, offset_y + plot_h + 45.0f});
        target.draw(xlabel);

        // Y-axis label (larger, bold for publication)
        sf::Text ylabel(font, "Y Position", FontSize::axisLabel);
        ylabel.setFillColor(Colors::text());
        ylabel.setStyle(sf::Text::Bold);
        ylabel.setRotation(sf::degrees(-90.0f));
        ylabel.setPosition({offset_x - 65.0f, offset_y + plot_h / 2 + 40.0f});
        target.draw(ylabel);

        // Tick labels
        for (int i = 0; i <= 4; ++i) {
            float data_x = i * map_w / 4;
            float data_y = i * map_h / 4;
            float sx = offset_x + i * plot_w / 4;
            float sy = offset_y + i * plot_h / 4;

            sf::Text xt(font, std::to_string((int)data_x), FontSize::tickLabel);
            xt.setFillColor(Colors::text());
            sf::FloatRect xtb = xt.getLocalBounds();
            xt.setPosition({sx - xtb.size.x / 2, offset_y + plot_h + 12.0f});
            target.draw(xt);

            sf::Text yt(font, std::to_string((int)data_y), FontSize::tickLabel);
            yt.setFillColor(Colors::text());
            sf::FloatRect ytb = yt.getLocalBounds();
            yt.setPosition({offset_x - ytb.size.x - 12.0f, sy - 9.0f});
            target.draw(yt);
        }

        // Title
        std::ostringstream title_ss;
        title_ss << "Iteration " << (current_iter + 1) << "/" << total_iters;
        sf::Text title(font, title_ss.str(), FontSize::title);
        title.setFillColor(Colors::text());
        title.setStyle(sf::Text::Bold);
        title.setPosition({offset_x, 15.0f});
        target.draw(title);

        // Cost display
        std::ostringstream cost_ss;
        cost_ss << "Cost: " << std::fixed << std::setprecision(2) << cost;
        sf::Text cost_text(font, cost_ss.str(), FontSize::axisLabel);
        cost_text.setFillColor(Colors::text());
        cost_text.setPosition({offset_x + plot_w - 160.0f, 18.0f});
        target.draw(cost_text);

        // Legend (adjusted for larger fonts)
        float lx = offset_x + plot_w - 165.0f;
        float ly = offset_y + 12.0f;
        
        target.draw(createBox(lx, ly, 155.0f, 115.0f, Colors::legendBg(), Colors::legendBorder()));

        float item_y = ly + 10.0f;
        drawLegendEntry(target, font, lx + 8.0f, item_y, "Current", Colors::current(), Sizes::currentLine);
        item_y += Layout::itemSpacing;
        drawLegendEntry(target, font, lx + 8.0f, item_y, "Previous", Colors::previous(), Sizes::previousLine);
        item_y += Layout::itemSpacing;
        drawLegendMarker(target, font, lx + 8.0f, item_y, "Start", Colors::start(), false);
        item_y += Layout::itemSpacing;
        drawLegendMarker(target, font, lx + 8.0f, item_y, "Goal", Colors::goal(), true);

        // Interactive hints (only for screen display)
        if (show_hints) {
            sf::Text hint(font, "S/P: Save PNG | G: Save GIF | Space: Play", FontSize::hint);
            hint.setFillColor(Colors::hint());
            hint.setPosition({12.0f, window_h - 25.0f});
            target.draw(hint);
        }
    }

    template<typename RenderTarget>
    void drawCostFigure(RenderTarget& target,
                        const OptimizationHistory& history,
                        const sf::Font& font, bool font_loaded,
                        bool show_hints,
                        unsigned int width, unsigned int height) {
        if (history.iterations.empty()) return;

        float plot_left = Layout::marginLeft + 10.0f;
        float plot_right = width - Layout::marginRight;
        float plot_top = Layout::marginTop;
        float plot_bottom = height - Layout::marginBottom;
        float plot_w = plot_right - plot_left;
        float plot_h = plot_bottom - plot_top;

        auto costs = history.getCostHistory();

        // Compute Y-axis range
        float max_cost = *std::max_element(costs.begin(), costs.end());
        float min_cost = *std::min_element(costs.begin(), costs.end());
        float range = max_cost - min_cost;
        if (range < 1e-6f) range = std::max(max_cost * 0.1f, 1.0f);
        min_cost -= range * 0.05f;
        max_cost += range * 0.05f;
        range = max_cost - min_cost;

        // Plot background
        target.draw(createBox(plot_left, plot_top, plot_w, plot_h,
                              sf::Color::White, Colors::axis(), Sizes::axisBorder));

        // Grid lines
        constexpr int n_grid = 5;
        for (int i = 1; i < n_grid; ++i) {
            float y = plot_top + i * plot_h / n_grid;
            sf::RectangleShape hline({plot_w, 1.0f});
            hline.setPosition({plot_left, y});
            hline.setFillColor(Colors::grid());
            target.draw(hline);

            float x = plot_left + i * plot_w / n_grid;
            sf::RectangleShape vline({1.0f, plot_h});
            vline.setPosition({x, plot_top});
            vline.setFillColor(Colors::grid());
            target.draw(vline);
        }

        // Draw total cost curve only
        if (costs.size() >= 2) {
            for (size_t i = 0; i + 1 < costs.size(); ++i) {
                float x1 = plot_left + (float)i / (costs.size() - 1) * plot_w;
                float x2 = plot_left + (float)(i + 1) / (costs.size() - 1) * plot_w;
                float y1 = plot_bottom - (costs[i] - min_cost) / range * plot_h;
                float y2 = plot_bottom - (costs[i + 1] - min_cost) / range * plot_h;
                drawThickLine(target, {x1, y1}, {x2, y2}, Sizes::costLine, Colors::total());
            }
        }

        if (font_loaded) {
            drawCostLabels(target, font, plot_left, plot_top, plot_w, plot_h,
                          min_cost, max_cost, range, costs, show_hints, height);
        }
    }

    template<typename RenderTarget>
    void drawCostLabels(RenderTarget& target, const sf::Font& font,
                        float plot_left, float plot_top, float plot_w, float plot_h,
                        float min_cost, float max_cost, float range,
                        const std::vector<float>& costs,
                        bool show_hints, unsigned int window_h) {
        // Title
        sf::Text title(font, "Cost Convergence", FontSize::title);
        title.setFillColor(Colors::text());
        title.setStyle(sf::Text::Bold);
        title.setPosition({plot_left, 15.0f});
        target.draw(title);

        // X-axis label (larger, bold for publication)
        sf::Text xlabel(font, "Iteration", FontSize::axisLabel);
        xlabel.setFillColor(Colors::text());
        xlabel.setStyle(sf::Text::Bold);
        sf::FloatRect xb = xlabel.getLocalBounds();
        xlabel.setPosition({plot_left + plot_w / 2 - xb.size.x / 2, plot_top + plot_h + 48.0f});
        target.draw(xlabel);

        // Y-axis label (larger, bold for publication)
        sf::Text ylabel(font, "Cost", FontSize::axisLabel);
        ylabel.setFillColor(Colors::text());
        ylabel.setStyle(sf::Text::Bold);
        ylabel.setRotation(sf::degrees(-90.0f));
        ylabel.setPosition({18.0f, plot_top + plot_h / 2 + 20.0f});
        target.draw(ylabel);

        // Y tick labels
        constexpr int n_grid = 5;
        for (int i = 0; i <= n_grid; ++i) {
            float val = max_cost - i * range / n_grid;
            float y = plot_top + i * plot_h / n_grid;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << val;
            sf::Text label(font, ss.str(), FontSize::tickLabel);
            label.setFillColor(Colors::text());
            sf::FloatRect lb = label.getLocalBounds();
            label.setPosition({plot_left - lb.size.x - 12.0f, y - 9.0f});
            target.draw(label);
        }

        // X tick labels
        int n_x_ticks = std::min(10, (int)costs.size());
        for (int i = 0; i <= n_x_ticks; ++i) {
            int iter = i * (costs.size() - 1) / n_x_ticks;
            float x = plot_left + (float)iter / (costs.size() - 1) * plot_w;
            sf::Text label(font, std::to_string(iter + 1), FontSize::tickLabel);
            label.setFillColor(Colors::text());
            sf::FloatRect lb = label.getLocalBounds();
            label.setPosition({x - lb.size.x / 2, plot_top + plot_h + 12.0f});
            target.draw(label);
        }

        // Stats box (adjusted for larger fonts)
        float sx = plot_left + 12.0f;
        float sy = plot_top + 12.0f;
        
        target.draw(createBox(sx, sy, 175.0f, 88.0f,
                              sf::Color(248, 250, 255, 248), Colors::legendBorder()));

        float stat_y = sy + 10.0f;
        float improve = (costs.front() - costs.back()) / costs.front() * 100;
        
        std::ostringstream ss1, ss2, ss3;
        ss1 << "Initial: " << std::fixed << std::setprecision(2) << costs.front();
        ss2 << "Final: " << std::fixed << std::setprecision(2) << costs.back();
        ss3 << "Reduction: " << std::fixed << std::setprecision(1) << improve << "%";

        for (const auto& text : {ss1.str(), ss2.str(), ss3.str()}) {
            sf::Text t(font, text, FontSize::stats);
            t.setFillColor(Colors::text());
            t.setPosition({sx + 10.0f, stat_y});
            target.draw(t);
            stat_y += 24.0f;
        }

        // Interactive hints
        if (show_hints) {
            sf::Text hint(font, "S/P: Save PNG", FontSize::hint);
            hint.setFillColor(Colors::hint());
            hint.setPosition({12.0f, window_h - 25.0f});
            target.draw(hint);
        }
    }

    // === 3D Visualization Methods ===
    // Uses primitives from visualization3d_base.h

    void computeHistoryBounds3D(const OptimizationHistory& history,
                                 const std::vector<ObstacleND>& obstacles,
                                 viz3d::Vec3& minB, viz3d::Vec3& maxB) {
        minB = viz3d::Vec3(1e10f, 1e10f, 1e10f);
        maxB = viz3d::Vec3(-1e10f, -1e10f, -1e10f);

        for (const auto& iter : history.iterations) {
            for (const auto& node : iter.mean_trajectory.nodes) {
                if (node.position.size() >= 3) {
                    minB.x = std::min(minB.x, node.position(0));
                    minB.y = std::min(minB.y, node.position(1));
                    minB.z = std::min(minB.z, node.position(2));
                    maxB.x = std::max(maxB.x, node.position(0));
                    maxB.y = std::max(maxB.y, node.position(1));
                    maxB.z = std::max(maxB.z, node.position(2));
                }
            }
            for (const auto& sample : iter.samples) {
                for (const auto& node : sample.nodes) {
                    if (node.position.size() >= 3) {
                        minB.x = std::min(minB.x, node.position(0));
                        minB.y = std::min(minB.y, node.position(1));
                        minB.z = std::min(minB.z, node.position(2));
                        maxB.x = std::max(maxB.x, node.position(0));
                        maxB.y = std::max(maxB.y, node.position(1));
                        maxB.z = std::max(maxB.z, node.position(2));
                    }
                }
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

        viz3d::Vec3 range = maxB - minB;
        minB = minB - range * 0.1f;
        maxB = maxB + range * 0.1f;
    }

    // Calculate 3D signed distance field for collision checking
    float calculateSDF3D(const viz3d::Vec3& pos, const std::vector<ObstacleND>& obstacles) {
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

    // Get collision status: 0=safe, 1=near, 2=collision
    int getCollisionStatus(float effective_sdf, float threshold = 10.0f) {
        if (effective_sdf < 0.0f) return 2;  // collision
        if (effective_sdf < threshold) return 1;  // near
        return 0;  // safe
    }

    sf::Color getCollisionColor(int status) {
        if (status == 2) return Colors::inCollision();
        if (status == 1) return Colors::nearCollision();
        return Colors::safe();
    }

    template<typename RenderTarget>
    void draw3DTrajectoryFigure(RenderTarget& target,
                                 const ObstacleMap& obstacle_map,
                                 const OptimizationHistory& history,
                                 int current_iter,
                                 bool show_samples, bool show_all_means,
                                 const viz3d::Camera3D& cam,
                                 const sf::Font& font, bool font_loaded,
                                 bool show_hints,
                                 unsigned int width, unsigned int height,
                                 const viz3d::Vec3& minB, const viz3d::Vec3& maxB,
                                 bool show_collision_spheres = false,
                                 float collision_threshold = 10.0f) {
        if (history.iterations.empty()) return;

        const auto& current = history.iterations[current_iter];
        const auto& mean_traj = current.mean_trajectory;

        // Check dimensionality
        if (mean_traj.nodes.empty() || mean_traj.nodes[0].position.size() < 3) {
            std::cerr << "Warning: Trajectory is not 3D\n";
            return;
        }

        // Background gradient
        viz3d::drawGradientBackground(target, width, height);

        // Grid and axes
        viz3d::drawGrid(target, cam, minB, maxB, 10, width, height);
        float axisLen = (maxB - minB).length() * 0.15f;
        viz3d::drawAxes(target, cam, minB, axisLen, width, height, font_loaded ? &font : nullptr);

        // Compute depth range for proper rendering
        float minDepth = cam.getDepth(cam.center) - (maxB - minB).length();
        float maxDepth = cam.getDepth(cam.center) + (maxB - minB).length();

        // Draw obstacles
        for (const auto& obs : obstacle_map.getObstacles()) {
            if (obs.dimensions() >= 3) {
                viz3d::Vec3 pos(obs.center(0), obs.center(1), obs.center(2));
                viz3d::drawCircle3D(target, cam, pos, obs.radius,
                                    viz3d::colors::obstacle(), width, height,
                                    true, minDepth, maxDepth);
            }
        }

        // Draw samples (thin, transparent)
        if (show_samples && !current.samples.empty()) {
            sf::Color sampleColor = Colors::samples();
            for (const auto& sample : current.samples) {
                for (size_t i = 0; i + 1 < sample.nodes.size(); ++i) {
                    if (sample.nodes[i].position.size() >= 3 &&
                        sample.nodes[i+1].position.size() >= 3) {
                        viz3d::Vec3 p1(sample.nodes[i].position(0),
                                       sample.nodes[i].position(1),
                                       sample.nodes[i].position(2));
                        viz3d::Vec3 p2(sample.nodes[i+1].position(0),
                                       sample.nodes[i+1].position(1),
                                       sample.nodes[i+1].position(2));
                        viz3d::drawLine3D(target, cam, p1, p2, 1.0f, sampleColor, width, height);
                    }
                }
            }
        }

        // Draw previous trajectories with gradient
        if (show_all_means && current_iter > 0) {
            for (int i = 0; i < current_iter; ++i) {
                const auto& traj = history.iterations[i].mean_trajectory;
                float t = (float)i / current_iter;
                uint8_t alpha = static_cast<uint8_t>(50 + 150 * t);
                sf::Color col(150, 180, 255, alpha);  // Light blue gradient

                for (size_t j = 0; j + 1 < traj.nodes.size(); ++j) {
                    if (traj.nodes[j].position.size() >= 3 &&
                        traj.nodes[j+1].position.size() >= 3) {
                        viz3d::Vec3 p1(traj.nodes[j].position(0),
                                       traj.nodes[j].position(1),
                                       traj.nodes[j].position(2));
                        viz3d::Vec3 p2(traj.nodes[j+1].position(0),
                                       traj.nodes[j+1].position(1),
                                       traj.nodes[j+1].position(2));
                        viz3d::drawLine3D(target, cam, p1, p2, 2.0f, col, width, height);
                    }
                }
            }
        }

        // Draw current trajectory
        if (isValidTrajectory3D(mean_traj)) {
            for (size_t i = 0; i + 1 < mean_traj.nodes.size(); ++i) {
                viz3d::Vec3 p1(mean_traj.nodes[i].position(0),
                               mean_traj.nodes[i].position(1),
                               mean_traj.nodes[i].position(2));
                viz3d::Vec3 p2(mean_traj.nodes[i+1].position(0),
                               mean_traj.nodes[i+1].position(1),
                               mean_traj.nodes[i+1].position(2));
                viz3d::drawLine3D(target, cam, p1, p2, 3.0f, Colors::current(), width, height);
            }
        }

        // Draw collision spheres if enabled
        if (show_collision_spheres) {
            // Count collision status for legend
            int collision_count = 0, near_count = 0, safe_count = 0;
            
            for (size_t i = 0; i < mean_traj.nodes.size(); ++i) {
                const auto& node = mean_traj.nodes[i];
                if (node.position.size() < 3) continue;
                
                viz3d::Vec3 pos(node.position(0), node.position(1), node.position(2));
                float sdf = calculateSDF3D(pos, obstacle_map.getObstacles());
                float effective_sdf = sdf - node.radius;
                int status = getCollisionStatus(effective_sdf, collision_threshold);
                
                if (status == 2) collision_count++;
                else if (status == 1) near_count++;
                else safe_count++;
                
                sf::Color col = getCollisionColor(status);
                col.a = (status == 2) ? 230 : (status == 1) ? 200 : 180;
                
                viz3d::drawCircle3D(target, cam, pos, node.radius, col,
                                    width, height, false, minDepth, maxDepth, true);
            }
            
            // Draw collision status box
            if (font_loaded) {
                float sx = 15.0f;
                float sy = height - 110.0f;
                target.draw(createBox(sx, sy, 175.0f, 80.0f,
                                      sf::Color(248, 250, 255, 248), Colors::legendBorder()));
                
                std::ostringstream ss1, ss2, ss3;
                ss1 << "Collision: " << collision_count;
                ss2 << "Near: " << near_count;
                ss3 << "Safe: " << safe_count;
                
                sf::Text t1(font, ss1.str(), FontSize::stats);
                t1.setFillColor(collision_count > 0 ? Colors::inCollision() : Colors::text());
                t1.setPosition({sx + 12.0f, sy + 12.0f});
                target.draw(t1);
                
                sf::Text t2(font, ss2.str(), FontSize::stats);
                t2.setFillColor(near_count > 0 ? Colors::nearCollision() : Colors::text());
                t2.setPosition({sx + 12.0f, sy + 34.0f});
                target.draw(t2);
                
                sf::Text t3(font, ss3.str(), FontSize::stats);
                t3.setFillColor(Colors::safe());
                t3.setPosition({sx + 12.0f, sy + 56.0f});
                target.draw(t3);
            }
        } else {
            // Draw waypoints (only when not showing collision spheres)
            for (size_t i = 1; i + 1 < mean_traj.nodes.size(); ++i) {
                viz3d::Vec3 pos(mean_traj.nodes[i].position(0),
                                mean_traj.nodes[i].position(1),
                                mean_traj.nodes[i].position(2));
                viz3d::drawCircle3D(target, cam, pos, 3.0f, Colors::current(),
                                    width, height, false, minDepth, maxDepth);
            }
        }

        // Draw start and goal markers
        if (!mean_traj.nodes.empty()) {
            const auto& start_node = mean_traj.nodes[mean_traj.start_index];
            const auto& goal_node = mean_traj.nodes[mean_traj.goal_index];

            viz3d::Vec3 start_pos(start_node.position(0), start_node.position(1), start_node.position(2));
            viz3d::Vec3 goal_pos(goal_node.position(0), goal_node.position(1), goal_node.position(2));

            viz3d::drawCircle3D(target, cam, start_pos, 8.0f, Colors::start(),
                                width, height, true, minDepth, maxDepth);
            viz3d::drawCircle3D(target, cam, goal_pos, 8.0f, Colors::goal(),
                                width, height, true, minDepth, maxDepth);
        }

        // Draw UI overlay
        if (font_loaded) {
            // Title with iteration info and costs
            std::ostringstream title_ss;
            title_ss << "Iter " << (current_iter + 1) << "/" << history.iterations.size()
                     << "  |  Total: " << std::fixed << std::setprecision(2) << current.total_cost
                     << "  |  Collision: " << std::setprecision(2) << current.collision_cost
                     << "  |  Smooth: " << std::setprecision(2) << current.smoothness_cost;
            if (show_collision_spheres) {
                title_ss << "  [Collision View]";
            }
            sf::Text title(font, title_ss.str(), FontSize::title);
            title.setFillColor(Colors::text());
            title.setStyle(sf::Text::Bold);
            title.setPosition({15.0f, 12.0f});
            target.draw(title);

            // Legend - expanded when showing collision spheres
            float lx = width - (show_collision_spheres ? 195.0f : 145.0f);
            float ly = 15.0f;
            float legend_h = show_collision_spheres ? 175.0f : 100.0f;
            float legend_w = show_collision_spheres ? 180.0f : 130.0f;
            target.draw(createBox(lx, ly, legend_w, legend_h, Colors::legendBg(), Colors::legendBorder()));

            float item_y = ly + 10.0f;
            
            if (show_collision_spheres) {
                drawLegendMarker(target, font, lx + 8.0f, item_y, "Safe", Colors::safe(), false);
                item_y += 22.0f;
                drawLegendMarker(target, font, lx + 8.0f, item_y, "Near collision", Colors::nearCollision(), false);
                item_y += 22.0f;
                drawLegendMarker(target, font, lx + 8.0f, item_y, "In collision", Colors::inCollision(), false);
                item_y += 22.0f;
                drawLegendMarker(target, font, lx + 8.0f, item_y, "Obstacles", viz3d::colors::obstacle(), false);
                item_y += 22.0f;
            }
            
            drawLegendEntry(target, font, lx + 8.0f, item_y, "Current", Colors::current(), 2.5f);
            item_y += 22.0f;
            drawLegendMarker(target, font, lx + 8.0f, item_y, "Start", Colors::start(), false);
            item_y += 22.0f;
            drawLegendMarker(target, font, lx + 8.0f, item_y, "Goal", Colors::goal(), false);

            // Hints
            if (show_hints) {
                sf::Text hint(font, "Drag: Rotate | Scroll: Zoom | Space: Play | R: Auto-rotate",
                             FontSize::hint);
                hint.setFillColor(Colors::hint());
                hint.setPosition({15.0f, height - 28.0f});
                target.draw(hint);
            }
        }
    }

    /**
     * @brief Draw 3D final frame with green optimized trajectory
     */
    template<typename RenderTarget>
    void draw3DFinalTrajectoryFrame(RenderTarget& target,
                                     const ObstacleMap& obstacle_map,
                                     const OptimizationHistory& history,
                                     const viz3d::Camera3D& cam,
                                     const sf::Font& font, bool font_loaded,
                                     unsigned int width, unsigned int height,
                                     const viz3d::Vec3& minB, const viz3d::Vec3& maxB) {
        if (history.iterations.empty()) return;

        const auto& final_iter = history.iterations.back();
        const auto& mean_traj = final_iter.mean_trajectory;

        if (mean_traj.nodes.empty() || mean_traj.nodes[0].position.size() < 3) return;

        // Background gradient
        viz3d::drawGradientBackground(target, width, height);

        // Grid and axes
        viz3d::drawGrid(target, cam, minB, maxB, 10, width, height);
        float axisLen = (maxB - minB).length() * 0.15f;
        viz3d::drawAxes(target, cam, minB, axisLen, width, height, font_loaded ? &font : nullptr);

        // Compute depth range
        float minDepth = cam.getDepth(cam.center) - (maxB - minB).length();
        float maxDepth = cam.getDepth(cam.center) + (maxB - minB).length();

        // Draw obstacles
        for (const auto& obs : obstacle_map.getObstacles()) {
            if (obs.dimensions() >= 3) {
                viz3d::Vec3 pos(obs.center(0), obs.center(1), obs.center(2));
                viz3d::drawCircle3D(target, cam, pos, obs.radius,
                                    viz3d::colors::obstacle(), width, height,
                                    true, minDepth, maxDepth);
            }
        }

        // Draw optimized trajectory in GREEN
        sf::Color green = optimizedGreen();
        if (isValidTrajectory3D(mean_traj)) {
            for (size_t i = 0; i + 1 < mean_traj.nodes.size(); ++i) {
                viz3d::Vec3 p1(mean_traj.nodes[i].position(0),
                               mean_traj.nodes[i].position(1),
                               mean_traj.nodes[i].position(2));
                viz3d::Vec3 p2(mean_traj.nodes[i+1].position(0),
                               mean_traj.nodes[i+1].position(1),
                               mean_traj.nodes[i+1].position(2));
                viz3d::drawLine3D(target, cam, p1, p2, 4.0f, green, width, height);
            }

            // Waypoints in green
            for (size_t i = 1; i + 1 < mean_traj.nodes.size(); ++i) {
                viz3d::Vec3 pos(mean_traj.nodes[i].position(0),
                                mean_traj.nodes[i].position(1),
                                mean_traj.nodes[i].position(2));
                viz3d::drawCircle3D(target, cam, pos, 4.0f, green,
                                    width, height, false, minDepth, maxDepth);
            }
        }

        // Draw start and goal markers
        if (!mean_traj.nodes.empty()) {
            const auto& start_node = mean_traj.nodes[mean_traj.start_index];
            const auto& goal_node = mean_traj.nodes[mean_traj.goal_index];

            viz3d::Vec3 start_pos(start_node.position(0), start_node.position(1), start_node.position(2));
            viz3d::Vec3 goal_pos(goal_node.position(0), goal_node.position(1), goal_node.position(2));

            viz3d::drawCircle3D(target, cam, start_pos, 8.0f, Colors::start(),
                                width, height, true, minDepth, maxDepth);
            viz3d::drawCircle3D(target, cam, goal_pos, 8.0f, Colors::goal(),
                                width, height, true, minDepth, maxDepth);
        }

        // Draw UI overlay
        if (font_loaded) {
            // Title - "Optimized Trajectory" in green
            std::ostringstream title_ss;
            title_ss << "Optimized Trajectory  |  Final Cost: " << std::fixed << std::setprecision(2) << final_iter.total_cost;
            sf::Text title(font, title_ss.str(), FontSize::title);
            title.setFillColor(green);
            title.setStyle(sf::Text::Bold);
            title.setPosition({15.0f, 12.0f});
            target.draw(title);

            // Legend
            float lx = width - 145.0f;
            float ly = 15.0f;
            target.draw(createBox(lx, ly, 130.0f, 100.0f, Colors::legendBg(), Colors::legendBorder()));

            float item_y = ly + 10.0f;
            drawLegendEntry(target, font, lx + 8.0f, item_y, "Optimized", green, 3.0f);
            item_y += 22.0f;
            drawLegendMarker(target, font, lx + 8.0f, item_y, "Start", Colors::start(), false);
            item_y += 22.0f;
            drawLegendMarker(target, font, lx + 8.0f, item_y, "Goal", Colors::goal(), false);
        }
    }

    void saveTrajectory3DImage(const ObstacleMap& obstacle_map,
                                const OptimizationHistory& history,
                                int iter, bool show_samples, bool show_all_means,
                                const viz3d::Camera3D& cam,
                                const sf::Font& font, bool font_loaded,
                                int counter, float scale) {
        unsigned int width_3d = window_width_ * 3 / 2;
        unsigned int height_3d = window_height_;

        unsigned int w = static_cast<unsigned int>(width_3d * scale);
        unsigned int h = static_cast<unsigned int>(height_3d * scale);

        sf::RenderTexture rt;
        if (!rt.resize({w, h})) {
            std::cerr << "Failed to create render texture\n";
            return;
        }

        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)width_3d, (float)height_3d}));
        view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
        rt.setView(view);

        viz3d::Vec3 minB, maxB;
        computeHistoryBounds3D(history, obstacle_map.getObstacles(), minB, maxB);

        rt.clear(sf::Color::White);
        draw3DTrajectoryFigure(rt, obstacle_map, history, iter, show_samples, show_all_means,
                               cam, font, font_loaded, false, width_3d, height_3d, minB, maxB);
        rt.display();

        std::string suffix = (scale > 1.0f) ? "_highres" : "";
        std::string fname = output_prefix_ + "_trajectory3d_iter" + std::to_string(iter + 1) + suffix + ".png";

        if (rt.getTexture().copyToImage().saveToFile(fname)) {
            std::cout << "Saved: " << fname << " (" << w << "x" << h << ")\n";
        }
    }

    void saveAnimation3DGIF(const ObstacleMap& obstacle_map,
                            const OptimizationHistory& history,
                            bool show_samples, bool show_all_means,
                            viz3d::Camera3D cam,  // Copy for rotation
                            const sf::Font& font, bool font_loaded,
                            int delay_ms = 100,
                            int collision_frames = 5,      // Frames with collision view at start/end
                            float collision_threshold = 10.0f,
                            int final_green_frames = 8) {
        if (history.iterations.empty()) return;

        std::cout << "Saving 3D animation frames (with collision visualization)...\n";

        unsigned int width_3d = window_width_ * 3 / 2;
        unsigned int height_3d = window_height_;

        unsigned int w = static_cast<unsigned int>(width_3d * GIF_SCALE);
        unsigned int h = static_cast<unsigned int>(height_3d * GIF_SCALE);

        sf::RenderTexture rt;
        if (!rt.resize({w, h})) {
            std::cerr << "Failed to create render texture\n";
            return;
        }

        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)width_3d, (float)height_3d}));
        view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
        rt.setView(view);

        viz3d::Vec3 minB, maxB;
        computeHistoryBounds3D(history, obstacle_map.getObstacles(), minB, maxB);

        std::string frame_dir = output_prefix_ + "_3d_frames";
        std::system(("mkdir -p " + frame_dir).c_str());

        int frame_num = 0;
        
        // Phase 1: Initial trajectory with collision visualization (static, first iteration)
        std::cout << "  Phase 1: Initial collision view (" << collision_frames << " frames)\n";
        for (int f = 0; f < collision_frames; ++f) {
            rt.clear(sf::Color::White);
            draw3DTrajectoryFigure(rt, obstacle_map, history, 0, false, false,
                                   cam, font, font_loaded, false, width_3d, height_3d, minB, maxB,
                                   true, collision_threshold);  // show_collision_spheres = true
            rt.display();

            std::ostringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << frame_num++ << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());
            
            // Slight rotation
            cam.rotate(0.015f, 0);
        }

        // Phase 2: Optimization evolution (normal trajectory view)
        std::cout << "  Phase 2: Optimization evolution (" << history.iterations.size() << " frames)\n";
        for (size_t i = 0; i < history.iterations.size(); ++i) {
            rt.clear(sf::Color::White);
            draw3DTrajectoryFigure(rt, obstacle_map, history, i, show_samples, show_all_means,
                                   cam, font, font_loaded, false, width_3d, height_3d, minB, maxB,
                                   false, collision_threshold);  // show_collision_spheres = false
            rt.display();

            std::ostringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << frame_num++ << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());

            // Slight rotation for visual interest
            cam.rotate(0.02f, 0);

            if ((i + 1) % 10 == 0) {
                std::cout << "    Iteration " << (i + 1) << "/" << history.iterations.size() << "\n";
            }
        }

        // Phase 3: Final trajectory with collision visualization
        std::cout << "  Phase 3: Final collision view (" << collision_frames << " frames)\n";
        int last_iter = history.iterations.size() - 1;
        for (int f = 0; f < collision_frames; ++f) {
            rt.clear(sf::Color::White);
            draw3DTrajectoryFigure(rt, obstacle_map, history, last_iter, false, false,
                                   cam, font, font_loaded, false, width_3d, height_3d, minB, maxB,
                                   true, collision_threshold);  // show_collision_spheres = true
            rt.display();

            std::ostringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << frame_num++ << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());
            
            // Slight rotation
            cam.rotate(0.015f, 0);
        }

        // Phase 4: Green optimized trajectory
        std::cout << "  Phase 4: Green optimized trajectory (" << final_green_frames << " frames)\n";
        for (int f = 0; f < final_green_frames; ++f) {
            rt.clear(sf::Color::White);
            draw3DFinalTrajectoryFrame(rt, obstacle_map, history, cam,
                                       font, font_loaded, width_3d, height_3d, minB, maxB);
            rt.display();

            std::ostringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << frame_num++ << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());
            
            // Slight rotation
            cam.rotate(0.015f, 0);
        }
        
        std::cout << "  Total frames: " << frame_num << "\n";

        std::string gif_name = output_prefix_ + "_3d_animation.gif";
        
        // Try multiple methods to create GIF
        bool success = false;
        
        // Method 1: ImageMagick 7+ (magick)
        std::ostringstream cmd1;
        cmd1 << "magick -delay " << (delay_ms / 10) << " -loop 0 "
             << frame_dir << "/frame_*.png " << gif_name << " 2>/dev/null";
        
        std::cout << "Creating GIF (trying magick)...\n";
        if (std::system(cmd1.str().c_str()) == 0) {
            success = true;
        } else {
            // Method 2: ImageMagick 6 (convert)
            std::ostringstream cmd2;
            cmd2 << "convert -delay " << (delay_ms / 10) << " -loop 0 "
                 << frame_dir << "/frame_*.png " << gif_name << " 2>/dev/null";
            
            std::cout << "Trying convert...\n";
            if (std::system(cmd2.str().c_str()) == 0) {
                success = true;
            } else {
                // Method 3: ffmpeg
                std::ostringstream cmd3;
                cmd3 << "ffmpeg -y -framerate " << (1000 / delay_ms) << " -i "
                     << frame_dir << "/frame_%04d.png "
                     << "-vf \"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" "
                     << gif_name << " 2>/dev/null";
                
                std::cout << "Trying ffmpeg...\n";
                if (std::system(cmd3.str().c_str()) == 0) {
                    success = true;
                }
            }
        }

        if (success) {
            std::cout << "Saved: " << gif_name << "\n";
            std::system(("rm -rf " + frame_dir).c_str());
        } else {
            std::cerr << "GIF creation failed. Install one of:\n"
                      << "  sudo apt install imagemagick\n"
                      << "  sudo apt install ffmpeg\n"
                      << "Frames saved in: " << frame_dir << "/\n";
        }
    }

    // === Save Functions ===

    void saveTrajectoryImage(const ObstacleMap& obstacle_map,
                             const OptimizationHistory& history,
                             int iter, bool show_samples, bool show_all_means,
                             const sf::Font& font, bool font_loaded,
                             int counter, float scale) {
        sf::RenderTexture rt;
        if (!createRenderTexture(rt, scale)) return;

        rt.clear(sf::Color::White);
        drawTrajectoryFigure(rt, obstacle_map, history, iter, show_samples, show_all_means,
                            font, font_loaded, false, window_width_, window_height_);
        rt.display();

        std::string suffix = (scale > 1.0f) ? "_highres" : "";
        std::string fname = output_prefix_ + "_trajectory_iter" + std::to_string(iter + 1) + suffix + ".png";
        
        unsigned int w = static_cast<unsigned int>(window_width_ * scale);
        unsigned int h = static_cast<unsigned int>(window_height_ * scale);
        
        if (rt.getTexture().copyToImage().saveToFile(fname)) {
            std::cout << "Saved: " << fname << " (" << w << "x" << h << ")\n";
        }
    }

    void saveCostImage(const OptimizationHistory& history,
                       const sf::Font& font, bool font_loaded,
                       int counter, float scale) {
        sf::RenderTexture rt;
        if (!createRenderTexture(rt, scale)) return;

        rt.clear(sf::Color::White);
        drawCostFigure(rt, history, font, font_loaded, false,
                      window_width_, window_height_);
        rt.display();

        std::string suffix = (scale > 1.0f) ? "_highres" : "";
        std::string fname = output_prefix_ + "_cost_convergence" + suffix + ".png";
        
        unsigned int w = static_cast<unsigned int>(window_width_ * scale);
        unsigned int h = static_cast<unsigned int>(window_height_ * scale);
        
        if (rt.getTexture().copyToImage().saveToFile(fname)) {
            std::cout << "Saved: " << fname << " (" << w << "x" << h << ")\n";
        }
    }
};

// =============================================================================
// STOMP Visualization Adapter
// =============================================================================
// 
// Converts STOMP planner data structures to OptimizationHistory format
// for use with OptimizationVisualizer.
//
// Usage:
//   StompMotionPlanner planner(collision_task);
//   planner.initialize(config);
//   planner.solve();
//   
//   OptimizationHistory history = StompAdapter::convert(planner);
//   OptimizationVisualizer viz;
//   viz.showTrajectoryEvolution(obstacle_map, history, "STOMP Optimization");
// =============================================================================

namespace StompAdapter {

/**
 * @brief Data for a single STOMP optimization iteration
 * 
 * This structure mirrors what StompMotionPlanner stores internally.
 * If you're using StompMotionPlanner directly, you can use its native types.
 */
struct StompIterationData {
    std::vector<Trajectory> samples;    // Noisy rollouts for this iteration
    Trajectory mean_trajectory;          // Mean/updated trajectory
    float cost = 0.0f;
    float collision_cost = 0.0f;
    float smoothness_cost = 0.0f;
    int iteration_number = 0;
};

/**
 * @brief Complete STOMP optimization history
 */
struct StompOptimizationHistory {
    std::vector<StompIterationData> iterations;
    
    size_t size() const { return iterations.size(); }
    bool empty() const { return iterations.empty(); }
};

/**
 * @brief Convert STOMP iteration data to visualization IterationData
 * 
 * This function works with pce::StompIterationData which may or may not have
 * collision_cost and smoothness_cost fields depending on the version.
 * If those fields don't exist, costs default to 0.
 * 
 * For the most reliable cost display, access the planner's internal history directly.
 */
template<typename StompIterType>
inline IterationData convertIteration(const StompIterType& stomp_iter) {
    IterationData iter_data;
    
    // Core fields that always exist
    iter_data.iteration = stomp_iter.iteration_number;
    iter_data.mean_trajectory = stomp_iter.mean_trajectory;
    iter_data.samples = stomp_iter.samples;
    iter_data.total_cost = stomp_iter.cost;
    
    // Initialize cost components to 0 - they may be overwritten if fields exist
    // Note: If pce::StompIterationData doesn't have these fields, don't try to access them
    iter_data.collision_cost = 0.0f;
    iter_data.smoothness_cost = 0.0f;
    
    return iter_data;
}

/**
 * @brief Specialized conversion for StompAdapter::StompIterationData (has all cost fields)
 */
inline IterationData convertIterationFull(const StompIterationData& stomp_iter) {
    IterationData iter_data;
    
    iter_data.iteration = stomp_iter.iteration_number;
    iter_data.mean_trajectory = stomp_iter.mean_trajectory;
    iter_data.samples = stomp_iter.samples;
    iter_data.total_cost = stomp_iter.cost;
    iter_data.collision_cost = stomp_iter.collision_cost;
    iter_data.smoothness_cost = stomp_iter.smoothness_cost;
    
    return iter_data;
}

/**
 * @brief Convert complete STOMP history to OptimizationHistory
 * 
 * Template version works with any type that has an `iterations` member
 * containing elements compatible with convertIteration().
 * 
 * Compatible with:
 * - StompAdapter::StompOptimizationHistory
 * - pce::StompOptimizationHistory
 * 
 * @param stomp_history The STOMP planner's optimization history
 * @return OptimizationHistory compatible with OptimizationVisualizer
 */
template<typename StompHistoryType>
inline OptimizationHistory convert(const StompHistoryType& stomp_history) {
    OptimizationHistory vis_history;
    
    for (const auto& stomp_iter : stomp_history.iterations) {
        vis_history.addIteration(convertIteration(stomp_iter));
    }
    
    return vis_history;
}

/**
 * @brief Convert trajectory history (mean only) to OptimizationHistory
 * 
 * Use this when you only have the mean trajectories without samples.
 * 
 * @param trajectories Vector of mean trajectories per iteration
 * @param costs Optional vector of costs per iteration
 * @return OptimizationHistory compatible with OptimizationVisualizer
 */
inline OptimizationHistory convertTrajectoryHistory(
    const std::vector<Trajectory>& trajectories,
    const std::vector<float>& costs = {}) 
{
    OptimizationHistory vis_history;
    
    for (size_t i = 0; i < trajectories.size(); ++i) {
        IterationData iter_data;
        iter_data.iteration = static_cast<int>(i);
        iter_data.mean_trajectory = trajectories[i];
        iter_data.total_cost = (i < costs.size()) ? costs[i] : 0.0f;
        // samples left empty - visualization will just show means
        vis_history.addIteration(iter_data);
    }
    
    return vis_history;
}

/**
 * @brief Builder class for constructing OptimizationHistory from STOMP data
 * 
 * Useful when collecting data during optimization:
 * 
 *   StompAdapter::HistoryBuilder builder;
 *   
 *   // In your optimization loop:
 *   builder.startIteration(iter_num);
 *   for (each rollout) {
 *       builder.addSample(noisy_trajectory);
 *   }
 *   builder.finishIteration(mean_trajectory, cost);
 *   
 *   // After optimization:
 *   OptimizationHistory history = builder.build();
 */
class HistoryBuilder {
public:
    HistoryBuilder() = default;
    
    /// Start a new iteration
    void startIteration(int iteration_number) {
        current_iteration_.iteration_number = iteration_number;
        current_iteration_.samples.clear();
    }
    
    /// Add a sample (noisy rollout) to current iteration
    void addSample(const Trajectory& sample) {
        current_iteration_.samples.push_back(sample);
    }
    
    /// Add multiple samples at once
    void addSamples(const std::vector<Trajectory>& samples) {
        current_iteration_.samples.insert(
            current_iteration_.samples.end(), 
            samples.begin(), 
            samples.end()
        );
    }
    
    /// Finish current iteration with mean trajectory and costs
    void finishIteration(const Trajectory& mean_trajectory, 
                         float total_cost,
                         float collision_cost = 0.0f,
                         float smoothness_cost = 0.0f) {
        current_iteration_.mean_trajectory = mean_trajectory;
        current_iteration_.cost = total_cost;
        current_iteration_.collision_cost = collision_cost;
        current_iteration_.smoothness_cost = smoothness_cost;
        
        stomp_history_.iterations.push_back(std::move(current_iteration_));
        current_iteration_ = StompIterationData{};
    }
    
    /// Build the final OptimizationHistory
    OptimizationHistory build() const {
        return convert(stomp_history_);
    }
    
    /// Get the raw STOMP history
    const StompOptimizationHistory& getStompHistory() const {
        return stomp_history_;
    }
    
    /// Clear all data
    void clear() {
        stomp_history_.iterations.clear();
        current_iteration_ = StompIterationData{};
    }
    
    /// Get number of iterations collected
    size_t size() const { return stomp_history_.size(); }

private:
    StompOptimizationHistory stomp_history_;
    StompIterationData current_iteration_;
};

} // namespace StompAdapter