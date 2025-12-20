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
#include "IterationData.h"
#include "ObstacleMap.h"

/**
 * @brief Publication-quality visualization for optimization results
 * 
 * Designed for tier-1 robotics journal submissions (pedestrian safety, TRO, pedestrian safety, pedestrian safety, pedestrian safety, etc.)
 * 
 * Color scheme follows matplotlib default for consistency with Python plots:
 * - Current trajectory: #1f77b4 (blue)
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
        drawTrajectoryFigure(rt, obstacle_map, history, 0, false, false, font, font_loaded, false);
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
        sf::RenderWindow window(sf::VideoMode({window_width_, window_height_}), title,
                                sf::Style::Titlebar | sf::Style::Close);
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
                                show_samples, show_all_means, font, font_loaded, true);
            window.display();
        }
    }

    /**
     * @brief Interactive cost convergence plot viewer
     */
    void showCostPlot(const OptimizationHistory& history,
                      const std::string& title = "Cost Convergence") {
        sf::RenderWindow window(sf::VideoMode({window_width_, window_height_}), title,
                                sf::Style::Titlebar | sf::Style::Close);
        window.setFramerateLimit(60);

        sf::Font font;
        bool font_loaded = loadFont(font);

        int save_counter = 0;
        bool show_collision = true;
        bool show_smoothness = true;

        printCostControls();

        while (window.isOpen()) {
            while (const auto event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>()) window.close();

                if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                    switch (key->code) {
                        case sf::Keyboard::Key::Escape: window.close(); break;
                        case sf::Keyboard::Key::C: show_collision = !show_collision; break;
                        case sf::Keyboard::Key::O: show_smoothness = !show_smoothness; break;
                        case sf::Keyboard::Key::S:
                            saveCostImage(history, font, font_loaded, show_collision, show_smoothness,
                                         save_counter++, 1.0f);
                            break;
                        case sf::Keyboard::Key::P:
                            saveCostImage(history, font, font_loaded, show_collision, show_smoothness,
                                         save_counter++, HIGHRES_SCALE);
                            break;
                        default: break;
                    }
                }
            }

            window.clear(sf::Color::White);
            drawCostFigure(window, history, font, font_loaded, show_collision, show_smoothness, true);
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
     * @brief Save animation as GIF (requires ImageMagick)
     */
    void saveAnimationGIF(const ObstacleMap& obstacle_map,
                          const OptimizationHistory& history,
                          bool show_samples, bool show_all_means,
                          const sf::Font& font, bool font_loaded,
                          int delay_ms = 100) {
        if (history.iterations.empty()) return;

        std::cout << "Saving animation frames...\n";

        sf::RenderTexture rt;
        if (!createRenderTexture(rt, GIF_SCALE)) return;

        std::string frame_dir = output_prefix_ + "_frames";
        std::system(("mkdir -p " + frame_dir).c_str());

        for (size_t i = 0; i < history.iterations.size(); ++i) {
            rt.clear(sf::Color::White);
            drawTrajectoryFigure(rt, obstacle_map, history, i, show_samples, show_all_means,
                                font, font_loaded, false);
            rt.display();

            std::ostringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << i << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());

            if ((i + 1) % 10 == 0 || i == history.iterations.size() - 1) {
                std::cout << "  Frame " << (i + 1) << "/" << history.iterations.size() << "\n";
            }
        }

        std::string gif_name = output_prefix_ + "_animation.gif";
        std::ostringstream cmd;
        cmd << "convert -delay " << (delay_ms / 10) << " -loop 0 "
            << frame_dir << "/frame_*.png " << gif_name;

        std::cout << "Creating GIF...\n";
        if (std::system(cmd.str().c_str()) == 0) {
            std::cout << "Saved: " << gif_name << "\n";
            std::system(("rm -rf " + frame_dir).c_str());
        } else {
            std::cerr << "ImageMagick failed. Install with: sudo apt install imagemagick\n"
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

    // === Publication Color Palette (matplotlib default) ===
    struct Colors {
        // Trajectory colors
        static constexpr auto current()     { return sf::Color(31, 119, 180); }      // #1f77b4
        static constexpr auto previous()    { return sf::Color(174, 199, 232); }     // #aec7e8
        static constexpr auto samples()     { return sf::Color(31, 119, 180, 25); }  // transparent blue
        
        // Marker colors
        static constexpr auto start()       { return sf::Color(44, 160, 44); }       // #2ca02c
        static constexpr auto goal()        { return sf::Color(255, 127, 14); }      // #ff7f0e
        
        // Cost curve colors
        static constexpr auto total()       { return sf::Color(31, 119, 180); }      // #1f77b4
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
    };

    // === Typography (publication-quality sizes) ===
    struct FontSize {
        static constexpr unsigned int title = 18;
        static constexpr unsigned int axisLabel = 15;
        static constexpr unsigned int tickLabel = 12;
        static constexpr unsigned int legend = 13;
        static constexpr unsigned int stats = 12;
        static constexpr unsigned int hint = 10;
    };

    // === Layout (margins and spacing) ===
    struct Layout {
        static constexpr float marginLeft = 70.0f;
        static constexpr float marginRight = 30.0f;
        static constexpr float marginTop = 55.0f;
        static constexpr float marginBottom = 60.0f;
        static constexpr float legendPadding = 10.0f;
        static constexpr float itemSpacing = 22.0f;
    };

    // === Line/Marker Sizes ===
    struct Sizes {
        static constexpr float currentLine = 3.0f;
        static constexpr float previousLine = 2.0f;
        static constexpr float sampleLine = 1.0f;
        static constexpr float costLine = 2.5f;
        static constexpr float costLineSecondary = 2.0f;
        static constexpr float waypoint = 4.0f;
        static constexpr float startMarker = 10.0f;
        static constexpr float goalMarker = 12.0f;
        static constexpr float axisBorder = 1.5f;
        static constexpr float legendBorder = 1.0f;
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
                  << "C:   Toggle collision cost\n"
                  << "O:   Toggle smoothness cost\n"
                  << "S:   Save PNG (1x)\n"
                  << "P:   Save PNG (4x high-res)\n"
                  << "ESC: Exit\n"
                  << "==========================\n\n";
    }

    void drawLine(sf::RenderTarget& target, sf::Vector2f p1, sf::Vector2f p2,
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

    void drawLegendLine(sf::RenderTarget& target, float x, float y, float width,
                        sf::Color color, float lineWidth = 3.0f) {
        drawLine(target, {x, y + 8.0f}, {x + width, y + 8.0f}, lineWidth, color);
    }

    void drawLegendMarker(sf::RenderTarget& target, const sf::Font& font,
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

    void drawLegendEntry(sf::RenderTarget& target, const sf::Font& font,
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

    // === Main Drawing Functions ===

    void drawTrajectoryFigure(sf::RenderTarget& target,
                               const ObstacleMap& obstacle_map,
                               const OptimizationHistory& history,
                               int current_iter,
                               bool show_samples, bool show_all_means,
                               const sf::Font& font, bool font_loaded,
                               bool show_hints) {
        if (history.iterations.empty()) return;

        const auto& current = history.iterations[current_iter];
        float map_w = obstacle_map.getMapWidth();
        float map_h = obstacle_map.getMapHeight();

        // Compute plot area
        float plot_w = window_width_ - Layout::marginLeft - Layout::marginRight;
        float plot_h = window_height_ - Layout::marginTop - Layout::marginBottom;
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
        for (const auto& obs : obstacle_map.getObstacles()) {
            if (obs.dimensions() >= 2) {
                sf::Vector2f pos = transform(obs.center(0), obs.center(1));
                float r = obs.radius * scale;
                sf::CircleShape circle(r);
                circle.setPosition({pos.x - r, pos.y - r});
                circle.setFillColor(Colors::obstacle());
                circle.setOutlineColor(Colors::obstacleBorder());
                circle.setOutlineThickness(1.5f);
                target.draw(circle);
            }
        }

        // Samples
        if (show_samples && !current.samples.empty()) {
            for (const auto& sample : current.samples) {
                for (size_t i = 0; i + 1 < sample.nodes.size(); ++i) {
                    sf::Vector2f p1 = transform(sample.nodes[i].position(0), sample.nodes[i].position(1));
                    sf::Vector2f p2 = transform(sample.nodes[i+1].position(0), sample.nodes[i+1].position(1));
                    drawLine(target, p1, p2, Sizes::sampleLine, Colors::samples());
                }
            }
        }

        // Previous trajectories with gradient
        if (show_all_means && current_iter > 0) {
            for (int i = 0; i < current_iter; ++i) {
                const auto& traj = history.iterations[i].mean_trajectory;
                float t = (float)i / current_iter;
                uint8_t alpha = static_cast<uint8_t>(50 + 150 * t);
                sf::Color col(174, 199, 232, alpha);
                
                for (size_t j = 0; j + 1 < traj.nodes.size(); ++j) {
                    sf::Vector2f p1 = transform(traj.nodes[j].position(0), traj.nodes[j].position(1));
                    sf::Vector2f p2 = transform(traj.nodes[j+1].position(0), traj.nodes[j+1].position(1));
                    drawLine(target, p1, p2, Sizes::previousLine, col);
                }
            }
        }

        // Current trajectory
        const auto& mean_traj = current.mean_trajectory;
        for (size_t i = 0; i + 1 < mean_traj.nodes.size(); ++i) {
            sf::Vector2f p1 = transform(mean_traj.nodes[i].position(0), mean_traj.nodes[i].position(1));
            sf::Vector2f p2 = transform(mean_traj.nodes[i+1].position(0), mean_traj.nodes[i+1].position(1));
            drawLine(target, p1, p2, Sizes::currentLine, Colors::current());
        }

        // Waypoints
        for (size_t i = 1; i + 1 < mean_traj.nodes.size(); ++i) {
            sf::Vector2f pos = transform(mean_traj.nodes[i].position(0), mean_traj.nodes[i].position(1));
            sf::CircleShape marker(Sizes::waypoint);
            marker.setPosition({pos.x - Sizes::waypoint, pos.y - Sizes::waypoint});
            marker.setFillColor(Colors::current());
            target.draw(marker);
        }

        // Start and Goal markers
        if (!mean_traj.nodes.empty()) {
            sf::Vector2f start_pos = transform(mean_traj.nodes[mean_traj.start_index].position(0),
                                               mean_traj.nodes[mean_traj.start_index].position(1));
            sf::Vector2f goal_pos = transform(mean_traj.nodes[mean_traj.goal_index].position(0),
                                              mean_traj.nodes[mean_traj.goal_index].position(1));

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
                                 current.total_cost, show_hints);
        }
    }

    void drawTrajectoryLabels(sf::RenderTarget& target, const sf::Font& font,
                               float offset_x, float offset_y, float plot_w, float plot_h,
                               float map_w, float map_h, int current_iter, size_t total_iters,
                               float cost, bool show_hints) {
        // X-axis label
        sf::Text xlabel(font, "X Position", FontSize::axisLabel);
        xlabel.setFillColor(Colors::text());
        sf::FloatRect xb = xlabel.getLocalBounds();
        xlabel.setPosition({offset_x + plot_w / 2 - xb.size.x / 2, offset_y + plot_h + 35.0f});
        target.draw(xlabel);

        // Y-axis label
        sf::Text ylabel(font, "Y Position", FontSize::axisLabel);
        ylabel.setFillColor(Colors::text());
        ylabel.setRotation(sf::degrees(-90.0f));
        ylabel.setPosition({offset_x - 50.0f, offset_y + plot_h / 2 + 35.0f});
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
            xt.setPosition({sx - xtb.size.x / 2, offset_y + plot_h + 10.0f});
            target.draw(xt);

            sf::Text yt(font, std::to_string((int)data_y), FontSize::tickLabel);
            yt.setFillColor(Colors::text());
            sf::FloatRect ytb = yt.getLocalBounds();
            yt.setPosition({offset_x - ytb.size.x - 10.0f, sy - 7.0f});
            target.draw(yt);
        }

        // Title
        std::ostringstream title_ss;
        title_ss << "Iteration " << (current_iter + 1) << "/" << total_iters;
        sf::Text title(font, title_ss.str(), FontSize::title);
        title.setFillColor(Colors::text());
        title.setStyle(sf::Text::Bold);
        title.setPosition({offset_x, 18.0f});
        target.draw(title);

        // Cost display
        std::ostringstream cost_ss;
        cost_ss << "Cost: " << std::fixed << std::setprecision(2) << cost;
        sf::Text cost_text(font, cost_ss.str(), FontSize::axisLabel);
        cost_text.setFillColor(Colors::text());
        cost_text.setPosition({offset_x + plot_w - 130.0f, 20.0f});
        target.draw(cost_text);

        // Legend
        float lx = offset_x + plot_w - 150.0f;
        float ly = offset_y + 12.0f;
        
        target.draw(createBox(lx, ly, 140.0f, 100.0f, Colors::legendBg(), Colors::legendBorder()));

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
            hint.setPosition({12.0f, window_height_ - 22.0f});
            target.draw(hint);
        }
    }

    void drawCostFigure(sf::RenderTarget& target,
                        const OptimizationHistory& history,
                        const sf::Font& font, bool font_loaded,
                        bool show_collision, bool show_smoothness,
                        bool show_hints) {
        if (history.iterations.empty()) return;

        float plot_left = Layout::marginLeft + 10.0f;
        float plot_right = window_width_ - Layout::marginRight;
        float plot_top = Layout::marginTop;
        float plot_bottom = window_height_ - Layout::marginBottom;
        float plot_w = plot_right - plot_left;
        float plot_h = plot_bottom - plot_top;

        auto costs = history.getCostHistory();
        auto collision_costs = history.getCollisionCostHistory();
        auto smoothness_costs = history.getSmoothnessCostHistory();

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

        // Draw curves
        auto drawCurve = [&](const std::vector<float>& data, sf::Color color, float width) {
            if (data.size() < 2) return;
            for (size_t i = 0; i + 1 < data.size(); ++i) {
                float x1 = plot_left + (float)i / (data.size() - 1) * plot_w;
                float x2 = plot_left + (float)(i + 1) / (data.size() - 1) * plot_w;
                float y1 = plot_bottom - (data[i] - min_cost) / range * plot_h;
                float y2 = plot_bottom - (data[i + 1] - min_cost) / range * plot_h;
                drawLine(target, {x1, y1}, {x2, y2}, width, color);
            }
        };

        bool has_collision = show_collision && 
            std::any_of(collision_costs.begin(), collision_costs.end(), [](float c) { return c > 1e-6f; });
        bool has_smoothness = show_smoothness && 
            std::any_of(smoothness_costs.begin(), smoothness_costs.end(), [](float c) { return c > 1e-6f; });

        if (has_smoothness) drawCurve(smoothness_costs, Colors::smoothness(), Sizes::costLineSecondary);
        if (has_collision) drawCurve(collision_costs, Colors::collision(), Sizes::costLineSecondary);
        drawCurve(costs, Colors::total(), Sizes::costLine);

        if (font_loaded) {
            drawCostLabels(target, font, plot_left, plot_top, plot_w, plot_h,
                          min_cost, max_cost, range, costs, has_collision, has_smoothness, show_hints);
        }
    }

    void drawCostLabels(sf::RenderTarget& target, const sf::Font& font,
                        float plot_left, float plot_top, float plot_w, float plot_h,
                        float min_cost, float max_cost, float range,
                        const std::vector<float>& costs,
                        bool has_collision, bool has_smoothness, bool show_hints) {
        // Title
        sf::Text title(font, "Cost Convergence", FontSize::title);
        title.setFillColor(Colors::text());
        title.setStyle(sf::Text::Bold);
        title.setPosition({plot_left, 18.0f});
        target.draw(title);

        // X-axis label
        sf::Text xlabel(font, "Iteration", FontSize::axisLabel);
        xlabel.setFillColor(Colors::text());
        sf::FloatRect xb = xlabel.getLocalBounds();
        xlabel.setPosition({plot_left + plot_w / 2 - xb.size.x / 2, plot_top + plot_h + 38.0f});
        target.draw(xlabel);

        // Y-axis label
        sf::Text ylabel(font, "Cost", FontSize::axisLabel);
        ylabel.setFillColor(Colors::text());
        ylabel.setRotation(sf::degrees(-90.0f));
        ylabel.setPosition({22.0f, plot_top + plot_h / 2 + 15.0f});
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
            label.setPosition({plot_left - lb.size.x - 10.0f, y - 7.0f});
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
            label.setPosition({x - lb.size.x / 2, plot_top + plot_h + 10.0f});
            target.draw(label);
        }

        // Legend
        float lx = plot_left + plot_w - 155.0f;
        float ly = plot_top + 12.0f;
        float legend_h = 35.0f + (has_collision ? Layout::itemSpacing : 0.0f) + 
                         (has_smoothness ? Layout::itemSpacing : 0.0f);
        
        target.draw(createBox(lx, ly, 145.0f, legend_h, Colors::legendBg(), Colors::legendBorder()));

        float item_y = ly + 10.0f;
        drawLegendEntry(target, font, lx + 8.0f, item_y, "Total Cost", Colors::total(), Sizes::costLine);
        
        if (has_collision) {
            item_y += Layout::itemSpacing;
            drawLegendEntry(target, font, lx + 8.0f, item_y, "Collision", Colors::collision(), Sizes::costLineSecondary);
        }
        if (has_smoothness) {
            item_y += Layout::itemSpacing;
            drawLegendEntry(target, font, lx + 8.0f, item_y, "Smoothness", Colors::smoothness(), Sizes::costLineSecondary);
        }

        // Stats box
        float sx = plot_left + 12.0f;
        float sy = plot_top + 12.0f;
        
        target.draw(createBox(sx, sy, 155.0f, 78.0f,
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
            stat_y += 20.0f;
        }

        // Interactive hints
        if (show_hints) {
            sf::Text hint(font, "S/P: Save PNG | C: Collision | O: Smoothness", FontSize::hint);
            hint.setFillColor(Colors::hint());
            hint.setPosition({12.0f, window_height_ - 22.0f});
            target.draw(hint);
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
                            font, font_loaded, false);
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
                       bool show_collision, bool show_smoothness,
                       int counter, float scale) {
        sf::RenderTexture rt;
        if (!createRenderTexture(rt, scale)) return;

        rt.clear(sf::Color::White);
        drawCostFigure(rt, history, font, font_loaded, show_collision, show_smoothness, false);
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