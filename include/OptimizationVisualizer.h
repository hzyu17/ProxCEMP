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
 * Color scheme: Blue tones for trajectories
 * - Current trajectory: Dark blue (#1f77b4)
 * - Previous trajectories: Light blue with gradient
 * - Start: Green (#2ca02c)
 * - Goal: Orange (#ff7f0e)
 */
class OptimizationVisualizer {
public:
    OptimizationVisualizer(unsigned int width = 800, unsigned int height = 600)
        : window_width_(width), window_height_(height), output_prefix_("optimization")
    {
        // Publication color palette (matplotlib default)
        color_current_ = sf::Color(31, 119, 180);      // #1f77b4 - dark blue
        color_previous_ = sf::Color(174, 199, 232);    // #aec7e8 - light blue
        color_samples_ = sf::Color(31, 119, 180, 20);  // transparent blue
        color_start_ = sf::Color(44, 160, 44);         // #2ca02c - green
        color_goal_ = sf::Color(255, 127, 14);         // #ff7f0e - orange
        color_obstacle_ = sf::Color(200, 200, 200);    // light gray
        color_obstacle_edge_ = sf::Color(150, 150, 150);
        color_axis_ = sf::Color(50, 50, 50);
        color_grid_ = sf::Color(230, 230, 230);
        color_text_ = sf::Color(30, 30, 30);
    }

    /**
     * @brief Non-interactive save of the final trajectory state.
     * This does NOT open a window and can be called in batch scripts.
     */
    void saveStaticPlot(
        const ObstacleMap& obstacle_map,
        const Trajectory& final_trajectory,
        const std::string& filename)
    {
        // 1. Create a dummy history object to reuse the existing drawTrajectoryFigure logic
        OptimizationHistory history;
        IterationData data;
        data.iteration = 0;
        data.mean_trajectory = final_trajectory;
        data.total_cost = 0.0f; // Could be computed if needed
        history.addIteration(data);

        // 2. Setup Off-screen Render Texture
        const float SCALE = 2.0f; // High-res scaling
        unsigned int w = static_cast<unsigned int>(window_width_ * SCALE);
        unsigned int h = static_cast<unsigned int>(window_height_ * SCALE);

        sf::RenderTexture rt;
        if (!rt.resize({w, h})) {
            std::cerr << "Failed to create off-screen render texture\n";
            return;
        }

        // Set view to match coordinate system used in drawTrajectoryFigure
        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)window_width_, (float)window_height_}));
        rt.setView(view);

        // 3. Load font for axis labels and text
        sf::Font font;
        bool font_loaded = loadFont(font);

        // 4. Render and Save
        rt.clear(sf::Color::White);
        // Reuse your existing figure drawing logic
        drawTrajectoryFigure(rt, obstacle_map, history, 0, false, false, font, font_loaded);
        rt.display();

        if (rt.getTexture().copyToImage().saveToFile(filename)) {
            // Only print if not in a tight loop, or handled by Python
        } else {
            std::cerr << "Failed to save image: " << filename << std::endl;
        }
    }

    /**
     * @brief Set output prefix for saved files (e.g., "pcem" or "ngd")
     */
    void setOutputPrefix(const std::string& prefix) {
        output_prefix_ = prefix;
    }

    void showTrajectoryEvolution(
        const ObstacleMap& obstacle_map,
        const OptimizationHistory& history,
        const std::string& title = "Trajectory Evolution")
    {
        sf::RenderWindow window(
            sf::VideoMode({window_width_, window_height_}),
            title,
            sf::Style::Titlebar | sf::Style::Close
        );
        window.setFramerateLimit(60);

        sf::Font font;
        bool font_loaded = loadFont(font);

        int current_iteration = history.iterations.size() - 1;
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

                if (event->is<sf::Event::KeyPressed>()) {
                    const auto& key = event->getIf<sf::Event::KeyPressed>();
                    
                    if (key->code == sf::Keyboard::Key::Escape) window.close();
                    if (key->code == sf::Keyboard::Key::Left)
                        current_iteration = std::max(0, current_iteration - 1);
                    if (key->code == sf::Keyboard::Key::Right)
                        current_iteration = std::min((int)history.iterations.size() - 1, current_iteration + 1);
                    if (key->code == sf::Keyboard::Key::Home) current_iteration = 0;
                    if (key->code == sf::Keyboard::Key::End) current_iteration = history.iterations.size() - 1;
                    if (key->code == sf::Keyboard::Key::Space) { playing = !playing; frame_counter = 0; }
                    if (key->code == sf::Keyboard::Key::S) show_samples = !show_samples;
                    if (key->code == sf::Keyboard::Key::M) show_all_means = !show_all_means;
                    if (key->code == sf::Keyboard::Key::P) {
                        saveTrajectoryHighRes(obstacle_map, history, current_iteration,
                                             show_samples, show_all_means, font, font_loaded, save_counter++);
                    }
                    if (key->code == sf::Keyboard::Key::G) {
                        saveAnimationGIF(obstacle_map, history, show_samples, show_all_means, font, font_loaded);
                    }
                }
            }

            if (playing) {
                frame_counter++;
                if (frame_counter >= play_speed) {
                    frame_counter = 0;
                    current_iteration = (current_iteration + 1) % history.iterations.size();
                }
            }

            window.clear(sf::Color::White);
            drawTrajectoryFigure(window, obstacle_map, history, current_iteration,
                                show_samples, show_all_means, font, font_loaded);
            window.display();
        }
    }

    void showCostPlot(
        const OptimizationHistory& history,
        const std::string& title = "Cost Convergence")
    {
        sf::RenderWindow window(
            sf::VideoMode({window_width_, window_height_}),
            title,
            sf::Style::Titlebar | sf::Style::Close
        );
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

                if (event->is<sf::Event::KeyPressed>()) {
                    const auto& key = event->getIf<sf::Event::KeyPressed>();
                    
                    if (key->code == sf::Keyboard::Key::Escape) window.close();
                    if (key->code == sf::Keyboard::Key::C) show_collision = !show_collision;
                    if (key->code == sf::Keyboard::Key::S) show_smoothness = !show_smoothness;
                    if (key->code == sf::Keyboard::Key::P) {
                        saveCostPlotHighRes(history, font, font_loaded, show_collision, show_smoothness, save_counter++);
                    }
                }
            }

            window.clear(sf::Color::White);
            drawCostFigure(window, history, font, font_loaded, show_collision, show_smoothness);
            window.display();
        }
    }

    void showAll(const ObstacleMap& obstacle_map, const OptimizationHistory& history, const std::string& name = "") {
        if (!name.empty()) setOutputPrefix(name);
        showTrajectoryEvolution(obstacle_map, history, name.empty() ? "Trajectory Evolution" : name + " - Trajectory");
        showCostPlot(history, name.empty() ? "Cost Convergence" : name + " - Cost");
    }

    /**
     * @brief Save animation as GIF (requires ImageMagick installed)
     */
    void saveAnimationGIF(
        const ObstacleMap& obstacle_map,
        const OptimizationHistory& history,
        bool show_samples,
        bool show_all_means,
        const sf::Font& font,
        bool font_loaded,
        int delay_ms = 100)
    {
        if (history.iterations.empty()) return;

        std::cout << "Saving animation frames...\n";

        const float SCALE = 2.0f;  // 2x for GIF (balance quality vs file size)
        unsigned int w = static_cast<unsigned int>(window_width_ * SCALE);
        unsigned int h = static_cast<unsigned int>(window_height_ * SCALE);

        sf::RenderTexture rt;
        if (!rt.resize({w, h})) {
            std::cerr << "Failed to create render texture!\n";
            return;
        }

        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)window_width_, (float)window_height_}));
        view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
        rt.setView(view);

        std::string frame_dir = output_prefix_ + "_frames";
        std::string mkdir_cmd = "mkdir -p " + frame_dir;
        std::system(mkdir_cmd.c_str());

        // Save each frame
        for (size_t i = 0; i < history.iterations.size(); ++i) {
            rt.clear(sf::Color::White);
            drawTrajectoryFigure(rt, obstacle_map, history, i, show_samples, show_all_means, font, font_loaded);
            rt.display();

            std::stringstream fname;
            fname << frame_dir << "/frame_" << std::setfill('0') << std::setw(4) << i << ".png";
            rt.getTexture().copyToImage().saveToFile(fname.str());

            if ((i + 1) % 10 == 0 || i == history.iterations.size() - 1) {
                std::cout << "  Saved frame " << (i + 1) << "/" << history.iterations.size() << "\n";
            }
        }

        // Use ImageMagick to create GIF
        std::string gif_name = output_prefix_ + "_trajectory_animation.gif";
        int delay_centisec = delay_ms / 10;  // ImageMagick uses centiseconds
        
        std::stringstream cmd;
        cmd << "convert -delay " << delay_centisec << " -loop 0 "
            << frame_dir << "/frame_*.png " << gif_name;

        std::cout << "Creating GIF with ImageMagick...\n";
        int result = std::system(cmd.str().c_str());

        if (result == 0) {
            std::cout << "Saved: " << gif_name << "\n";
            
            // Clean up frames
            std::string cleanup_cmd = "rm -rf " + frame_dir;
            std::system(cleanup_cmd.c_str());
            std::cout << "Cleaned up temporary frames.\n";
        } else {
            std::cerr << "ImageMagick conversion failed. Frames saved in: " << frame_dir << "/\n";
            std::cerr << "Install ImageMagick with: sudo apt install imagemagick\n";
            std::cerr << "Then run manually: " << cmd.str() << "\n";
        }
    }

private:
    unsigned int window_width_;
    unsigned int window_height_;
    std::string output_prefix_;
    
    // Publication color palette
    sf::Color color_current_;
    sf::Color color_previous_;
    sf::Color color_samples_;
    sf::Color color_start_;
    sf::Color color_goal_;
    sf::Color color_obstacle_;
    sf::Color color_obstacle_edge_;
    sf::Color color_axis_;
    sf::Color color_grid_;
    sf::Color color_text_;

    bool loadFont(sf::Font& font) {
        std::vector<std::string> paths = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        };
        for (const auto& p : paths) {
            if (font.openFromFile(p)) return true;
        }
        return false;
    }

    void printTrajectoryControls() {
        std::cout << "\n=== Trajectory Controls ===\n"
                  << "Arrows: Navigate | Space: Play | S: Samples | M: Means\n"
                  << "P: Save PNG | G: Save GIF animation\n\n";
    }

    void printCostControls() {
        std::cout << "\n=== Cost Plot Controls ===\n"
                  << "C: Collision | S: Smoothness | P: Save PNG\n\n";
    }

    void drawLine(sf::RenderTarget& t, sf::Vector2f p1, sf::Vector2f p2, float w, sf::Color c) {
        sf::Vector2f d = p2 - p1;
        float len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len < 0.001f) return;
        sf::RectangleShape line({len, w});
        line.setPosition(p1);
        line.setFillColor(c);
        line.setRotation(sf::radians(std::atan2(d.y, d.x)));
        line.setOrigin({0, w / 2});
        t.draw(line);
    }

    void drawTrajectoryFigure(
        sf::RenderTarget& target,
        const ObstacleMap& obstacle_map,
        const OptimizationHistory& history,
        int current_iteration,
        bool show_samples,
        bool show_all_means,
        const sf::Font& font,
        bool font_loaded)
    {
        if (history.iterations.empty()) return;

        const auto& current = history.iterations[current_iteration];
        float map_w = obstacle_map.getMapWidth();
        float map_h = obstacle_map.getMapHeight();

        // Margins for axes labels
        float margin_left = 65.0f;
        float margin_right = 25.0f;
        float margin_top = 50.0f;
        float margin_bottom = 55.0f;

        float plot_w = window_width_ - margin_left - margin_right;
        float plot_h = window_height_ - margin_top - margin_bottom;
        float scale = std::min(plot_w / map_w, plot_h / map_h);
        
        float plot_actual_w = map_w * scale;
        float plot_actual_h = map_h * scale;
        float offset_x = margin_left + (plot_w - plot_actual_w) / 2;
        float offset_y = margin_top + (plot_h - plot_actual_h) / 2;

        auto tx = [&](float x, float y) -> sf::Vector2f {
            return {offset_x + x * scale, offset_y + y * scale};
        };

        // Plot area background
        sf::RectangleShape plot_bg({plot_actual_w, plot_actual_h});
        plot_bg.setPosition({offset_x, offset_y});
        plot_bg.setFillColor(sf::Color::White);
        plot_bg.setOutlineColor(color_axis_);
        plot_bg.setOutlineThickness(1.5f);
        target.draw(plot_bg);

        // Obstacles
        for (const auto& obs : obstacle_map.getObstacles()) {
            if (obs.dimensions() >= 2) {
                sf::Vector2f pos = tx(obs.center(0), obs.center(1));
                float r = obs.radius * scale;
                sf::CircleShape circle(r);
                circle.setPosition({pos.x - r, pos.y - r});
                circle.setFillColor(color_obstacle_);
                circle.setOutlineColor(color_obstacle_edge_);
                circle.setOutlineThickness(1.0f);
                target.draw(circle);
            }
        }

        // Samples (light blue, very transparent)
        if (show_samples && !current.samples.empty()) {
            for (const auto& sample : current.samples) {
                for (size_t i = 0; i < sample.nodes.size() - 1; ++i) {
                    sf::Vector2f p1 = tx(sample.nodes[i].position(0), sample.nodes[i].position(1));
                    sf::Vector2f p2 = tx(sample.nodes[i+1].position(0), sample.nodes[i+1].position(1));
                    drawLine(target, p1, p2, 1.0f, color_samples_);
                }
            }
        }

        // Previous trajectories (light blue with gradient)
        if (show_all_means && current_iteration > 0) {
            for (int i = 0; i < current_iteration; ++i) {
                const auto& traj = history.iterations[i].mean_trajectory;
                float t = (float)i / current_iteration;
                uint8_t alpha = static_cast<uint8_t>(40 + 120 * t);
                sf::Color col(174, 199, 232, alpha);
                
                for (size_t j = 0; j < traj.nodes.size() - 1; ++j) {
                    sf::Vector2f p1 = tx(traj.nodes[j].position(0), traj.nodes[j].position(1));
                    sf::Vector2f p2 = tx(traj.nodes[j+1].position(0), traj.nodes[j+1].position(1));
                    drawLine(target, p1, p2, 1.5f, col);
                }
            }
        }

        // Current trajectory (dark blue, bold)
        const auto& mean_traj = current.mean_trajectory;
        for (size_t i = 0; i < mean_traj.nodes.size() - 1; ++i) {
            sf::Vector2f p1 = tx(mean_traj.nodes[i].position(0), mean_traj.nodes[i].position(1));
            sf::Vector2f p2 = tx(mean_traj.nodes[i+1].position(0), mean_traj.nodes[i+1].position(1));
            drawLine(target, p1, p2, 2.5f, color_current_);
        }

        // Waypoints on current trajectory
        for (size_t i = 1; i < mean_traj.nodes.size() - 1; ++i) {
            sf::Vector2f pos = tx(mean_traj.nodes[i].position(0), mean_traj.nodes[i].position(1));
            sf::CircleShape marker(3.0f);
            marker.setPosition({pos.x - 3.0f, pos.y - 3.0f});
            marker.setFillColor(color_current_);
            target.draw(marker);
        }

        // Start and Goal markers
        if (!mean_traj.nodes.empty()) {
            sf::Vector2f start = tx(mean_traj.nodes[mean_traj.start_index].position(0),
                                   mean_traj.nodes[mean_traj.start_index].position(1));
            sf::Vector2f goal = tx(mean_traj.nodes[mean_traj.goal_index].position(0),
                                  mean_traj.nodes[mean_traj.goal_index].position(1));

            sf::CircleShape start_m(8.0f);
            start_m.setPosition({start.x - 8.0f, start.y - 8.0f});
            start_m.setFillColor(color_start_);
            start_m.setOutlineColor(sf::Color(30, 100, 30));
            start_m.setOutlineThickness(1.5f);
            target.draw(start_m);

            sf::RectangleShape goal_m({14.0f, 14.0f});
            goal_m.setPosition({goal.x - 7.0f, goal.y - 7.0f});
            goal_m.setFillColor(color_goal_);
            goal_m.setOutlineColor(sf::Color(180, 90, 10));
            goal_m.setOutlineThickness(1.5f);
            target.draw(goal_m);
        }

        if (font_loaded) {
            // Axis labels
            sf::Text xlabel(font, "X Position", 14);
            xlabel.setFillColor(color_text_);
            sf::FloatRect xb = xlabel.getLocalBounds();
            xlabel.setPosition({offset_x + plot_actual_w / 2 - xb.size.x / 2, offset_y + plot_actual_h + 30.0f});
            target.draw(xlabel);

            sf::Text ylabel(font, "Y Position", 14);
            ylabel.setFillColor(color_text_);
            ylabel.setRotation(sf::degrees(-90.0f));
            ylabel.setPosition({offset_x - 45.0f, offset_y + plot_actual_h / 2 + 30.0f});
            target.draw(ylabel);

            // Tick labels
            for (int i = 0; i <= 4; ++i) {
                float data_x = i * map_w / 4;
                float data_y = i * map_h / 4;
                float sx = offset_x + i * plot_actual_w / 4;
                float sy = offset_y + i * plot_actual_h / 4;

                std::stringstream ssx;
                ssx << (int)data_x;
                sf::Text xt(font, ssx.str(), 11);
                xt.setFillColor(color_text_);
                sf::FloatRect xtb = xt.getLocalBounds();
                xt.setPosition({sx - xtb.size.x / 2, offset_y + plot_actual_h + 8.0f});
                target.draw(xt);

                std::stringstream ssy;
                ssy << (int)data_y;
                sf::Text yt(font, ssy.str(), 11);
                yt.setFillColor(color_text_);
                sf::FloatRect ytb = yt.getLocalBounds();
                yt.setPosition({offset_x - ytb.size.x - 8.0f, sy - 6.0f});
                target.draw(yt);
            }

            // Title with iteration info
            std::stringstream title_ss;
            title_ss << "Iteration " << current_iteration + 1 << "/" << history.iterations.size();
            sf::Text title(font, title_ss.str(), 16);
            title.setFillColor(color_text_);
            title.setStyle(sf::Text::Bold);
            title.setPosition({offset_x, 15.0f});
            target.draw(title);

            // Cost value
            std::stringstream cost_ss;
            cost_ss << "Cost: " << std::fixed << std::setprecision(2) << current.total_cost;
            sf::Text cost_text(font, cost_ss.str(), 13);
            cost_text.setFillColor(color_text_);
            cost_text.setPosition({offset_x + plot_actual_w - 120.0f, 18.0f});
            target.draw(cost_text);

            // Legend
            float lx = offset_x + plot_actual_w - 140.0f;
            float ly = offset_y + 10.0f;
            
            sf::RectangleShape legend_bg({130.0f, 85.0f});
            legend_bg.setPosition({lx, ly});
            legend_bg.setFillColor(sf::Color(255, 255, 255, 240));
            legend_bg.setOutlineColor(sf::Color(180, 180, 180));
            legend_bg.setOutlineThickness(1.0f);
            target.draw(legend_bg);

            float item_y = ly + 8.0f;
            
            drawLine(target, {lx + 8, item_y + 7}, {lx + 30, item_y + 7}, 2.5f, color_current_);
            sf::Text l1(font, "Current", 11);
            l1.setFillColor(color_text_);
            l1.setPosition({lx + 38, item_y});
            target.draw(l1);
            
            item_y += 18.0f;
            drawLine(target, {lx + 8, item_y + 7}, {lx + 30, item_y + 7}, 2.0f, color_previous_);
            sf::Text l2(font, "Previous", 11);
            l2.setFillColor(color_text_);
            l2.setPosition({lx + 38, item_y});
            target.draw(l2);
            
            item_y += 18.0f;
            sf::CircleShape sm(5.0f);
            sm.setPosition({lx + 14, item_y + 3});
            sm.setFillColor(color_start_);
            target.draw(sm);
            sf::Text l3(font, "Start", 11);
            l3.setFillColor(color_text_);
            l3.setPosition({lx + 38, item_y});
            target.draw(l3);
            
            item_y += 18.0f;
            sf::RectangleShape gm({10.0f, 10.0f});
            gm.setPosition({lx + 14, item_y + 2});
            gm.setFillColor(color_goal_);
            target.draw(gm);
            sf::Text l4(font, "Goal", 11);
            l4.setFillColor(color_text_);
            l4.setPosition({lx + 38, item_y});
            target.draw(l4);

            // Controls hint
            sf::Text hint(font, "P: Save PNG | G: Save GIF | Space: Play", 9);
            hint.setFillColor(sf::Color(140, 140, 140));
            hint.setPosition({10.0f, window_height_ - 18.0f});
            target.draw(hint);
        }
    }

    void drawCostFigure(
        sf::RenderTarget& target,
        const OptimizationHistory& history,
        const sf::Font& font,
        bool font_loaded,
        bool show_collision,
        bool show_smoothness)
    {
        if (history.iterations.empty()) return;

        float margin_left = 75.0f;
        float margin_right = 25.0f;
        float margin_top = 50.0f;
        float margin_bottom = 55.0f;

        float plot_left = margin_left;
        float plot_right = window_width_ - margin_right;
        float plot_top = margin_top;
        float plot_bottom = window_height_ - margin_bottom;
        float plot_w = plot_right - plot_left;
        float plot_h = plot_bottom - plot_top;

        auto costs = history.getCostHistory();
        auto collision_costs = history.getCollisionCostHistory();
        auto smoothness_costs = history.getSmoothnessCostHistory();

        float max_cost = *std::max_element(costs.begin(), costs.end());
        float min_cost = *std::min_element(costs.begin(), costs.end());
        float range = max_cost - min_cost;
        if (range < 1e-6f) range = max_cost * 0.1f;
        if (range < 1e-6f) range = 1.0f;
        
        min_cost -= range * 0.05f;
        max_cost += range * 0.05f;
        range = max_cost - min_cost;

        // Plot background
        sf::RectangleShape plot_bg({plot_w, plot_h});
        plot_bg.setPosition({plot_left, plot_top});
        plot_bg.setFillColor(sf::Color::White);
        plot_bg.setOutlineColor(color_axis_);
        plot_bg.setOutlineThickness(1.5f);
        target.draw(plot_bg);

        // Grid lines
        int n_grid = 5;
        for (int i = 1; i < n_grid; ++i) {
            float y = plot_top + i * plot_h / n_grid;
            sf::RectangleShape hline({plot_w, 1.0f});
            hline.setPosition({plot_left, y});
            hline.setFillColor(color_grid_);
            target.draw(hline);

            float x = plot_left + i * plot_w / n_grid;
            sf::RectangleShape vline({1.0f, plot_h});
            vline.setPosition({x, plot_top});
            vline.setFillColor(color_grid_);
            target.draw(vline);
        }

        // Draw curves
        auto drawCurve = [&](const std::vector<float>& data, sf::Color color, float width) {
            if (data.size() < 2) return;
            for (size_t i = 0; i < data.size() - 1; ++i) {
                float x1 = plot_left + (float)i / (data.size() - 1) * plot_w;
                float x2 = plot_left + (float)(i + 1) / (data.size() - 1) * plot_w;
                float y1 = plot_bottom - (data[i] - min_cost) / range * plot_h;
                float y2 = plot_bottom - (data[i + 1] - min_cost) / range * plot_h;
                drawLine(target, {x1, y1}, {x2, y2}, width, color);
            }
        };

        bool has_collision = show_collision && std::any_of(collision_costs.begin(), collision_costs.end(), 
                                                           [](float c) { return c > 1e-6f; });
        bool has_smoothness = show_smoothness && std::any_of(smoothness_costs.begin(), smoothness_costs.end(), 
                                                              [](float c) { return c > 1e-6f; });

        sf::Color color_total(31, 119, 180);
        sf::Color color_collision(255, 127, 14);
        sf::Color color_smooth(44, 160, 44);

        if (has_smoothness) drawCurve(smoothness_costs, color_smooth, 2.0f);
        if (has_collision) drawCurve(collision_costs, color_collision, 2.0f);
        drawCurve(costs, color_total, 2.5f);

        if (font_loaded) {
            // Title
            sf::Text title(font, "Cost Convergence", 16);
            title.setFillColor(color_text_);
            title.setStyle(sf::Text::Bold);
            title.setPosition({plot_left, 15.0f});
            target.draw(title);

            // Axis labels
            sf::Text xlabel(font, "Iteration", 14);
            xlabel.setFillColor(color_text_);
            sf::FloatRect xb = xlabel.getLocalBounds();
            xlabel.setPosition({plot_left + plot_w / 2 - xb.size.x / 2, plot_bottom + 32.0f});
            target.draw(xlabel);

            sf::Text ylabel(font, "Cost", 14);
            ylabel.setFillColor(color_text_);
            ylabel.setRotation(sf::degrees(-90.0f));
            ylabel.setPosition({20.0f, plot_top + plot_h / 2 + 15.0f});
            target.draw(ylabel);

            // Y ticks
            for (int i = 0; i <= n_grid; ++i) {
                float val = max_cost - i * range / n_grid;
                float y = plot_top + i * plot_h / n_grid;
                std::stringstream ss;
                ss << std::fixed << std::setprecision(1) << val;
                sf::Text label(font, ss.str(), 11);
                label.setFillColor(color_text_);
                sf::FloatRect lb = label.getLocalBounds();
                label.setPosition({plot_left - lb.size.x - 8.0f, y - 6.0f});
                target.draw(label);
            }

            // X ticks
            int n_x_ticks = std::min(10, (int)costs.size());
            for (int i = 0; i <= n_x_ticks; ++i) {
                int iter = i * (costs.size() - 1) / n_x_ticks;
                float x = plot_left + (float)iter / (costs.size() - 1) * plot_w;
                sf::Text label(font, std::to_string(iter + 1), 11);
                label.setFillColor(color_text_);
                sf::FloatRect lb = label.getLocalBounds();
                label.setPosition({x - lb.size.x / 2, plot_bottom + 8.0f});
                target.draw(label);
            }

            // Legend
            float lx = plot_right - 145.0f;
            float ly = plot_top + 10.0f;
            float lh = 30.0f + (has_collision ? 18.0f : 0.0f) + (has_smoothness ? 18.0f : 0.0f);
            
            sf::RectangleShape legend_bg({135.0f, lh});
            legend_bg.setPosition({lx, ly});
            legend_bg.setFillColor(sf::Color(255, 255, 255, 240));
            legend_bg.setOutlineColor(sf::Color(180, 180, 180));
            legend_bg.setOutlineThickness(1.0f);
            target.draw(legend_bg);

            float item_y = ly + 8.0f;
            
            drawLine(target, {lx + 8, item_y + 7}, {lx + 30, item_y + 7}, 2.5f, color_total);
            sf::Text lt(font, "Total Cost", 11);
            lt.setFillColor(color_text_);
            lt.setPosition({lx + 38, item_y});
            target.draw(lt);

            if (has_collision) {
                item_y += 18.0f;
                drawLine(target, {lx + 8, item_y + 7}, {lx + 30, item_y + 7}, 2.0f, color_collision);
                sf::Text lc(font, "Collision", 11);
                lc.setFillColor(color_text_);
                lc.setPosition({lx + 38, item_y});
                target.draw(lc);
            }

            if (has_smoothness) {
                item_y += 18.0f;
                drawLine(target, {lx + 8, item_y + 7}, {lx + 30, item_y + 7}, 2.0f, color_smooth);
                sf::Text ls(font, "Smoothness", 11);
                ls.setFillColor(color_text_);
                ls.setPosition({lx + 38, item_y});
                target.draw(ls);
            }

            // Stats box
            float sx = plot_left + 10.0f;
            float sy = plot_top + 10.0f;
            
            sf::RectangleShape stats_bg({140.0f, 68.0f});
            stats_bg.setPosition({sx, sy});
            stats_bg.setFillColor(sf::Color(250, 252, 255, 245));
            stats_bg.setOutlineColor(sf::Color(180, 180, 180));
            stats_bg.setOutlineThickness(1.0f);
            target.draw(stats_bg);

            float stat_y = sy + 8.0f;
            std::stringstream ss1, ss2, ss3;
            ss1 << "Initial: " << std::fixed << std::setprecision(2) << costs.front();
            ss2 << "Final: " << std::fixed << std::setprecision(2) << costs.back();
            float improve = (costs.front() - costs.back()) / costs.front() * 100;
            ss3 << "Reduction: " << std::fixed << std::setprecision(1) << improve << "%";

            sf::Text t1(font, ss1.str(), 11); t1.setFillColor(color_text_); t1.setPosition({sx + 8, stat_y}); target.draw(t1);
            stat_y += 18.0f;
            sf::Text t2(font, ss2.str(), 11); t2.setFillColor(color_text_); t2.setPosition({sx + 8, stat_y}); target.draw(t2);
            stat_y += 18.0f;
            sf::Text t3(font, ss3.str(), 11); t3.setFillColor(color_text_); t3.setPosition({sx + 8, stat_y}); target.draw(t3);

            // Controls hint
            sf::Text hint(font, "P: Save PNG | C: Collision | S: Smoothness", 9);
            hint.setFillColor(sf::Color(140, 140, 140));
            hint.setPosition({10.0f, window_height_ - 18.0f});
            target.draw(hint);
        }
    }

    void saveTrajectoryHighRes(
        const ObstacleMap& obstacle_map,
        const OptimizationHistory& history,
        int iter, bool show_samples, bool show_all_means,
        const sf::Font& font, bool font_loaded, int counter)
    {
        const float SCALE = 4.0f;
        unsigned int w = static_cast<unsigned int>(window_width_ * SCALE);
        unsigned int h = static_cast<unsigned int>(window_height_ * SCALE);

        sf::RenderTexture rt;
        if (!rt.resize({w, h})) return;

        rt.clear(sf::Color::White);
        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)window_width_, (float)window_height_}));
        view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
        rt.setView(view);

        drawTrajectoryFigure(rt, obstacle_map, history, iter, show_samples, show_all_means, font, font_loaded);
        rt.display();

        std::string fname = output_prefix_ + "_trajectory_iter" + std::to_string(iter + 1) + ".png";
        if (rt.getTexture().copyToImage().saveToFile(fname))
            std::cout << "Saved: " << fname << " (" << w << "x" << h << ")\n";
    }

    void saveCostPlotHighRes(
        const OptimizationHistory& history,
        const sf::Font& font, bool font_loaded,
        bool show_collision, bool show_smoothness, int counter)
    {
        const float SCALE = 4.0f;
        unsigned int w = static_cast<unsigned int>(window_width_ * SCALE);
        unsigned int h = static_cast<unsigned int>(window_height_ * SCALE);

        sf::RenderTexture rt;
        if (!rt.resize({w, h})) return;

        rt.clear(sf::Color::White);
        sf::View view(sf::FloatRect({0.f, 0.f}, {(float)window_width_, (float)window_height_}));
        view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
        rt.setView(view);

        drawCostFigure(rt, history, font, font_loaded, show_collision, show_smoothness);
        rt.display();

        std::string fname = output_prefix_ + "_cost_convergence.png";
        if (rt.getTexture().copyToImage().saveToFile(fname))
            std::cout << "Saved: " << fname << " (" << w << "x" << h << ")\n";
    }
};