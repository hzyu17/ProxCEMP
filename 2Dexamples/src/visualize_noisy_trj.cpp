#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "../include/PCEMotionPlanner.h"
#include "../include/CollisionAvoidanceTask.h"
#include "../include/visualization.h"

// --- Statistics Struct ---
struct NoiseStats {
    float avg_perturbation = 0.0f;
    float max_perturbation = 0.0f;
    size_t num_samples = 0;
    size_t num_nodes = 0;
};

/**
 * @brief Export trajectory data to CSV files for matplotlib plotting
 */
void exportDataForMatplotlib(
    const std::vector<ObstacleND>& obstacles,
    const Trajectory& base_trajectory,
    const std::vector<Trajectory>& noisy_samples,
    const NoiseStats& stats,
    const std::string& prefix = "noise_viz") 
{
    // Export obstacles
    std::ofstream obs_file(prefix + "_obstacles.csv");
    obs_file << "x,y,radius\n";
    for (const auto& obs : obstacles) {
        if (obs.dimensions() >= 2) {
            obs_file << obs.center(0) << "," << obs.center(1) << "," << obs.radius << "\n";
        }
    }
    obs_file.close();
    std::cout << "Saved: " << prefix << "_obstacles.csv\n";
    
    // Export base trajectory
    std::ofstream base_file(prefix + "_base_trajectory.csv");
    base_file << "x,y\n";
    for (const auto& node : base_trajectory.nodes) {
        base_file << node.position(0) << "," << node.position(1) << "\n";
    }
    base_file.close();
    std::cout << "Saved: " << prefix << "_base_trajectory.csv\n";
    
    // Export start and goal indices
    std::ofstream info_file(prefix + "_info.csv");
    info_file << "start_idx,goal_idx,num_samples,num_nodes,avg_pert,max_pert\n";
    info_file << base_trajectory.start_index << "," 
              << base_trajectory.goal_index << ","
              << stats.num_samples << ","
              << stats.num_nodes << ","
              << stats.avg_perturbation << ","
              << stats.max_perturbation << "\n";
    info_file.close();
    std::cout << "Saved: " << prefix << "_info.csv\n";
    
    // Export noisy samples (one file with sample_id column)
    std::ofstream samples_file(prefix + "_samples.csv");
    samples_file << "sample_id,node_id,x,y\n";
    for (size_t m = 0; m < noisy_samples.size(); ++m) {
        for (size_t i = 0; i < noisy_samples[m].nodes.size(); ++i) {
            samples_file << m << "," << i << ","
                        << noisy_samples[m].nodes[i].position(0) << ","
                        << noisy_samples[m].nodes[i].position(1) << "\n";
        }
    }
    samples_file.close();
    std::cout << "Saved: " << prefix << "_samples.csv\n";
    
    std::cout << "\nUse the provided Python script to generate PDF/EPS figures.\n";
}

// --- Visualization Function ---

/**
 * @brief Visualizes smoothness noise distribution N(0, R^-1) in workspace
 */
void visualizeNoise(const std::vector<ObstacleND>& obstacles,
                    const Trajectory& workspace_base_trajectory, 
                    const std::vector<Trajectory>& workspace_noisy_samples,
                    const NoiseStats& stats = NoiseStats()) {
    
    sf::RenderWindow window(sf::VideoMode({MAP_WIDTH, MAP_HEIGHT}), 
                           "Smoothness Noise Visualization N(0, R^-1)", 
                           sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    // Load font for text rendering
    sf::Font font;
    bool font_loaded = font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    if (!font_loaded) {
        font_loaded = font.openFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf");
    }
    if (!font_loaded) {
        font_loaded = font.openFromFile("/usr/share/fonts/TTF/DejaVuSans.ttf");
    }
    
    if (!font_loaded) {
        std::cerr << "Warning: Could not load font. Text labels will not be displayed.\n";
    }

    std::cout << "\n=== Controls ===\n";
    std::cout << "ESC: Exit\n";
    std::cout << "L:   Toggle legend\n";
    std::cout << "P:   Save as high-res PNG (3x scale)\n";
    std::cout << "================\n\n";

    bool show_legend = true;
    int save_counter = 0;
    
    // Scale factor for high-resolution export (3x = ~300 DPI for typical screen)
    const float SAVE_SCALE = 3.0f;

    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            
            if (event->is<sf::Event::KeyPressed>()) {
                const auto& key_event = event->getIf<sf::Event::KeyPressed>();
                if (key_event->code == sf::Keyboard::Key::Escape) {
                    window.close();
                }
                if (key_event->code == sf::Keyboard::Key::L) {
                    show_legend = !show_legend;
                }
                if (key_event->code == sf::Keyboard::Key::P) {
                    // Create high-resolution render texture
                    unsigned int hi_res_width = static_cast<unsigned int>(MAP_WIDTH * SAVE_SCALE);
                    unsigned int hi_res_height = static_cast<unsigned int>(MAP_HEIGHT * SAVE_SCALE);
                    
                    sf::RenderTexture renderTexture;
                    if (!renderTexture.resize({hi_res_width, hi_res_height})) {
                        std::cerr << "Failed to create render texture!\n";
                        continue;
                    }
                    
                    renderTexture.clear(sf::Color(240, 240, 240));
                    
                    // Apply scaling transform
                    sf::View scaledView(sf::FloatRect({0.f, 0.f}, {(float)MAP_WIDTH, (float)MAP_HEIGHT}));
                    scaledView.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
                    renderTexture.setView(scaledView);
                    
                    // Redraw everything to the high-res texture
                    // 1. Obstacles
                    drawObstacles(renderTexture, obstacles);
                    
                    // 2. Noisy samples
                    for (const auto& sample : workspace_noisy_samples) {
                        drawTrajectorySegments(renderTexture, sample, sf::Color(50, 50, 255, 30));
                    }
                    
                    // 3. Base trajectory
                    drawTrajectorySegments(renderTexture, workspace_base_trajectory, sf::Color(255, 0, 0, 255));
                    for (const auto& node : workspace_base_trajectory.nodes) {
                        drawNode(renderTexture, node, 3.0f, sf::Color(255, 100, 100));
                    }
                    
                    // 4. Start/Goal
                    if (!workspace_base_trajectory.nodes.empty()) {
                        const auto& start_node = workspace_base_trajectory.nodes[workspace_base_trajectory.start_index];
                        const auto& goal_node = workspace_base_trajectory.nodes[workspace_base_trajectory.goal_index];
                        drawNode(renderTexture, start_node, 8.0f, sf::Color::Green);
                        drawNode(renderTexture, goal_node, 8.0f, sf::Color::Red);
                        
                        if (font_loaded) {
                            sf::Text start_label(font, "Start", 14);
                            start_label.setFillColor(sf::Color(0, 150, 0));
                            start_label.setStyle(sf::Text::Bold);
                            start_label.setPosition({start_node.position(0) + 12.0f, start_node.position(1) - 8.0f});
                            renderTexture.draw(start_label);
                            
                            sf::Text goal_label(font, "Goal", 14);
                            goal_label.setFillColor(sf::Color(200, 0, 0));
                            goal_label.setStyle(sf::Text::Bold);
                            goal_label.setPosition({goal_node.position(0) + 12.0f, goal_node.position(1) - 8.0f});
                            renderTexture.draw(goal_label);
                        }
                    }
                    
                    // 5. Legend
                    if (font_loaded && show_legend) {
                        float legend_x = 10.0f;
                        float legend_y = 10.0f;
                        float legend_width = 220.0f;
                        float legend_height = 180.0f;
                        
                        sf::RectangleShape legend_bg(sf::Vector2f(legend_width, legend_height));
                        legend_bg.setPosition({legend_x, legend_y});
                        legend_bg.setFillColor(sf::Color(255, 255, 255, 230));
                        legend_bg.setOutlineColor(sf::Color(100, 100, 100));
                        legend_bg.setOutlineThickness(1.0f);
                        renderTexture.draw(legend_bg);
                        
                        sf::Text title(font, "Noise Distribution N(0, R^-1)", 13);
                        title.setFillColor(sf::Color::Black);
                        title.setStyle(sf::Text::Bold);
                        title.setPosition({legend_x + 8.0f, legend_y + 5.0f});
                        renderTexture.draw(title);
                        
                        sf::RectangleShape separator(sf::Vector2f(legend_width - 16.0f, 1.0f));
                        separator.setPosition({legend_x + 8.0f, legend_y + 28.0f});
                        separator.setFillColor(sf::Color(150, 150, 150));
                        renderTexture.draw(separator);
                        
                        float item_y = legend_y + 35.0f;
                        float item_spacing = 22.0f;
                        
                        auto drawLegendItemHiRes = [&](const std::string& label, sf::Color color, float y, bool is_line = false) {
                            if (is_line) {
                                sf::RectangleShape line(sf::Vector2f(20.0f, 3.0f));
                                line.setPosition({legend_x + 12.0f, y + 6.0f});
                                line.setFillColor(color);
                                renderTexture.draw(line);
                            } else {
                                sf::CircleShape marker(6.0f);
                                marker.setPosition({legend_x + 12.0f, y + 2.0f});
                                marker.setFillColor(color);
                                renderTexture.draw(marker);
                            }
                            
                            sf::Text text(font, label, 12);
                            text.setFillColor(sf::Color(50, 50, 50));
                            text.setPosition({legend_x + 40.0f, y});
                            renderTexture.draw(text);
                        };
                        
                        drawLegendItemHiRes("Noise samples", sf::Color(50, 50, 255, 150), item_y, true);
                        item_y += item_spacing;
                        drawLegendItemHiRes("Mean trajectory", sf::Color(255, 0, 0), item_y, true);
                        item_y += item_spacing;
                        drawLegendItemHiRes("Start position", sf::Color::Green, item_y);
                        item_y += item_spacing;
                        drawLegendItemHiRes("Goal position", sf::Color::Red, item_y);
                        item_y += item_spacing;
                        drawLegendItemHiRes("Obstacles", sf::Color(128, 128, 128), item_y);
                        
                        // Stats box
                        if (stats.num_samples > 0) {
                            float stats_y = legend_y + legend_height + 10.0f;
                            float stats_height = 85.0f;
                            
                            sf::RectangleShape stats_bg(sf::Vector2f(legend_width, stats_height));
                            stats_bg.setPosition({legend_x, stats_y});
                            stats_bg.setFillColor(sf::Color(245, 250, 255, 230));
                            stats_bg.setOutlineColor(sf::Color(100, 100, 100));
                            stats_bg.setOutlineThickness(1.0f);
                            renderTexture.draw(stats_bg);
                            
                            sf::Text stats_title(font, "Statistics", 13);
                            stats_title.setFillColor(sf::Color::Black);
                            stats_title.setStyle(sf::Text::Bold);
                            stats_title.setPosition({legend_x + 8.0f, stats_y + 5.0f});
                            renderTexture.draw(stats_title);
                            
                            char buf[128];
                            float text_y = stats_y + 28.0f;
                            
                            snprintf(buf, sizeof(buf), "Samples: %zu", stats.num_samples);
                            sf::Text samples_text(font, buf, 11);
                            samples_text.setFillColor(sf::Color(50, 50, 50));
                            samples_text.setPosition({legend_x + 12.0f, text_y});
                            renderTexture.draw(samples_text);
                            
                            snprintf(buf, sizeof(buf), "Nodes: %zu", stats.num_nodes);
                            sf::Text nodes_text(font, buf, 11);
                            nodes_text.setFillColor(sf::Color(50, 50, 50));
                            nodes_text.setPosition({legend_x + 110.0f, text_y});
                            renderTexture.draw(nodes_text);
                            
                            text_y += 16.0f;
                            snprintf(buf, sizeof(buf), "Avg perturbation: %.2f", stats.avg_perturbation);
                            sf::Text avg_text(font, buf, 11);
                            avg_text.setFillColor(sf::Color(50, 50, 50));
                            avg_text.setPosition({legend_x + 12.0f, text_y});
                            renderTexture.draw(avg_text);
                            
                            text_y += 16.0f;
                            snprintf(buf, sizeof(buf), "Max perturbation: %.2f", stats.max_perturbation);
                            sf::Text max_text(font, buf, 11);
                            max_text.setFillColor(sf::Color(50, 50, 50));
                            max_text.setPosition({legend_x + 12.0f, text_y});
                            renderTexture.draw(max_text);
                        }
                    }
                    
                    renderTexture.display();
                    
                    // Save to file
                    sf::Image screenshot = renderTexture.getTexture().copyToImage();
                    std::string filename = "noise_distribution_" + std::to_string(save_counter++) + "_highres.png";
                    if (screenshot.saveToFile(filename)) {
                        std::cout << "Saved: " << filename << " (" << hi_res_width << "x" << hi_res_height << " pixels)\n";
                    } else {
                        std::cerr << "Failed to save image!\n";
                    }
                }
            }
        }
        
        window.clear(sf::Color(240, 240, 240));

        // 1. Draw obstacles
        drawObstacles(window, obstacles);

        // 2. Draw all noisy samples (faded blue cloud)
        for (const auto& sample : workspace_noisy_samples) {
            drawTrajectorySegments(window, sample, sf::Color(50, 50, 255, 30));
        }

        // 3. Draw base trajectory (red line)
        drawTrajectorySegments(window, workspace_base_trajectory, sf::Color(255, 0, 0, 255));
        
        for (const auto& node : workspace_base_trajectory.nodes) {
            drawNode(window, node, 3.0f, sf::Color(255, 100, 100));
        }
        
        // 4. Draw Start/Goal with labels
        if (!workspace_base_trajectory.nodes.empty()) {
            const auto& start_node = workspace_base_trajectory.nodes[workspace_base_trajectory.start_index];
            const auto& goal_node = workspace_base_trajectory.nodes[workspace_base_trajectory.goal_index];
            
            drawNode(window, start_node, 8.0f, sf::Color::Green);
            drawNode(window, goal_node, 8.0f, sf::Color::Red);
            
            // Draw Start/Goal labels
            if (font_loaded) {
                sf::Text start_label(font, "Start", 14);
                start_label.setFillColor(sf::Color(0, 150, 0));
                start_label.setStyle(sf::Text::Bold);
                start_label.setPosition({start_node.position(0) + 12.0f, start_node.position(1) - 8.0f});
                window.draw(start_label);
                
                sf::Text goal_label(font, "Goal", 14);
                goal_label.setFillColor(sf::Color(200, 0, 0));
                goal_label.setStyle(sf::Text::Bold);
                goal_label.setPosition({goal_node.position(0) + 12.0f, goal_node.position(1) - 8.0f});
                window.draw(goal_label);
            }
        }

        // 5. Draw Legend and Statistics
        if (font_loaded && show_legend) {
            // Legend box background
            float legend_x = 10.0f;
            float legend_y = 10.0f;
            float legend_width = 220.0f;
            float legend_height = 180.0f;
            
            sf::RectangleShape legend_bg(sf::Vector2f(legend_width, legend_height));
            legend_bg.setPosition({legend_x, legend_y});
            legend_bg.setFillColor(sf::Color(255, 255, 255, 230));
            legend_bg.setOutlineColor(sf::Color(100, 100, 100));
            legend_bg.setOutlineThickness(1.0f);
            window.draw(legend_bg);
            
            // Title
            sf::Text title(font, "Noise Distribution N(0, R^-1)", 13);
            title.setFillColor(sf::Color::Black);
            title.setStyle(sf::Text::Bold);
            title.setPosition({legend_x + 8.0f, legend_y + 5.0f});
            window.draw(title);
            
            // Separator line
            sf::RectangleShape separator(sf::Vector2f(legend_width - 16.0f, 1.0f));
            separator.setPosition({legend_x + 8.0f, legend_y + 28.0f});
            separator.setFillColor(sf::Color(150, 150, 150));
            window.draw(separator);
            
            float item_y = legend_y + 35.0f;
            float item_spacing = 22.0f;
            
            // Lambda for drawing legend items
            auto drawLegendItem = [&](const std::string& label, sf::Color color, float y, bool is_line = false) {
                if (is_line) {
                    sf::RectangleShape line(sf::Vector2f(20.0f, 3.0f));
                    line.setPosition({legend_x + 12.0f, y + 6.0f});
                    line.setFillColor(color);
                    window.draw(line);
                } else {
                    sf::CircleShape marker(6.0f);
                    marker.setPosition({legend_x + 12.0f, y + 2.0f});
                    marker.setFillColor(color);
                    window.draw(marker);
                }
                
                sf::Text text(font, label, 12);
                text.setFillColor(sf::Color(50, 50, 50));
                text.setPosition({legend_x + 40.0f, y});
                window.draw(text);
            };
            
            drawLegendItem("Noise samples", sf::Color(50, 50, 255, 150), item_y, true);
            item_y += item_spacing;
            drawLegendItem("Mean trajectory", sf::Color(255, 0, 0), item_y, true);
            item_y += item_spacing;
            drawLegendItem("Start position", sf::Color::Green, item_y);
            item_y += item_spacing;
            drawLegendItem("Goal position", sf::Color::Red, item_y);
            item_y += item_spacing;
            drawLegendItem("Obstacles", sf::Color(128, 128, 128), item_y);
            
            // Statistics section
            if (stats.num_samples > 0) {
                float stats_y = legend_y + legend_height + 10.0f;
                float stats_height = 85.0f;
                
                sf::RectangleShape stats_bg(sf::Vector2f(legend_width, stats_height));
                stats_bg.setPosition({legend_x, stats_y});
                stats_bg.setFillColor(sf::Color(245, 250, 255, 230));
                stats_bg.setOutlineColor(sf::Color(100, 100, 100));
                stats_bg.setOutlineThickness(1.0f);
                window.draw(stats_bg);
                
                sf::Text stats_title(font, "Statistics", 13);
                stats_title.setFillColor(sf::Color::Black);
                stats_title.setStyle(sf::Text::Bold);
                stats_title.setPosition({legend_x + 8.0f, stats_y + 5.0f});
                window.draw(stats_title);
                
                char buf[128];
                float text_y = stats_y + 28.0f;
                
                snprintf(buf, sizeof(buf), "Samples: %zu", stats.num_samples);
                sf::Text samples_text(font, buf, 11);
                samples_text.setFillColor(sf::Color(50, 50, 50));
                samples_text.setPosition({legend_x + 12.0f, text_y});
                window.draw(samples_text);
                
                snprintf(buf, sizeof(buf), "Nodes: %zu", stats.num_nodes);
                sf::Text nodes_text(font, buf, 11);
                nodes_text.setFillColor(sf::Color(50, 50, 50));
                nodes_text.setPosition({legend_x + 110.0f, text_y});
                window.draw(nodes_text);
                
                text_y += 16.0f;
                snprintf(buf, sizeof(buf), "Avg perturbation: %.2f", stats.avg_perturbation);
                sf::Text avg_text(font, buf, 11);
                avg_text.setFillColor(sf::Color(50, 50, 50));
                avg_text.setPosition({legend_x + 12.0f, text_y});
                window.draw(avg_text);
                
                text_y += 16.0f;
                snprintf(buf, sizeof(buf), "Max perturbation: %.2f", stats.max_perturbation);
                sf::Text max_text(font, buf, 11);
                max_text.setFillColor(sf::Color(50, 50, 50));
                max_text.setPosition({legend_x + 12.0f, text_y});
                window.draw(max_text);
            }
        }
        
        // 6. Draw keyboard hint at bottom
        if (font_loaded) {
            sf::Text hint(font, "P: Save high-res PNG (3x) | L: Toggle legend | ESC: Exit", 11);
            hint.setFillColor(sf::Color(100, 100, 100));
            hint.setPosition({10.0f, MAP_HEIGHT - 25.0f});
            window.draw(hint);
        }

        window.display();
    }
}


int main() {
    std::cout << "========================================\n";
    std::cout << "  Trajectory Noise Visualization\n";
    std::cout << "  (Smoothness Distribution N(0, R^-1))\n";
    std::cout << "========================================\n\n";

    // --- 1. Load Configuration ---
    std::string config_file = "../configs/config.yaml";
    YAML::Node config;
    
    std::cout << "=== Load from YAML ===\n\n";
    
    try {
        config = YAML::LoadFile(config_file);
        std::cout << "Loaded configuration from: " << config_file << "\n";
    } catch (const YAML::BadFile& e) {
        std::cerr << "Error: Could not open config file: " << config_file << "\n";
        return 1;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing config file: " << e.what() << "\n";
        return 1;
    }

    // --- 2. Create Task (handles obstacle generation and management) ---
    std::cout << "\n=== Creating Collision Avoidance Task ===\n";
    auto task = std::make_shared<pce::CollisionAvoidanceTask>(config);

    // Get obstacle map from task
    auto obstacle_map = task->getObstacleMap();
    std::cout << "Task created with " << obstacle_map->size() << " obstacles\n";

    // --- 3. Create and Initialize Planner ---
    std::cout << "\n=== Creating PCEM Planner ===\n";
    PCEConfig pce_config;
    if (!pce_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load PCE configuration from file\n";
        return -1;
    }
    ProximalCrossEntropyMotionPlanner planner(task);
    
    // Initialize planner (loads config and sets up trajectory)
    std::cout << "\n=== Initializing Planner ===\n";
    if (!planner.initialize(pce_config)) {
        std::cerr << "Error: Planner initialization failed\n";
        return 1;
    }   
    
    // Get trajectory and environment
    const Trajectory& config_trajectory = planner.getCurrentTrajectory();
    const size_t N = config_trajectory.nodes.size();
    const size_t D = config_trajectory.dimensions();
    
    // Get obstacles from task
    std::vector<ObstacleND> obstacles;
    if (task) {
        obstacles = task->getObstacles();
    }
    
    std::cout << "Configuration:\n";
    std::cout << "  Trajectory nodes: " << N << "\n";
    std::cout << "  Dimensions: " << D << "\n";
    std::cout << "  Obstacles: " << obstacles.size() << "\n\n";

    // --- 6. Setup Random Number Generator ---
    std::mt19937 rng;
    unsigned int seed = pce_config.random_seed;
    rng.seed(seed);
    std::cout << "Random seed: " << seed << "\n\n";

    int num_samples = pce_config.num_samples;

    // --- 7. Generate Noise Samples ---
    std::cout << "Generating " << num_samples << " noise samples from N(0, R^-1)...\n";
    
    // Access base class method for sampling noise
    std::vector<Eigen::MatrixXf> epsilon_samples;
    
    // For now, assuming sampleNoiseMatrices is made public or accessible
    // If it's protected, you'll need to add a public wrapper method in the planner
    epsilon_samples = planner.sampleNoiseMatrices(num_samples, N, D);
    
    std::cout << "Noise sampling complete!\n\n";

    // --- 8. Create Perturbed Trajectories ---
    std::cout << "Creating perturbed trajectories...\n";
    
    // Get base trajectory as matrix
    Eigen::MatrixXf Y_base(D, N);
    for (size_t i = 0; i < N; ++i) {
        Y_base.col(i) = config_trajectory.nodes[i].position;
    }
    
    std::vector<Trajectory> config_noisy_samples;
    config_noisy_samples.reserve(num_samples);
    
    for (size_t m = 0; m < num_samples; ++m) {
        // Create perturbed trajectory
        Trajectory perturbed_traj = config_trajectory; // Copy base
        Eigen::MatrixXf Y_perturbed = Y_base + epsilon_samples[m];
        
        // Update positions
        for (size_t i = 0; i < N; ++i) {
            perturbed_traj.nodes[i].position = Y_perturbed.col(i);
        }
        
        config_noisy_samples.push_back(perturbed_traj);
    }
    
    std::cout << "Perturbed trajectories created!\n\n";

    // --- 9. Apply Forward Kinematics (Config → Workspace) ---
    std::cout << "Applying forward kinematics to workspace...\n";
    
    auto fk = planner.getForwardKinematics();
    Trajectory workspace_base = fk->apply(config_trajectory);
    
    std::vector<Trajectory> workspace_noisy_samples;
    workspace_noisy_samples.reserve(num_samples);
    
    for (size_t m = 0; m < num_samples; ++m) {
        workspace_noisy_samples.push_back(fk->apply(config_noisy_samples[m]));
    }
    std::cout << "Workspace transformation complete!\n\n";

    // --- 10. Compute Noise Statistics ---
    NoiseStats stats;
    stats.num_samples = num_samples;
    stats.num_nodes = N;
    
    float total_perturbation = 0.0f;
    float max_perturbation = 0.0f;
    
    for (size_t m = 0; m < num_samples; ++m) {
        for (size_t i = 0; i < N; ++i) {
            Eigen::VectorXf diff = workspace_noisy_samples[m].nodes[i].position 
                                  - workspace_base.nodes[i].position;
            float perturbation = diff.norm();
            
            total_perturbation += perturbation;
            max_perturbation = std::max(max_perturbation, perturbation);
        }
    }
    
    stats.avg_perturbation = total_perturbation / (num_samples * N);
    stats.max_perturbation = max_perturbation;
    
    std::cout << "=== Noise Statistics (Workspace) ===\n";
    std::cout << "  Average perturbation: " << stats.avg_perturbation << " units\n";
    std::cout << "  Maximum perturbation: " << stats.max_perturbation << " units\n";
    std::cout << "  Total samples: " << stats.num_samples << "\n";
    std::cout << "  Nodes per trajectory: " << stats.num_nodes << "\n\n";
    
    std::cout << "Visualization Legend:\n";
    std::cout << "  Blue cloud    = Noise distribution N(0, R^-1)\n";
    std::cout << "  Red line      = Base trajectory (mean)\n";
    std::cout << "  Green dot     = Start position\n";
    std::cout << "  Red dot       = Goal position\n";
    std::cout << "  Gray circles  = Obstacles\n\n";
    
    std::cout << "Opening visualization window...\n";

    // --- 11. Visualize (now with stats) ---
    visualizeNoise(obstacles, workspace_base, workspace_noisy_samples, stats);

    std::cout << "\nVisualization closed. Exiting.\n";
    return 0;
}