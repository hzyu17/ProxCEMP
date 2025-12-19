#include <SFML/Graphics.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "../include/PCEMotionPlanner.h"
#include "../examples/CollisionAvoidanceTask.h"
#include "../include/visualization.h"

// --- Constants ---
namespace {
    constexpr float SAVE_SCALE = 4.0f;  // 4x scale for ~400 DPI print quality
    constexpr float LEGEND_X = 15.0f;
    constexpr float LEGEND_Y = 15.0f;
    constexpr float LEGEND_WIDTH = 280.0f;
    constexpr float LEGEND_HEIGHT = 220.0f;
    constexpr float STATS_HEIGHT = 110.0f;
    constexpr float ITEM_SPACING = 28.0f;
    constexpr float LEGEND_PADDING = 12.0f;
    constexpr float LEGEND_BORDER_THICKNESS = 1.5f;
    
    // Font sizes for journal quality
    constexpr unsigned int TITLE_FONT_SIZE = 16;
    constexpr unsigned int ITEM_FONT_SIZE = 14;
    constexpr unsigned int STATS_FONT_SIZE = 13;
    constexpr unsigned int LABEL_FONT_SIZE = 15;
    
    // Marker/line sizes
    constexpr float LEGEND_LINE_WIDTH = 28.0f;
    constexpr float LEGEND_LINE_THICKNESS = 4.0f;
    constexpr float LEGEND_MARKER_RADIUS = 8.0f;
    
    const sf::Color BG_COLOR(255, 255, 255);  // White background for publication
    const sf::Color SAMPLE_COLOR(31, 119, 180, 40);  // Matplotlib blue with transparency
    const sf::Color BASE_TRAJ_COLOR(214, 39, 40);    // Matplotlib red
    const sf::Color BASE_NODE_COLOR(214, 39, 40);
    const sf::Color LEGEND_BG_COLOR(255, 255, 255, 245);
    const sf::Color STATS_BG_COLOR(250, 250, 250, 245);
    const sf::Color TEXT_COLOR(20, 20, 20);          // Near-black for readability
    const sf::Color BORDER_COLOR(60, 60, 60);
    const sf::Color START_COLOR(44, 160, 44);        // Matplotlib green
    const sf::Color GOAL_COLOR(214, 39, 40);         // Matplotlib red
    const sf::Color OBSTACLE_COLOR(127, 127, 127);   // Neutral gray
}

// --- Statistics Struct ---
struct NoiseStats {
    float avg_perturbation = 0.0f;
    float max_perturbation = 0.0f;
    size_t num_samples = 0;
    size_t num_nodes = 0;
};

// --- Helper Functions ---
namespace {

bool tryLoadFont(sf::Font& font) {
    // Prefer professional fonts suitable for academic publications
    const std::vector<std::string> font_paths = {
        // Liberation fonts (metrically compatible with Arial/Times)
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        // Nimbus (PostScript-compatible, good for publications)
        "/usr/share/fonts/type1/gsfonts/n019003l.pfb",  // Nimbus Sans
        // DejaVu (good Unicode coverage, clean)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        // Ubuntu (modern, readable)
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        // Fallbacks
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/opentype/fira/FiraSans-Regular.otf"
    };
    
    for (const auto& path : font_paths) {
        if (font.openFromFile(path)) {
            return true;
        }
    }
    std::cerr << "Warning: Could not load font. Text labels will not be displayed.\n";
    return false;
}

void drawLegendItem(sf::RenderTarget& target, const sf::Font& font,
                    const std::string& label, sf::Color color, 
                    float x, float y, bool is_line) {
    if (is_line) {
        sf::RectangleShape line(sf::Vector2f(LEGEND_LINE_WIDTH, LEGEND_LINE_THICKNESS));
        line.setPosition({x + LEGEND_PADDING, y + 8.0f});
        line.setFillColor(color);
        target.draw(line);
    } else {
        sf::CircleShape marker(LEGEND_MARKER_RADIUS);
        marker.setPosition({x + LEGEND_PADDING, y + 4.0f});
        marker.setFillColor(color);
        marker.setOutlineColor(sf::Color(BORDER_COLOR.r, BORDER_COLOR.g, BORDER_COLOR.b, 100));
        marker.setOutlineThickness(1.0f);
        target.draw(marker);
    }
    
    sf::Text text(font, label, ITEM_FONT_SIZE);
    text.setFillColor(TEXT_COLOR);
    text.setPosition({x + LEGEND_PADDING + LEGEND_LINE_WIDTH + 12.0f, y + 1.0f});
    target.draw(text);
}

void drawLegendBox(sf::RenderTarget& target, const sf::Font& font,
                   float x, float y) {
    // Background with shadow effect
    sf::RectangleShape shadow(sf::Vector2f(LEGEND_WIDTH, LEGEND_HEIGHT));
    shadow.setPosition({x + 2.0f, y + 2.0f});
    shadow.setFillColor(sf::Color(0, 0, 0, 30));
    target.draw(shadow);
    
    sf::RectangleShape bg(sf::Vector2f(LEGEND_WIDTH, LEGEND_HEIGHT));
    bg.setPosition({x, y});
    bg.setFillColor(LEGEND_BG_COLOR);
    bg.setOutlineColor(BORDER_COLOR);
    bg.setOutlineThickness(LEGEND_BORDER_THICKNESS);
    target.draw(bg);
    
    // Title
    sf::Text title(font, "Noise Distribution N(0, R^{-1})", TITLE_FONT_SIZE);  // Proper superscript
    title.setFillColor(TEXT_COLOR);
    title.setStyle(sf::Text::Bold);
    title.setPosition({x + LEGEND_PADDING, y + 10.0f});
    target.draw(title);
    
    // Separator
    sf::RectangleShape separator(sf::Vector2f(LEGEND_WIDTH - 2 * LEGEND_PADDING, 1.5f));
    separator.setPosition({x + LEGEND_PADDING, y + 38.0f});
    separator.setFillColor(sf::Color(120, 120, 120));
    target.draw(separator);
    
    // Legend items with consistent colors
    float item_y = y + 50.0f;
    drawLegendItem(target, font, "Sampled trajectories", sf::Color(31, 119, 180, 200), x, item_y, true);
    item_y += ITEM_SPACING;
    drawLegendItem(target, font, "Mean trajectory", BASE_TRAJ_COLOR, x, item_y, true);
    item_y += ITEM_SPACING;
    drawLegendItem(target, font, "Start position", START_COLOR, x, item_y, false);
    item_y += ITEM_SPACING;
    drawLegendItem(target, font, "Goal position", GOAL_COLOR, x, item_y, false);
    item_y += ITEM_SPACING;
    drawLegendItem(target, font, "Obstacles", OBSTACLE_COLOR, x, item_y, false);
}

void drawStatsBox(sf::RenderTarget& target, const sf::Font& font,
                  const NoiseStats& stats, float x, float y) {
    if (stats.num_samples == 0) return;
    
    // Shadow
    sf::RectangleShape shadow(sf::Vector2f(LEGEND_WIDTH, STATS_HEIGHT));
    shadow.setPosition({x + 2.0f, y + 2.0f});
    shadow.setFillColor(sf::Color(0, 0, 0, 30));
    target.draw(shadow);
    
    // Background
    sf::RectangleShape bg(sf::Vector2f(LEGEND_WIDTH, STATS_HEIGHT));
    bg.setPosition({x, y});
    bg.setFillColor(STATS_BG_COLOR);
    bg.setOutlineColor(BORDER_COLOR);
    bg.setOutlineThickness(LEGEND_BORDER_THICKNESS);
    target.draw(bg);
    
    // Title
    sf::Text title(font, "Statistics", TITLE_FONT_SIZE);
    title.setFillColor(TEXT_COLOR);
    title.setStyle(sf::Text::Bold);
    title.setPosition({x + LEGEND_PADDING, y + 10.0f});
    target.draw(title);
    
    // Separator
    sf::RectangleShape separator(sf::Vector2f(LEGEND_WIDTH - 2 * LEGEND_PADDING, 1.5f));
    separator.setPosition({x + LEGEND_PADDING, y + 38.0f});
    separator.setFillColor(sf::Color(120, 120, 120));
    target.draw(separator);
    
    // Stats text with better formatting
    char buf[128];
    float text_y = y + 48.0f;
    float line_spacing = 20.0f;
    
    snprintf(buf, sizeof(buf), "Samples: %zu", stats.num_samples);
    sf::Text samples_text(font, buf, STATS_FONT_SIZE);
    samples_text.setFillColor(TEXT_COLOR);
    samples_text.setPosition({x + LEGEND_PADDING, text_y});
    target.draw(samples_text);
    
    snprintf(buf, sizeof(buf), "Nodes: %zu", stats.num_nodes);
    sf::Text nodes_text(font, buf, STATS_FONT_SIZE);
    nodes_text.setFillColor(TEXT_COLOR);
    nodes_text.setPosition({x + LEGEND_WIDTH / 2.0f, text_y});
    target.draw(nodes_text);
    
    text_y += line_spacing;
    snprintf(buf, sizeof(buf), "Avg. perturbation: %.2f px", stats.avg_perturbation);
    sf::Text avg_text(font, buf, STATS_FONT_SIZE);
    avg_text.setFillColor(TEXT_COLOR);
    avg_text.setPosition({x + LEGEND_PADDING, text_y});
    target.draw(avg_text);
    
    text_y += line_spacing;
    snprintf(buf, sizeof(buf), "Max. perturbation: %.2f px", stats.max_perturbation);
    sf::Text max_text(font, buf, STATS_FONT_SIZE);
    max_text.setFillColor(TEXT_COLOR);
    max_text.setPosition({x + LEGEND_PADDING, text_y});
    target.draw(max_text);
}

void drawScene(sf::RenderTarget& target,
               const std::vector<ObstacleND>& obstacles,
               const Trajectory& base_trajectory,
               const std::vector<Trajectory>& noisy_samples,
               const sf::Font* font,
               bool show_legend,
               const NoiseStats& stats) {
    // 1. Obstacles
    drawObstacles(target, obstacles);
    
    // 2. Noisy samples
    for (const auto& sample : noisy_samples) {
        drawTrajectorySegments(target, sample, SAMPLE_COLOR);
    }
    
    // 3. Base trajectory
    drawTrajectorySegments(target, base_trajectory, BASE_TRAJ_COLOR);
    for (const auto& node : base_trajectory.nodes) {
        drawNode(target, node, 3.0f, BASE_NODE_COLOR);
    }
    
    // 4. Start/Goal markers and labels
    if (!base_trajectory.nodes.empty()) {
        const auto& start_node = base_trajectory.nodes[base_trajectory.start_index];
        const auto& goal_node = base_trajectory.nodes[base_trajectory.goal_index];
        
        // Larger markers with outlines for visibility
        sf::CircleShape start_marker(10.0f);
        start_marker.setPosition({start_node.position(0) - 10.0f, start_node.position(1) - 10.0f});
        start_marker.setFillColor(START_COLOR);
        start_marker.setOutlineColor(sf::Color(20, 100, 20));
        start_marker.setOutlineThickness(2.0f);
        target.draw(start_marker);
        
        sf::CircleShape goal_marker(10.0f);
        goal_marker.setPosition({goal_node.position(0) - 10.0f, goal_node.position(1) - 10.0f});
        goal_marker.setFillColor(GOAL_COLOR);
        goal_marker.setOutlineColor(sf::Color(150, 20, 20));
        goal_marker.setOutlineThickness(2.0f);
        target.draw(goal_marker);
        
        if (font) {
            // Text with shadow for better visibility on any background
            auto drawLabelWithShadow = [&](const std::string& text, float x, float y, sf::Color color) {
                sf::Text shadow(*font, text, LABEL_FONT_SIZE);
                shadow.setFillColor(sf::Color(255, 255, 255, 200));
                shadow.setStyle(sf::Text::Bold);
                shadow.setPosition({x + 1.0f, y + 1.0f});
                target.draw(shadow);
                
                sf::Text label(*font, text, LABEL_FONT_SIZE);
                label.setFillColor(color);
                label.setStyle(sf::Text::Bold);
                label.setPosition({x, y});
                target.draw(label);
            };
            
            drawLabelWithShadow("Start", start_node.position(0) + 14.0f, start_node.position(1) - 10.0f, 
                               sf::Color(20, 120, 20));
            drawLabelWithShadow("Goal", goal_node.position(0) + 14.0f, goal_node.position(1) - 10.0f,
                               sf::Color(180, 20, 20));
        }
    }
    
    // 5. Legend and statistics
    if (font && show_legend) {
        drawLegendBox(target, *font, LEGEND_X, LEGEND_Y);
        drawStatsBox(target, *font, stats, LEGEND_X, LEGEND_Y + LEGEND_HEIGHT + 15.0f);
    }
}

bool saveImage(const std::vector<ObstacleND>& obstacles,
               const Trajectory& base_trajectory,
               const std::vector<Trajectory>& noisy_samples,
               const sf::Font* font,
               bool show_legend,
               const NoiseStats& stats,
               int& save_counter,
               float scale = 1.0f) {
    unsigned int width = static_cast<unsigned int>(MAP_WIDTH * scale);
    unsigned int height = static_cast<unsigned int>(MAP_HEIGHT * scale);
    
    sf::RenderTexture render_texture;
    if (!render_texture.resize({width, height})) {
        std::cerr << "Failed to create render texture!\n";
        return false;
    }
    
    render_texture.clear(BG_COLOR);
    
    sf::View scaled_view(sf::FloatRect({0.f, 0.f}, {(float)MAP_WIDTH, (float)MAP_HEIGHT}));
    scaled_view.setViewport(sf::FloatRect({0.f, 0.f}, {1.f, 1.f}));
    render_texture.setView(scaled_view);
    
    drawScene(render_texture, obstacles, base_trajectory, noisy_samples, 
              font, show_legend, stats);
    
    render_texture.display();
    
    sf::Image screenshot = render_texture.getTexture().copyToImage();
    
    std::string scale_suffix = (scale > 1.0f) ? "_highres" : "";
    std::string filename = "noise_distribution_" + std::to_string(save_counter++) + scale_suffix + ".png";
    
    if (screenshot.saveToFile(filename)) {
        std::cout << "Saved: " << filename << " (" << width << "x" << height << " pixels)\n";
        return true;
    }
    
    std::cerr << "Failed to save image!\n";
    return false;
}

} // anonymous namespace

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
    
    // Export info
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
    
    // Export noisy samples
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

    sf::Font font;
    bool font_loaded = tryLoadFont(font);

    std::cout << "\n=== Controls ===\n"
              << "ESC: Exit\n"
              << "L:   Toggle legend\n"
              << "S:   Save as PNG (1x scale)\n"
              << "P:   Save as high-res PNG (4x scale)\n"
              << "================\n\n";

    bool show_legend = true;
    int save_counter = 0;

    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            
            if (const auto* key_event = event->getIf<sf::Event::KeyPressed>()) {
                switch (key_event->code) {
                    case sf::Keyboard::Key::Escape:
                        window.close();
                        break;
                    case sf::Keyboard::Key::L:
                        show_legend = !show_legend;
                        break;
                    case sf::Keyboard::Key::S:
                        saveImage(obstacles, workspace_base_trajectory, 
                                  workspace_noisy_samples,
                                  font_loaded ? &font : nullptr,
                                  show_legend, stats, save_counter, 1.0f);
                        break;
                    case sf::Keyboard::Key::P:
                        saveImage(obstacles, workspace_base_trajectory, 
                                  workspace_noisy_samples,
                                  font_loaded ? &font : nullptr,
                                  show_legend, stats, save_counter, SAVE_SCALE);
                        break;
                    default:
                        break;
                }
            }
        }
        
        window.clear(BG_COLOR);
        
        drawScene(window, obstacles, workspace_base_trajectory, workspace_noisy_samples,
                  font_loaded ? &font : nullptr, show_legend, stats);
        
        // Keyboard hint
        if (font_loaded) {
            sf::Text hint(font, "S: Save PNG | P: Save high-res PNG (4x) | L: Toggle legend | ESC: Exit", 11);
            hint.setFillColor(sf::Color(100, 100, 100));
            hint.setPosition({10.0f, MAP_HEIGHT - 25.0f});
            window.draw(hint);
        }

        window.display();
    }
}

int main() {
    std::cout << "========================================\n"
              << "  Trajectory Noise Visualization\n"
              << "  (Smoothness Distribution N(0, R^-1))\n"
              << "========================================\n\n";

    // --- Load Configuration ---
    const std::string config_file = "../configs/config.yaml";
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

    // --- Create Task ---
    std::cout << "\n=== Creating Collision Avoidance Task ===\n";
    auto task = std::make_shared<pce::CollisionAvoidanceTask>(config);
    auto obstacle_map = task->getObstacleMap();
    std::cout << "Task created with " << obstacle_map->size() << " obstacles\n";

    // --- Create and Initialize Planner ---
    std::cout << "\n=== Creating PCEM Planner ===\n";
    PCEConfig pce_config;
    if (!pce_config.loadFromFile(config_file)) {
        std::cerr << "Failed to load PCE configuration from file\n";
        return 1;
    }
    
    ProximalCrossEntropyMotionPlanner planner(task);
    
    std::cout << "\n=== Initializing Planner ===\n";
    if (!planner.initialize(pce_config)) {
        std::cerr << "Error: Planner initialization failed\n";
        return 1;
    }   
    
    const Trajectory& config_trajectory = planner.getCurrentTrajectory();
    const size_t N = config_trajectory.nodes.size();
    const size_t D = config_trajectory.dimensions();
    
    std::vector<ObstacleND> obstacles = task->getObstacles();
    
    std::cout << "Configuration:\n"
              << "  Trajectory nodes: " << N << "\n"
              << "  Dimensions: " << D << "\n"
              << "  Obstacles: " << obstacles.size() << "\n\n";

    // --- Setup RNG ---
    std::mt19937 rng(pce_config.random_seed);
    std::cout << "Random seed: " << pce_config.random_seed << "\n\n";

    const int num_samples = pce_config.num_samples;

    // --- Generate Noise Samples ---
    std::cout << "Generating " << num_samples << " noise samples from N(0, R^-1)...\n";
    auto epsilon_samples = planner.sampleNoiseMatrices(num_samples, N, D);
    std::cout << "Noise sampling complete!\n\n";

    // --- Create Perturbed Trajectories ---
    std::cout << "Creating perturbed trajectories...\n";
    
    Eigen::MatrixXf Y_base(D, N);
    for (size_t i = 0; i < N; ++i) {
        Y_base.col(i) = config_trajectory.nodes[i].position;
    }
    
    std::vector<Trajectory> config_noisy_samples;
    config_noisy_samples.reserve(num_samples);
    
    for (int m = 0; m < num_samples; ++m) {
        Trajectory perturbed_traj = config_trajectory;
        Eigen::MatrixXf Y_perturbed = Y_base + epsilon_samples[m];
        
        for (size_t i = 0; i < N; ++i) {
            perturbed_traj.nodes[i].position = Y_perturbed.col(i);
        }
        config_noisy_samples.push_back(std::move(perturbed_traj));
    }
    std::cout << "Perturbed trajectories created!\n\n";

    // --- Apply Forward Kinematics ---
    std::cout << "Applying forward kinematics to workspace...\n";
    
    auto fk = planner.getForwardKinematics();
    Trajectory workspace_base = fk->apply(config_trajectory);
    
    std::vector<Trajectory> workspace_noisy_samples;
    workspace_noisy_samples.reserve(num_samples);
    for (const auto& sample : config_noisy_samples) {
        workspace_noisy_samples.push_back(sample);
    }
    std::cout << "Workspace transformation complete!\n\n";

    // --- Compute Statistics ---
    NoiseStats stats;
    stats.num_samples = num_samples;
    stats.num_nodes = N;
    
    float total_perturbation = 0.0f;
    
    for (size_t m = 0; m < workspace_noisy_samples.size(); ++m) {
        for (size_t i = 0; i < N; ++i) {
            float perturbation = (workspace_noisy_samples[m].nodes[i].position 
                                 - workspace_base.nodes[i].position).norm();
            total_perturbation += perturbation;
            stats.max_perturbation = std::max(stats.max_perturbation, perturbation);
        }
    }
    stats.avg_perturbation = total_perturbation / (num_samples * N);
    
    std::cout << "=== Noise Statistics (Workspace) ===\n"
              << "  Average perturbation: " << stats.avg_perturbation << " units\n"
              << "  Maximum perturbation: " << stats.max_perturbation << " units\n"
              << "  Total samples: " << stats.num_samples << "\n"
              << "  Nodes per trajectory: " << stats.num_nodes << "\n\n";
    
    std::cout << "Visualization Legend:\n"
              << "  Blue cloud    = Noise distribution N(0, R^-1)\n"
              << "  Red line      = Base trajectory (mean)\n"
              << "  Green dot     = Start position\n"
              << "  Red dot       = Goal position\n"
              << "  Gray circles  = Obstacles\n\n"
              << "Opening visualization window...\n";

    // --- Visualize ---
    visualizeNoise(obstacles, workspace_base, workspace_noisy_samples, stats);

    std::cout << "\nVisualization closed. Exiting.\n";
    return 0;
}