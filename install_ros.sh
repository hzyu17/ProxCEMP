#!/bin/bash

# MoveIt ROS Benchmarks Installation Script
# For Ubuntu 20.04 + ROS Noetic

set -e  # Exit on error

echo "=========================================="
echo "MoveIt ROS Benchmarks Installation Script"
echo "Ubuntu 20.04 + ROS Noetic"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running Ubuntu 20.04
echo "Checking Ubuntu version..."
if [ "$(lsb_release -sc)" != "focal" ]; then
    print_warning "This script is designed for Ubuntu 20.04 (Focal). You are running $(lsb_release -sc)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
print_status "Ubuntu version check passed"
echo ""

# Update system
echo "Updating system packages..."
sudo apt update
print_status "System updated"
echo ""

# Install ROS Noetic if not already installed
if [ ! -f /opt/ros/noetic/setup.bash ]; then
    echo "Installing ROS Noetic..."
    
    # Setup sources
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    
    # Setup keys
    sudo apt install curl -y
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    
    # Install ROS
    sudo apt update
    sudo apt install ros-noetic-desktop-full -y
    
    # Setup environment
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source /opt/ros/noetic/setup.bash
    
    # Install dependencies
    sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y
    
    # Initialize rosdep
    if [ ! -d /etc/ros/rosdep ]; then
        sudo rosdep init
    fi
    rosdep update
    
    print_status "ROS Noetic installed"
else
    print_status "ROS Noetic already installed"
fi
echo ""

# Source ROS
source /opt/ros/noetic/setup.bash

# Install MoveIt and required packages
echo "Installing MoveIt and planners..."
sudo apt install -y \
    ros-noetic-moveit \
    ros-noetic-moveit-planners-ompl \
    ros-noetic-moveit-planners-stomp \
    ros-noetic-moveit-planners-chomp \
    ros-noetic-moveit-ros-benchmarks \
    ros-noetic-warehouse-ros \
    ros-noetic-warehouse-ros-mongo \
    ros-noetic-moveit-visual-tools \
    ros-noetic-moveit-commander \
    ros-noetic-trac-ik-kinematics-plugin

print_status "MoveIt packages installed"
echo ""

# Install MongoDB
echo "Installing MongoDB..."
if ! command -v mongo &> /dev/null; then
    sudo apt install -y mongodb
    
    # Start and enable MongoDB
    sudo systemctl start mongodb
    sudo systemctl enable mongodb
    
    print_status "MongoDB installed and started"
else
    print_status "MongoDB already installed"
    
    # Make sure it's running
    sudo systemctl start mongodb
    print_status "MongoDB started"
fi
echo ""

# Verify MongoDB is running
if systemctl is-active --quiet mongodb; then
    print_status "MongoDB is running"
else
    print_error "MongoDB is not running"
    echo "Try starting it manually: sudo systemctl start mongodb"
fi
echo ""

# Install Python dependencies
echo "Installing Python packages..."
sudo apt install -y python3-pip

pip3 install --user \
    pymongo \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scipy

print_status "Python packages installed"
echo ""

# Create catkin workspace if it doesn't exist
if [ ! -d ~/catkin_ws ]; then
    echo "Creating catkin workspace..."
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws
    catkin_make
    
    # Add to bashrc
    if ! grep -q "source ~/catkin_ws/devel/setup.bash" ~/.bashrc; then
        echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    fi
    
    print_status "Catkin workspace created"
else
    print_status "Catkin workspace already exists"
fi
echo ""

# Install example robot (Panda) for testing
echo "Installing Panda robot for testing..."
sudo apt install -y \
    ros-noetic-panda-moveit-config \
    ros-noetic-franka-description

print_status "Panda robot installed"
echo ""

# Setup environment variables
echo "Setting up environment variables..."
if ! grep -q "export MONGO_PORT=27017" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# MoveIt Benchmark Environment
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
export MONGO_PORT=27017
EOF
    print_status "Environment variables added to ~/.bashrc"
else
    print_status "Environment variables already configured"
fi
echo ""

# Source workspace
source ~/.bashrc 2>/dev/null || true
source /opt/ros/noetic/setup.bash
if [ -f ~/catkin_ws/devel/setup.bash ]; then
    source ~/catkin_ws/devel/setup.bash
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""

# Verification
echo "Verifying installation..."
echo ""

# Check ROS
if [ -f /opt/ros/noetic/setup.bash ]; then
    print_status "ROS Noetic"
else
    print_error "ROS Noetic"
fi

# Check MoveIt
if rospack find moveit_core &> /dev/null; then
    print_status "MoveIt"
else
    print_error "MoveIt"
fi

# Check Benchmarks
if rospack find moveit_ros_benchmarks &> /dev/null; then
    print_status "MoveIt Benchmarks"
else
    print_error "MoveIt Benchmarks"
fi

# Check OMPL
if rospack find moveit_planners_ompl &> /dev/null; then
    print_status "OMPL Planner"
else
    print_error "OMPL Planner"
fi

# Check STOMP
if rospack find moveit_planners_stomp &> /dev/null; then
    print_status "STOMP Planner"
else
    print_error "STOMP Planner"
fi

# Check MongoDB
if systemctl is-active --quiet mongodb; then
    print_status "MongoDB (running)"
else
    print_error "MongoDB (not running)"
fi

# Check Warehouse
if rospack find warehouse_ros_mongo &> /dev/null; then
    print_status "Warehouse ROS"
else
    print_error "Warehouse ROS"
fi

# Check Python
if python3 -c "import pymongo" 2>/dev/null; then
    print_status "Python pymongo"
else
    print_error "Python pymongo"
fi

if python3 -c "import pandas" 2>/dev/null; then
    print_status "Python pandas"
else
    print_error "Python pandas"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Source your environment:"
echo "   source ~/.bashrc"
echo ""
echo "2. Test the installation:"
echo "   roslaunch panda_moveit_config demo.launch"
echo ""
echo "3. Read the documentation:"
echo "   See INSTALLATION_GUIDE.md for detailed information"
echo ""
echo "4. Start benchmarking:"
echo "   See QUICKSTART.md for your first benchmark"
echo ""
echo "=========================================="
