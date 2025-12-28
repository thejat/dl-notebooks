#!/bin/bash
# Miniconda setup script for Linux and macOS

set -e

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected OS: $OS, Architecture: $ARCH"

# Set download URL based on OS and architecture
case "$OS" in
    Linux)
        if [ "$ARCH" = "x86_64" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif [ "$ARCH" = "aarch64" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else
            echo "Unsupported Linux architecture: $ARCH"
            exit 1
        fi
        ;;
    Darwin)
        if [ "$ARCH" = "x86_64" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif [ "$ARCH" = "arm64" ]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            echo "Unsupported macOS architecture: $ARCH"
            exit 1
        fi
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Please install Miniconda manually from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
        ;;
esac

echo "Downloading Miniconda from: $MINICONDA_URL"

# Create directory and download
mkdir -p ~/miniconda3
curl -fsSL "$MINICONDA_URL" -o ~/miniconda3/miniconda.sh

# Install
echo "Installing Miniconda..."
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Cleanup
rm ~/miniconda3/miniconda.sh

# Initialize for the current shell
case "$SHELL" in
    */zsh)
        ~/miniconda3/bin/conda init zsh
        echo "Conda initialized for zsh. Please restart your terminal or run: source ~/.zshrc"
        ;;
    */bash)
        ~/miniconda3/bin/conda init bash
        echo "Conda initialized for bash. Please restart your terminal or run: source ~/.bashrc"
        ;;
    *)
        ~/miniconda3/bin/conda init
        echo "Conda initialized. Please restart your terminal."
        ;;
esac

echo ""
echo "✅ Miniconda installation complete!"
echo "   Location: ~/miniconda3"
echo ""
echo "Next steps:"
echo "  1. Restart your terminal or source your shell config"
echo "  2. Run: pip install -r requirements.txt"
echo "  3. Run: jupyter notebook"