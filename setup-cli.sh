#!/bin/bash
# One-time setup for img-enhance CLI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=================================================="
echo "  img-enhance CLI Setup"
echo "=================================================="
echo ""

# Check if already in PATH
if echo "$PATH" | grep -q "$SCRIPT_DIR"; then
    echo "✓ img-enhance is already in your PATH"
    echo ""
    echo "You can use it from anywhere:"
    echo "  img-enhance --help"
    echo "  img-enhance"
    echo ""
else
    echo "Adding img-enhance to your PATH..."
    echo ""
    
    # Add to .bashrc
    echo "export PATH=\"\$PATH:$SCRIPT_DIR\"" >> ~/.bashrc
    
    echo "✓ Added to ~/.bashrc"
    echo ""
    echo "=================================================="
    echo "  Setup Complete!"
    echo "=================================================="
    echo ""
    echo "To start using img-enhance, run ONE of these:"
    echo ""
    echo "  Option 1 (recommended):"
    echo "    source ~/.bashrc"
    echo ""
    echo "  Option 2:"
    echo "    Open a new terminal window"
    echo ""
    echo "Then you can use img-enhance from anywhere:"
    echo "  img-enhance --help"
    echo "  img-enhance -i photo.jpg -m ultrasharp --fp16"
    echo ""
fi
