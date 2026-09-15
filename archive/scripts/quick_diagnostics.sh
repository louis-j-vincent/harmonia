#!/bin/bash
# Quick diagnostic runner for Autumn Leaves dashboard
# Usage: bash scripts/quick_diagnostics.sh [root-mismatch|semi-markov|chroma|all]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"
}

# Print status message
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

# Print error message
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Main menu if no argument
if [ -z "$1" ] || [ "$1" == "menu" ]; then
    print_header "Autumn Leaves Diagnostic Dashboard — Quick Runner"
    echo "Select diagnostic to run:"
    echo "  1) Root Label Mismatches"
    echo "  2) Semi-Markov Configuration"
    echo "  3) Chroma Geometry Audit"
    echo "  4) All Diagnostics"
    echo "  5) Open Dashboard (browser)"
    echo "  6) Exit"
    echo ""
    read -p "Enter selection (1-6): " choice

    case $choice in
        1) set -- "root-mismatch" ;;
        2) set -- "semi-markov" ;;
        3) set -- "chroma" ;;
        4) set -- "all" ;;
        5) open "$PROJECT_ROOT/docs/plots/autumn_leaves_complete_diagnostics.html" 2>/dev/null || echo "Please open manually: docs/plots/autumn_leaves_complete_diagnostics.html"; exit 0 ;;
        6) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid selection"; exit 1 ;;
    esac
fi

diagnostic_type="$1"

case "$diagnostic_type" in
    root-mismatch)
        print_header "Generating Root Label Mismatch Analysis"
        python scripts/generate_root_mismatch_diagnostics.py
        ;;
    semi-markov)
        print_header "Generating Semi-Markov Configuration Diagnostics"
        python scripts/generate_semi_markov_diagnostics.py
        ;;
    chroma)
        print_header "Generating Chroma Geometry Audit"
        python scripts/generate_chroma_geometry_audit.py
        ;;
    all)
        print_header "Running All Diagnostics"
        python scripts/generate_all_diagnostics.py
        ;;
    *)
        print_error "Unknown diagnostic type: $diagnostic_type"
        echo "Usage: bash scripts/quick_diagnostics.sh [root-mismatch|semi-markov|chroma|all|menu]"
        exit 1
        ;;
esac

# Check if dashboard exists
if [ -f "$PROJECT_ROOT/docs/plots/autumn_leaves_complete_diagnostics.html" ]; then
    print_header "Diagnostics Complete!"
    print_status "Dashboard: docs/plots/autumn_leaves_complete_diagnostics.html"
    echo ""
    read -p "Open dashboard in browser? (y/n): " open_browser
    if [ "$open_browser" == "y" ] || [ "$open_browser" == "Y" ]; then
        open "$PROJECT_ROOT/docs/plots/autumn_leaves_complete_diagnostics.html" 2>/dev/null || echo "Please open manually: docs/plots/autumn_leaves_complete_diagnostics.html"
    fi
else
    print_error "Dashboard not found"
    exit 1
fi
