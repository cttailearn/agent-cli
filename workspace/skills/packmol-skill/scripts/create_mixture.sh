#!/bin/bash
# Create mixture of molecules using Packmol

set -e

# Default values
OUTPUT="mixture.pdb"
BOXSIZE=40.0
TOLERANCE=2.0
SEED=-1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [OPTIONS] MOL1:NUM1 MOL2:NUM2 ..."
    echo ""
    echo "Create a mixture of molecules using Packmol"
    echo ""
    echo "Arguments:"
    echo "  MOL:NUM    Molecule PDB file and number of copies (e.g., ethanol.pdb:50)"
    echo ""
    echo "Options:"
    echo "  -o FILE    Output file (default: $OUTPUT)"
    echo "  -b SIZE    Box size in Å (default: $BOXSIZE)"
    echo "  -t TOL     Tolerance in Å (default: $TOLERANCE)"
    echo "  -s SEED    Random seed (-1 for random) (default: $SEED)"
    echo "  -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 ethanol.pdb:50 water.pdb:200"
    echo "  $0 -o system.pdb -b 50.0 methane.pdb:100 ethane.pdb:50"
}

# Parse command line arguments
while getopts "o:b:t:s:h" opt; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        b) BOXSIZE="$OPTARG" ;;
        t) TOLERANCE="$OPTARG" ;;
        s) SEED="$OPTARG" ;;
        h) print_usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

shift $((OPTIND-1))

# Check if molecules are provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No molecules specified${NC}"
    print_usage
    exit 1
fi

# Create temporary input file
TMP_INPUT=$(mktemp packmol_input_XXXXXX.inp)

echo -e "${YELLOW}Creating Packmol input file: $TMP_INPUT${NC}"
echo -e "${YELLOW}Output: $OUTPUT${NC}"
echo -e "${YELLOW}Box size: ${BOXSIZE}Å³${NC}"
echo -e "${YELLOW}Tolerance: ${TOLERANCE}Å${NC}"

# Write Packmol header
cat > "$TMP_INPUT" << EOF
tolerance $TOLERANCE
output $OUTPUT
filetype pdb
seed $SEED

EOF

# Calculate half box size
HALF_BOX=$(echo "$BOXSIZE / 2" | bc -l)

# Add each molecule definition
MOL_COUNT=0
TOTAL_MOLECULES=0

for molspec in "$@"; do
    # Split molecule file and number
    IFS=':' read -r MOLFILE NUM <<< "$molspec"
    
    if [ ! -f "$MOLFILE" ]; then
        echo -e "${RED}Error: Molecule file not found: $MOLFILE${NC}"
        exit 1
    fi
    
    if [ -z "$NUM" ] || ! [[ "$NUM" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Invalid number for $MOLFILE: $NUM${NC}"
        exit 1
    fi
    
    MOL_COUNT=$((MOL_COUNT + 1))
    TOTAL_MOLECULES=$((TOTAL_MOLECULES + NUM))
    
    echo -e "${GREEN}  Adding $NUM copies of $MOLFILE${NC}"
    
    # Add to Packmol input
    cat >> "$TMP_INPUT" << EOF
structure $MOLFILE
  number $NUM
  inside box -$HALF_BOX -$HALF_BOX -$HALF_BOX $HALF_BOX $HALF_BOX $HALF_BOX
end structure

EOF
done

echo -e "${YELLOW}Total molecules: $TOTAL_MOLECULES${NC}"
echo -e "${YELLOW}Total components: $MOL_COUNT${NC}"

# Run Packmol
echo -e "\n${YELLOW}Running Packmol...${NC}"
if command -v packmol >/dev/null 2>&1; then
    packmol < "$TMP_INPUT"
    PACKMOL_EXIT=$?
else
    echo -e "${RED}Error: Packmol not found in PATH${NC}"
    echo "Please install Packmol or add it to your PATH"
    rm "$TMP_INPUT"
    exit 1
fi

# Check result
if [ $PACKMOL_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Packmol completed successfully${NC}"
    
    # Count atoms in output
    if [ -f "$OUTPUT" ]; then
        ATOM_COUNT=$(grep -c '^ATOM\|^HETATM' "$OUTPUT" 2>/dev/null || echo "0")
        echo -e "${GREEN}Output file created: $OUTPUT${NC}"
        echo -e "${GREEN}Total atoms: $ATOM_COUNT${NC}"
        
        # Calculate approximate density
        VOLUME=$(echo "$BOXSIZE^3" | bc -l)
        DENSITY=$(echo "scale=3; $TOTAL_MOLECULES / $VOLUME" | bc -l)
        echo -e "${GREEN}Approximate density: ${DENSITY} molecules/Å³${NC}"
    fi
else
    echo -e "${RED}✗ Packmol failed with exit code $PACKMOL_EXIT${NC}"
    echo "Check the input file: $TMP_INPUT"
    exit $PACKMOL_EXIT
fi

# Clean up
rm "$TMP_INPUT"
echo -e "${GREEN}Done!${NC}"