#!/usr/bin/env python3
"""
Packmol wrapper for solvating proteins in water.
"""

import subprocess
import os
import argparse
import tempfile

def create_packmol_input(protein_pdb, water_pdb, output_pdb, 
                         box_size=30.0, water_density=0.033, 
                         tolerance=2.0, seed=-1):
    """
    Create Packmol input file for solvating a protein.
    
    Args:
        protein_pdb: Path to protein PDB file
        water_pdb: Path to water molecule PDB file
        output_pdb: Output PDB file name
        box_size: Cubic box size in Ångströms
        water_density: Water density in molecules/Å³ (default ~1 g/cm³)
        tolerance: Packmol tolerance in Å
        seed: Random seed (-1 for random)
    """
    
    # Calculate number of water molecules
    volume = box_size ** 3
    n_water = int(volume * water_density)
    
    # Create exclusion region (protein + buffer)
    buffer = 5.0  # Å buffer around protein
    half_box = box_size / 2
    
    input_content = f"""tolerance {tolerance}
output {output_pdb}
filetype pdb
seed {seed}

structure {protein_pdb}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure {water_pdb}
  number {n_water}
  inside box {-half_box} {-half_box} {-half_box} {half_box} {half_box} {half_box}
  outside box {-buffer} {-buffer} {-buffer} {buffer} {buffer} {buffer}
end structure
"""
    
    return input_content

def run_packmol(input_content, verbose=True):
    """
    Run Packmol with given input content.
    
    Args:
        input_content: Packmol input file content
        verbose: Print progress information
    
    Returns:
        True if successful, False otherwise
    """
    
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as f:
        f.write(input_content)
        input_file = f.name
    
    try:
        if verbose:
            print(f"Running Packmol with input file: {input_file}")
            print("Input content:")
            print(input_content)
        
        # Run Packmol
        result = subprocess.run(
            ['packmol'],
            input=input_content.encode(),
            capture_output=True,
            text=True
        )
        
        if verbose:
            print("\nPackmol output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Packmol completed successfully")
            return True
        else:
            print(f"✗ Packmol failed with return code {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("Error: Packmol not found. Make sure it's installed and in PATH.")
        return False
    except Exception as e:
        print(f"Error running Packmol: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(input_file):
            os.unlink(input_file)

def main():
    parser = argparse.ArgumentParser(description='Solvate protein using Packmol')
    parser.add_argument('protein', help='Protein PDB file')
    parser.add_argument('water', help='Water molecule PDB file')
    parser.add_argument('-o', '--output', default='solvated.pdb', 
                       help='Output PDB file (default: solvated.pdb)')
    parser.add_argument('-b', '--box', type=float, default=30.0,
                       help='Box size in Å (default: 30.0)')
    parser.add_argument('-d', '--density', type=float, default=0.033,
                       help='Water density in molecules/Å³ (default: 0.033)')
    parser.add_argument('-t', '--tolerance', type=float, default=2.0,
                       help='Packmol tolerance in Å (default: 2.0)')
    parser.add_argument('-s', '--seed', type=int, default=-1,
                       help='Random seed (-1 for random, default: -1)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Create Packmol input
    input_content = create_packmol_input(
        protein_pdb=args.protein,
        water_pdb=args.water,
        output_pdb=args.output,
        box_size=args.box,
        water_density=args.density,
        tolerance=args.tolerance,
        seed=args.seed
    )
    
    # Run Packmol
    success = run_packmol(input_content, verbose=not args.quiet)
    
    if success:
        print(f"\nSolvated system saved to {args.output}")
        # Calculate some statistics
        import sys
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                lines = f.readlines()
                n_atoms = sum(1 for line in lines if line.startswith('ATOM') or line.startswith('HETATM'))
                print(f"Total atoms: {n_atoms}")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()