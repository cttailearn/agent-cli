#!/usr/bin/env python3
"""
Python wrapper for Packmol with object-oriented interface.
"""

import subprocess
import tempfile
import os
from typing import List, Dict, Optional, Tuple
import numpy as np

class Molecule:
    """Represents a molecule in Packmol."""
    
    def __init__(self, filename: str, number: int = 1):
        self.filename = filename
        self.number = number
        self.constraints = []
        
    def inside_box(self, xmin: float, ymin: float, zmin: float, 
                   xmax: float, ymax: float, zmax: float):
        """Constrain molecule to be inside a box."""
        self.constraints.append(f"inside box {xmin} {ymin} {zmin} {xmax} {ymax} {zmax}")
        return self
    
    def inside_sphere(self, xc: float, yc: float, zc: float, radius: float):
        """Constrain molecule to be inside a sphere."""
        self.constraints.append(f"inside sphere {xc} {yc} {zc} {radius}")
        return self
    
    def outside_box(self, xmin: float, ymin: float, zmin: float,
                    xmax: float, ymax: float, zmax: float):
        """Constrain molecule to be outside a box."""
        self.constraints.append(f"outside box {xmin} {ymin} {zmin} {xmax} {ymax} {zmax}")
        return self
    
    def fixed(self, x: float, y: float, z: float, 
              a: float = 0.0, b: float = 0.0, c: float = 0.0):
        """Fix molecule at specific position and orientation."""
        self.constraints.append(f"fixed {x} {y} {z} {a} {b} {c}")
        return self
    
    def center(self):
        """Center molecule at origin."""
        self.constraints.append("center")
        return self
    
    def rotate(self, axis: str = "random"):
        """Apply rotation to molecule."""
        if axis == "random":
            self.constraints.append("rotate random")
        else:
            self.constraints.append(f"rotate {axis}")
        return self
    
    def add_constraint(self, constraint: str):
        """Add custom constraint."""
        self.constraints.append(constraint)
        return self
    
    def to_packmol(self) -> str:
        """Convert to Packmol input format."""
        lines = [f"structure {self.filename}"]
        lines.append(f"  number {self.number}")
        for constraint in self.constraints:
            lines.append(f"  {constraint}")
        lines.append("end structure")
        return "\n".join(lines)

class PackmolSystem:
    """Represents a complete Packmol system."""
    
    def __init__(self, output_file: str = "output.pdb", 
                 filetype: str = "pdb", tolerance: float = 2.0):
        self.output_file = output_file
        self.filetype = filetype
        self.tolerance = tolerance
        self.seed = -1  # Random seed
        self.molecules = []
        self.global_options = {}
        
    def add_molecule(self, molecule: Molecule):
        """Add a molecule to the system."""
        self.molecules.append(molecule)
        return self
    
    def set_seed(self, seed: int):
        """Set random seed."""
        self.seed = seed
        return self
    
    def set_option(self, key: str, value):
        """Set global option."""
        self.global_options[key] = value
        return self
    
    def create_input(self) -> str:
        """Create Packmol input file content."""
        lines = []
        
        # Global parameters
        lines.append(f"tolerance {self.tolerance}")
        lines.append(f"output {self.output_file}")
        lines.append(f"filetype {self.filetype}")
        lines.append(f"seed {self.seed}")
        
        # Global options
        for key, value in self.global_options.items():
            lines.append(f"{key} {value}")
        
        lines.append("")  # Blank line
        
        # Molecule definitions
        for molecule in self.molecules:
            lines.append(molecule.to_packmol())
            lines.append("")  # Blank line between molecules
        
        return "\n".join(lines)
    
    def run(self, verbose: bool = True) -> bool:
        """Run Packmol with current configuration."""
        input_content = self.create_input()
        
        if verbose:
            print("Packmol input:")
            print(input_content)
            print("\n" + "="*50 + "\n")
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as f:
            f.write(input_content)
            input_file = f.name
        
        try:
            # Run Packmol
            result = subprocess.run(
                ['packmol'],
                input=input_content.encode(),
                capture_output=True,
                text=True
            )
            
            if verbose:
                print("Packmol output:")
                print(result.stdout)
                if result.stderr:
                    print("Errors:")
                    print(result.stderr)
            
            success = result.returncode == 0
            
            if success:
                print(f"✓ Packmol completed successfully")
                print(f"Output saved to: {self.output_file}")
                
                # Report statistics
                self._report_statistics()
            else:
                print(f"✗ Packmol failed with return code {result.returncode}")
            
            return success
            
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
    
    def _report_statistics(self):
        """Report statistics about the created system."""
        if not os.path.exists(self.output_file):
            return
        
        # Count atoms based on file type
        if self.filetype == "pdb":
            with open(self.output_file, 'r') as f:
                lines = f.readlines()
                atom_lines = [l for l in lines if l.startswith('ATOM') or l.startswith('HETATM')]
                n_atoms = len(atom_lines)
                print(f"Total atoms: {n_atoms}")
        
        total_molecules = sum(mol.number for mol in self.molecules)
        print(f"Total molecules: {total_molecules}")
        print(f"Number of species: {len(self.molecules)}")

# Example usage functions
def create_solvated_protein(protein_pdb: str, water_pdb: str, 
                           output_pdb: str = "solvated.pdb",
                           box_size: float = 30.0,
                           water_density: float = 0.033) -> PackmolSystem:
    """
    Create a solvated protein system.
    
    Args:
        protein_pdb: Protein PDB file
        water_pdb: Water molecule PDB file
        output_pdb: Output file name
        box_size: Cubic box size in Å
        water_density: Water density in molecules/Å³
    
    Returns:
        PackmolSystem object
    """
    system = PackmolSystem(output_file=output_pdb)
    
    # Add protein (fixed at center)
    protein = Molecule(protein_pdb, number=1)
    protein.fixed(0, 0, 0)
    system.add_molecule(protein)
    
    # Calculate number of water molecules
    volume = box_size ** 3
    n_water = int(volume * water_density)
    
    # Add water (in box but outside protein region)
    half_box = box_size / 2
    buffer = 5.0  # Å buffer around protein
    
    water = Molecule(water_pdb, number=n_water)
    water.inside_box(-half_box, -half_box, -half_box, half_box, half_box, half_box)
    water.outside_box(-buffer, -buffer, -buffer, buffer, buffer, buffer)
    system.add_molecule(water)
    
    return system

def create_lipid_bilayer(lipid_pdb: str, water_pdb: str,
                        output_pdb: str = "bilayer.pdb",
                        box_xy: float = 60.0,
                        box_z: float = 10.0,
                        n_lipids: int = 64,
                        water_layers: float = 20.0) -> PackmolSystem:
    """
    Create a lipid bilayer with water.
    
    Args:
        lipid_pdb: Lipid molecule PDB file
        water_pdb: Water molecule PDB file
        output_pdb: Output file name
        box_xy: Box dimensions in x and y directions
        box_z: Box dimension in z direction (bilayer thickness)
        n_lipids: Number of lipid molecules
        water_layers: Thickness of water layers above/below bilayer
    
    Returns:
        PackmolSystem object
    """
    system = PackmolSystem(output_file=output_pdb)
    
    # Add lipids in bilayer region
    lipid = Molecule(lipid_pdb, number=n_lipids)
    lipid.inside_box(0, 0, 0, box_xy, box_xy, box_z)
    system.add_molecule(lipid)
    
    # Add water above and below bilayer
    half_xy = box_xy / 2
    n_water = int(box_xy * box_xy * water_layers * 0.033)  # Approximate
    
    water = Molecule(water_pdb, number=n_water)
    water.inside_box(-half_xy, -half_xy, -water_layers, half_xy, half_xy, water_layers)
    system.add_molecule(water)
    
    return system

if __name__ == "__main__":
    # Example: Create command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description='Packmol Python Wrapper')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Solvate protein command
    solvate_parser = subparsers.add_parser('solvate', help='Solvate a protein')
    solvate_parser.add_argument('protein', help='Protein PDB file')
    solvate_parser.add_argument('water', help='Water PDB file')
    solvate_parser.add_argument('-o', '--output', default='solvated.pdb')
    solvate_parser.add_argument('-b', '--box', type=float, default=30.0)
    
    # Create mixture command
    mixture_parser = subparsers.add_parser('mixture', help='Create mixture')
    mixture_parser.add_argument('molecules', nargs='+', 
                               help='Molecule files with counts (file:count)')
    mixture_parser.add_argument('-o', '--output', default='mixture.pdb')
    mixture_parser.add_argument('-b', '--box', type=float, default=40.0)
    
    args = parser.parse_args()
    
    if args.command == 'solvate':
        system = create_solvated_protein(args.protein, args.water, args.output, args.box)
        system.run()
    
    elif args.command == 'mixture':
        system = PackmolSystem(output_file=args.output)
        half_box = args.box / 2
        
        for molspec in args.molecules:
            if ':' in molspec:
                molfile, count = molspec.split(':')
                count = int(count)
            else:
                molfile = molspec
                count = 100  # Default
            
            molecule = Molecule(molfile, number=count)
            molecule.inside_box(-half_box, -half_box, -half_box, 
                               half_box, half_box, half_box)
            system.add_molecule(molecule)
        
        system.run()
    
    else:
        parser.print_help()