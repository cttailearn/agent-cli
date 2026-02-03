---
name: packmol
description: "Packmol is a molecular packing tool for creating initial configurations for molecular dynamics simulations. Use when users need to: (1) Create solvated systems (proteins in water, lipids in membranes), (2) Pack molecules in defined regions of space, (3) Generate initial coordinates for MD simulations, (4) Build complex molecular assemblies, (5) Create mixtures of molecules at specific concentrations, (6) Prepare systems with periodic boundary conditions"
---

# Packmol Molecular Packing Tool

Packmol creates initial configurations for molecular dynamics simulations by packing molecules in defined regions of space while minimizing overlaps.

## Quick Start

### Installation

```bash
# Download from GitHub
git clone https://github.com/m3g/packmol
cd packmol
make

# Or use conda
conda install -c conda-forge packmol
```

### Basic Usage

Create an input file `input.inp`:

```fortran
tolerance 2.0
output packed.pdb
filetype pdb

structure molecule.pdb
  number 100
  inside box 0. 0. 0. 40. 40. 40.
end structure
```

Run Packmol:
```bash
packmol < input.inp
```

## Core Concepts

### Input File Structure

Packmol input files use a simple domain-specific language:

```fortran
# Global parameters
tolerance 2.0          # Distance tolerance in Ångströms
output output.pdb      # Output file name
filetype pdb          # Output format (pdb, xyz, etc.)

# Molecule definitions
structure molecule1.pdb
  number 10           # Number of copies
  inside box 0. 0. 0. 50. 50. 50.  # Box dimensions
end structure

structure molecule2.pdb
  number 5
  inside sphere 0. 0. 0. 20.  # Inside sphere
end structure
```

### Common Constraints

#### Spatial Constraints
- `inside box xmin ymin zmin xmax ymax zmax`
- `inside sphere xc yc zc radius`
- `inside cylinder xc yc zc xd yd zd radius length`
- `outside box/sphere/cylinder` (exclusion regions)

#### Orientation Constraints
- `fixed x y z a b c` (position and Euler angles)
- `center` (center molecule at origin)
- `rotate` (random rotation)

#### Distance Constraints
- `overlap_checks 1000` (number of overlap checks)
- `maxit 20` (maximum iterations)

## Common Workflows

### 1. Solvating a Protein

```fortran
tolerance 2.0
output solvated.pdb
filetype pdb

structure protein.pdb
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure water.pdb
  number 1000
  inside box -15. -15. -15. 15. 15. 15.
  outside box -10. -10. -10. 10. 10. 10.
end structure
```

### 2. Creating a Lipid Bilayer

```fortran
tolerance 2.0
output bilayer.pdb
filetype pdb

structure lipid.pdb
  number 64
  inside box 0. 0. 0. 60. 60. 10.
end structure

structure water.pdb
  number 1000
  inside box 0. 0. -20. 60. 60. 20.
end structure
```

### 3. Mixture of Molecules

```fortran
tolerance 2.0
output mixture.pdb
filetype pdb

structure ethanol.pdb
  number 50
  inside box 0. 0. 0. 40. 40. 40.
end structure

structure water.pdb
  number 200
  inside box 0. 0. 0. 40. 40. 40.
end structure
```

## Advanced Features

### Seed Control
```fortran
seed -1  # Random seed (-1 = random, positive = fixed)
```

### Restart Files
```fortran
write_restart ./restart.dat
read_restart ./restart.dat
```

### Constraints File
Create constraints in separate file:
```fortran
constraints fix_region.const
```

### Multiple Output Formats
```fortran
filetype pdb  # Protein Data Bank
filetype xyz  # XYZ format
filetype lmp  # LAMMPS data file
```

## Troubleshooting

### Common Issues

1. **Packing fails**: Increase tolerance or decrease number of molecules
2. **Slow convergence**: Adjust `maxit` and `overlap_checks`
3. **Memory issues**: Use `discale` parameter for large systems

### Optimization Tips

```fortran
discale 1.2          # Discretization scale (1.0-1.5)
nloop 100            # Number of loops for optimization
maxit 20             # Maximum iterations per loop
```

## Integration with MD Packages

### GROMACS
```bash
# Convert PDB to GROMACS format
gmx pdb2gmx -f packed.pdb -o system.gro
```

### LAMMPS
```fortran
filetype lmp
output system.data
```

### NAMD
```bash
# Use psfgen to create PSF file after packing
```

## Script Examples

See `scripts/` directory for ready-to-use scripts:
- `solvate_protein.py`: Python script for protein solvation
- `create_mixture.sh`: Shell script for creating mixtures
- `packmol_wrapper.py`: Python wrapper for Packmol

## References

For detailed documentation and advanced usage, see:
- [Official Packmol Documentation](https://m3g.github.io/packmol/)
- [Packmol GitHub Repository](https://github.com/m3g/packmol)
- [Example Gallery](https://m3g.github.io/packmol/examples.shtml)