# Packmol Skill

A comprehensive skill for using Packmol to create initial configurations for molecular dynamics simulations.

## Skill Structure

```
packmol-skill/
├── SKILL.md                    # Main skill documentation
├── scripts/                    # Executable scripts
│   ├── solvate_protein.py     # Python script for protein solvation
│   ├── create_mixture.sh      # Shell script for creating mixtures
│   └── packmol_wrapper.py     # Python wrapper with OOP interface
├── references/                 # Reference documentation
│   ├── input_format.md        # Complete input file reference
│   └── examples.md            # Example gallery with common use cases
└── assets/                    # Example molecule files
    ├── water.pdb              # TIP3P water molecule
    ├── ethanol.pdb            # Ethanol molecule
    ├── na.pdb                 # Sodium ion
    └── cl.pdb                 # Chloride ion
```

## Quick Start

### Installation

1. Install Packmol:
   ```bash
   # Using conda
   conda install -c conda-forge packmol
   
   # Or compile from source
   git clone https://github.com/m3g/packmol
   cd packmol
   make
   ```

2. Test installation:
   ```bash
   packmol --version
   ```

### Basic Usage

Create a simple input file `test.inp`:
```fortran
tolerance 2.0
output test.pdb
filetype pdb

structure water.pdb
  number 100
  inside box 0. 0. 0. 20. 20. 20.
end structure
```

Run Packmol:
```bash
packmol < test.inp
```

## Script Usage

### Python Scripts

**Solvate a protein:**
```bash
python scripts/solvate_protein.py protein.pdb water.pdb -o solvated.pdb
```

**Using the wrapper:**
```python
from scripts.packmol_wrapper import PackmolSystem, Molecule

system = PackmolSystem(output_file="system.pdb")
water = Molecule("water.pdb", number=1000)
water.inside_box(0, 0, 0, 30, 30, 30)
system.add_molecule(water)
system.run()
```

### Shell Scripts

**Create a mixture:**
```bash
bash scripts/create_mixture.sh ethanol.pdb:50 water.pdb:200 -o mixture.pdb
```

## Common Workflows

### 1. Solvating Biomolecules
- Proteins in water
- Nucleic acids in ionic solutions
- Membranes with lipids

### 2. Creating Mixtures
- Binary/ternary mixtures
- Solutions at specific concentrations
- Phase-separated systems

### 3. Complex Geometries
- Spherical droplets
- Cylindrical pores
- Layered systems
- Interfaces

## Integration with MD Packages

Packmol output can be used directly with:
- **GROMACS**: Convert PDB to GRO format
- **LAMMPS**: Use `filetype lmp` for LAMMPS data files
- **AMBER**: Use `add_amber_ter` for AMBER-compatible PDBs
- **NAMD**: Combine with PSF generation

## Resources

- [Official Packmol Documentation](https://m3g.github.io/packmol/)
- [GitHub Repository](https://github.com/m3g/packmol)
- [Example Gallery](https://m3g.github.io/packmol/examples.shtml)
- [User Guide](https://m3g.github.io/packmol/userguide.shtml)

## Troubleshooting

### Common Issues

1. **Packing fails**: Increase tolerance or decrease molecule count
2. **Slow convergence**: Adjust `discale`, `maxit`, `nloop` parameters
3. **Memory issues**: Use restart files for large systems
4. **Format issues**: Ensure PDB files are correctly formatted

### Optimization Tips

- Start with `tolerance 2.5` for large systems
- Use `discale 1.2-1.4` for better performance
- Set `seed` for reproducible results
- Use restart files for systems > 10,000 molecules