# Packmol Examples Gallery

## Basic Examples

### 1. Simple Water Box
Create a cubic box of water molecules.

**Input: `water_box.inp`**
```fortran
tolerance 2.0
output water_box.pdb
filetype pdb

structure water.pdb
  number 1000
  inside box 0. 0. 0. 30. 30. 30.
end structure
```

**Run:**
```bash
packmol < water_box.inp
```

### 2. Ethanol-Water Mixture
Create a mixture of ethanol and water molecules.

**Input: `mixture.inp`**
```fortran
tolerance 2.0
output mixture.pdb
filetype pdb
seed 12345

structure ethanol.pdb
  number 50
  inside box 0. 0. 0. 40. 40. 40.
  rotate random
end structure

structure water.pdb
  number 200
  inside box 0. 0. 0. 40. 40. 40.
end structure
```

## Biomolecular Systems

### 3. Solvated Protein
Solvate a protein in a water box with exclusion region.

**Input: `solvate_protein.inp`**
```fortran
tolerance 2.0
output solvated.pdb
filetype pdb
add_amber_ter

structure protein.pdb
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure water.pdb
  number 3000
  inside box -15. -15. -15. 15. 15. 15.
  outside box -10. -10. -10. 10. 10. 10.
end structure
```

### 4. Protein in Water Sphere
Solvate protein in spherical water droplet.

**Input: `protein_sphere.inp`**
```fortran
tolerance 2.0
output protein_sphere.pdb
filetype pdb

structure protein.pdb
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure water.pdb
  number 1000
  inside sphere 0. 0. 0. 25.
  outside sphere 0. 0. 0. 12.
end structure
```

### 5. Lipid Bilayer with Water
Create a lipid bilayer solvated in water.

**Input: `bilayer.inp`**
```fortran
tolerance 2.0
output bilayer.pdb
filetype pdb

structure lipid.pdb
  number 64
  inside box 0. 0. 0. 60. 60. 10.
end structure

structure water.pdb
  number 2000
  inside box 0. 0. -20. 60. 60. 20.
end structure
```

## Complex Geometries

### 6. Concentric Spheres
Create system with molecules in concentric spherical shells.

**Input: `concentric.inp`**
```fortran
tolerance 2.0
output concentric.pdb
filetype pdb

structure inner.pdb
  number 100
  inside sphere 0. 0. 0. 15.
end structure

structure middle.pdb
  number 200
  inside sphere 0. 0. 0. 25.
  outside sphere 0. 0. 0. 15.
end structure

structure outer.pdb
  number 300
  inside sphere 0. 0. 0. 35.
  outside sphere 0. 0. 0. 25.
end structure
```

### 7. Cylindrical Pore
Create molecules inside a cylindrical pore.

**Input: `pore.inp`**
```fortran
tolerance 2.0
output pore.pdb
filetype pdb

structure water.pdb
  number 500
  inside cylinder 0. 0. 0. 0. 0. 1. 10. 40.
end structure

structure ions.pdb
  number 20
  inside cylinder 0. 0. 0. 0. 0. 1. 8. 40.
end structure
```

### 8. Layered System
Create alternating layers of different molecules.

**Input: `layers.inp`**
```fortran
tolerance 2.0
output layers.pdb
filetype pdb

structure layer1.pdb
  number 100
  inside box 0. 0. 0. 40. 40. 10.
end structure

structure layer2.pdb
  number 100
  inside box 0. 0. 10. 40. 40. 20.
end structure

structure layer3.pdb
  number 100
  inside box 0. 0. 20. 40. 40. 30.
end structure
```

## Advanced Examples

### 9. System with Ions
Create solvated system with ions at specific concentration.

**Input: `ions.inp`**
```fortran
tolerance 2.0
output system_ions.pdb
filetype pdb
add_amber_ter

structure protein.pdb
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure water.pdb
  number 3000
  inside box -15. -15. -15. 15. 15. 15.
  outside box -10. -10. -10. 10. 10. 10.
end structure

structure na.pdb
  number 15
  inside box -15. -15. -15. 15. 15. 15.
  outside box -10. -10. -10. 10. 10. 10.
end structure

structure cl.pdb
  number 15
  inside box -15. -15. -15. 15. 15. 15.
  outside box -10. -10. -10. 10. 10. 10.
end structure
```

### 10. Multiple Proteins
Create system with multiple protein copies.

**Input: `multiprotein.inp`**
```fortran
tolerance 2.0
output multiprotein.pdb
filetype pdb
discale 1.2
maxit 50

structure protein.pdb
  number 4
  inside box 0. 0. 0. 80. 80. 80.
end structure

structure water.pdb
  number 5000
  inside box 0. 0. 0. 80. 80. 80.
end structure
```

### 11. Restart Example
Use restart files for large systems.

**Input: `restart_write.inp`**
```fortran
tolerance 2.0
output large_system.pdb
filetype pdb
write_restart ./restart.dat
maxit 100
nloop 1000

structure molecule1.pdb
  number 5000
  inside box 0. 0. 0. 100. 100. 100.
end structure
```

**Continue from restart: `restart_read.inp`**
```fortran
tolerance 2.0
output large_system_complete.pdb
filetype pdb
read_restart ./restart.dat
maxit 50
nloop 500

structure molecule2.pdb
  number 2000
  inside box 0. 0. 0. 100. 100. 100.
end structure
```

## Specialized Examples

### 12. Interface System
Create liquid-vapor interface.

**Input: `interface.inp`**
```fortran
tolerance 2.0
output interface.pdb
filetype pdb

structure liquid.pdb
  number 1000
  inside box 0. 0. 0. 40. 40. 20.
end structure

structure vapor.pdb
  number 100
  inside box 0. 0. 20. 40. 40. 40.
end structure
```

### 13. Nanoparticle in Solvent
Create nanoparticle surrounded by solvent.

**Input: `nanoparticle.inp`**
```fortran
tolerance 2.0
output nanoparticle.pdb
filetype pdb

structure nanoparticle.pdb
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure solvent.pdb
  number 2000
  inside sphere 0. 0. 0. 30.
  outside sphere 0. 0. 0. 10.
end structure
```

### 14. Mixed Solvent System
Create system with multiple solvent types.

**Input: `mixed_solvent.inp`**
```fortran
tolerance 2.0
output mixed_solvent.pdb
filetype pdb

structure water.pdb
  number 800
  inside box 0. 0. 0. 40. 40. 40.
end structure

structure methanol.pdb
  number 200
  inside box 0. 0. 0. 40. 40. 40.
end structure

structure ethanol.pdb
  number 100
  inside box 0. 0. 0. 40. 40. 40.
end structure
```

## Troubleshooting Examples

### 15. High Density System
For systems that fail to pack, adjust parameters.

**Input: `high_density.inp`**
```fortran
tolerance 1.5      # Reduced tolerance
output high_density.pdb
filetype pdb
discale 1.3        # Increased discretization
maxit 100          # More iterations
overlap_checks 5000 # More overlap checks

structure molecule.pdb
  number 500
  inside box 0. 0. 0. 30. 30. 30.
end structure
```

### 16. Large System Optimization
Optimize parameters for very large systems.

**Input: `large_optimized.inp`**
```fortran
tolerance 2.5      # Larger tolerance for speed
output large_system.pdb
filetype pdb
discale 1.4        # Larger discretization
maxit 30           # Fewer iterations (but more loops)
nloop 2000         # More loops
seed 12345         # Fixed seed for reproducibility

structure water.pdb
  number 10000
  inside box 0. 0. 0. 100. 100. 100.
end structure
```

## Python Script Examples

### 17. Python Wrapper Usage
```python
from packmol_wrapper import PackmolSystem, Molecule

# Create system
system = PackmolSystem(output_file="system.pdb")

# Add molecules
water = Molecule("water.pdb", number=1000)
water.inside_box(0, 0, 0, 30, 30, 30)
system.add_molecule(water)

ethanol = Molecule("ethanol.pdb", number=50)
ethanol.inside_box(0, 0, 0, 30, 30, 30)
system.add_molecule(ethanol)

# Run Packmol
system.run()
```

### 18. Automated Solvation
```python
from packmol_wrapper import create_solvated_protein

# Create solvated protein
system = create_solvated_protein(
    protein_pdb="protein.pdb",
    water_pdb="water.pdb",
    output_pdb="solvated.pdb",
    box_size=30.0
)

# Run with custom options
system.set_seed(12345)
system.set_option("discale", 1.2)
system.run()
```