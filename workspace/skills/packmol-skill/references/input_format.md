# Packmol Input File Format Reference

## Basic Structure

Packmol input files use a simple domain-specific language with the following structure:

```
# Global parameters (required)
tolerance 2.0
output output.pdb
filetype pdb

# Molecule definitions (one or more)
structure molecule.pdb
  number 100
  inside box 0. 0. 0. 40. 40. 40.
end structure
```

## Global Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `tolerance` | Minimum distance between atoms (Å) | `tolerance 2.0` |
| `output` | Output file name | `output system.pdb` |
| `filetype` | Output format | `filetype pdb` |

### Optional Global Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `seed` | Random seed (-1 = random) | -1 | `seed 12345` |
| `discale` | Discretization scale | 1.0 | `discale 1.2` |
| `maxit` | Maximum iterations | 20 | `maxit 50` |
| `nloop` | Number of optimization loops | 100 | `nloop 1000` |
| `overlap_checks` | Number of overlap checks | 1000 | `overlap_checks 5000` |
| `write_restart` | Write restart file | - | `write_restart restart.dat` |
| `read_restart` | Read restart file | - | `read_restart restart.dat` |

## Molecule Definition

Each molecule is defined with a `structure...end structure` block:

```
structure <filename>
  number <N>
  <constraints>
end structure
```

### Required in Structure Block

| Parameter | Description | Example |
|-----------|-------------|---------|
| `number` | Number of copies | `number 100` |

### Constraints

#### Spatial Constraints

| Constraint | Format | Description |
|------------|--------|-------------|
| `inside box` | `inside box xmin ymin zmin xmax ymax zmax` | Inside rectangular box |
| `inside sphere` | `inside sphere xc yc zc radius` | Inside sphere |
| `inside cylinder` | `inside cylinder xc yc zc xd yd zd radius length` | Inside cylinder |
| `outside box` | `outside box xmin ymin zmin xmax ymax zmax` | Outside rectangular box |
| `outside sphere` | `outside sphere xc yc zc radius` | Outside sphere |
| `outside cylinder` | `outside cylinder xc yc zc xd yd zd radius length` | Outside cylinder |

#### Position/Orientation Constraints

| Constraint | Format | Description |
|------------|--------|-------------|
| `fixed` | `fixed x y z a b c` | Fixed position and Euler angles |
| `center` | `center` | Center molecule at origin |
| `rotate` | `rotate random` | Apply random rotation |
| `rotate` | `rotate x y z angle` | Rotate around axis |

#### Advanced Constraints

| Constraint | Format | Description |
|------------|--------|-------------|
| `atoms` | `atoms <list>` | Apply constraint to specific atoms |
| `residues` | `residues <list>` | Apply constraint to specific residues |
| `chains` | `chains <list>` | Apply constraint to specific chains |

## Output Formats

### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| PDB | `.pdb` | Protein Data Bank format |
| XYZ | `.xyz` | Simple XYZ coordinates |
| LAMMPS | `.lmp` or `.data` | LAMMPS data file |
| MOLDY | `.config` | MOLDY format |
| TINKER | `.xyz` | TINKER format |
| Gaussian | `.com` | Gaussian input |

### Format-Specific Options

#### PDB Format
```fortran
filetype pdb
add_amber_ter      # Add TER cards for AMBER
add_box_sides      # Add CRYST1 record with box dimensions
```

#### LAMMPS Format
```fortran
filetype lmp
# Additional LAMMPS-specific options can be added
```

## Examples

### Simple Box of Water
```fortran
tolerance 2.0
output water_box.pdb
filetype pdb

structure water.pdb
  number 1000
  inside box 0. 0. 0. 30. 30. 30.
end structure
```

### Protein in Water Sphere
```fortran
tolerance 2.0
output protein_solvated.pdb
filetype pdb

structure protein.pdb
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure water.pdb
  number 500
  inside sphere 0. 0. 0. 25.
  outside sphere 0. 0. 0. 10.
end structure
```

### Layered System
```fortran
tolerance 2.0
output layered.pdb
filetype pdb

structure layer1.pdb
  number 50
  inside box 0. 0. 0. 40. 40. 10.
end structure

structure layer2.pdb
  number 50
  inside box 0. 0. 10. 40. 40. 20.
end structure
```

### Mixture with Constraints
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
  outside box 0. 0. 0. 10. 10. 10.
end structure
```

## Advanced Features

### Restart Files
```fortran
# Write restart file for continuation
write_restart ./restart.dat

# Read restart file to continue
read_restart ./restart.dat
```

### Constraints File
```fortran
# Load constraints from external file
constraints my_constraints.const
```

### Multiple Constraints per Molecule
```fortran
structure molecule.pdb
  number 100
  inside box 0. 0. 0. 40. 40. 40.
  outside sphere 20. 20. 20. 10.
  rotate random
end structure
```

## Tips and Best Practices

1. **Start with larger tolerance** (2.0-3.0 Å) and decrease if needed
2. **Use `discale 1.2`** for better performance with large systems
3. **Increase `maxit` and `overlap_checks`** if packing fails
4. **Use `seed` for reproducible results**
5. **Check molecule PDB files** for correct formatting
6. **Use `add_amber_ter`** for AMBER-compatible PDB files
7. **Consider using restart files** for large systems