# Q-MECH
### Quantum Mechanical (DFT & QM/MM) Characterization of Protein–Ligand Complexes

**Q-MECH** is an open-source, GPU-accelerated Python pipeline that takes a single protein–ligand complex (PDB) and computes the electronic structure of the bound ligand and its interaction with the binding pocket: frontier orbitals, conceptual-DFT descriptors, Fukui functions, molecular electrostatic potential, QM/MM interaction energies with per-residue decomposition, geometric hydrogen bonds and drug-likeness.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PySCF](https://img.shields.io/badge/PySCF-2.14.0-green.svg)](https://pyscf.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.22908684)

---

## Overview

| Step | Analysis | Method |
|------|----------|--------|
| 1 | Ligand selection | Auto-detected (largest non-protein residue) or `RESNAME[:CHAIN:RESID]` |
| 2 | Ligand chemistry | RDKit bond perception; net charge from user / PDB / 3D structure, always cross-checked; missing-H detection |
| 3 | Protein environment | AMBER ff14SB charges + LJ (incl. H) via OpenMM; whole residues within 8 Å; chain breaks, ACE/NME caps, metal ions |
| 4 | DFT | B3LYP/def2-SVP, IEF-PCM water (default) or gas phase; PySCF / GPU4PySCF |
| 5 | Global descriptors | HOMO, LUMO, gap, I, A, χ, η, S, ω, ΔN<sub>max</sub> (Koopmans) + ΔSCF vertical IP/EA |
| 6 | Fukui functions | f⁺, f⁻, f⁰, Δf from N / N+1 / N−1 DFT, Hirshfeld-condensed |
| 7 | Molecular electrostatic potential | Analytical integrals on the 1.0–1.4 × vdW surface shell |
| 8 | QM/MM interaction energy | `pyscf.qmmm` electrostatic embedding: E(embedded) − E(gas) |
| 9 | Energy decomposition | Frozen-density electrostatics (per residue) + polarization + Lennard-Jones |
| 10 | Hydrogen bonds | D–H···A geometry with explicit H; Jeffrey classification |
| 11 | Interpretation | Covalent bond / warhead (SMARTS) detection, metal coordination, Domingo ω scale, Lipinski, Veber, QED |
| 12 | Output | 2 figures (8 panels, 600 dpi) + TXT / JSON / CSV reports per complex; multi-compound figures and draft results text |

---

## Features

- ✅ **Single input** — one PDB file with protein + ligand; no SMILES, no topology files
- ✅ **Works with any preparation tool** — Maestro, Chimera, PyMOL, OpenBabel, MOE, CHARMM-GUI, docking outputs, …
- ✅ **Input checks** — ligand charge and hydrogens are cross-validated against the 3D structure; inconsistent input stops the run with a clear message (no silent radicals)
- ✅ **Implicit solvent** — IEF-PCM makes descriptors of neutral and charged ligands comparable
- ✅ **Full ff14SB protein charges** — from OpenMM, including hydrogens; automatic protonation if needed
- ✅ **Convergence tracking** — every SCF is checked; no silent basis-set or spin changes
- ✅ **GPU optional** — GPU4PySCF used automatically when available; pip-installed CUDA libraries are found automatically; identical results on CPU
- ✅ **Automatic memory setting** — uses 85 % of free RAM (minimum 32 GB, adjustable)
- ✅ **Batch mode** — every PDB in a folder, per-complex settings via CSV, resume with `-k`
- ✅ **Multi-compound figures** — 8 combined figures, a comparison figure and a draft results section

---

## Installation

### Requirements
- Linux (tested), Python 3.10+
- ≥ 32 GB RAM (see `MIN_MEMORY_GB`)
- NVIDIA GPU with CUDA 11/12 — optional, for speed only

### Install dependencies

```bash
# Create a conda environment (recommended)
conda create -n qmech python=3.10 -y
conda activate qmech

# Required
pip install pyscf rdkit biopython openmm pdbfixer numpy scipy pandas matplotlib

# Optional: GPU acceleration (choose your CUDA version)
pip install gpu4pyscf-cuda12x cupy-cuda12x      # CUDA 12.x
# pip install gpu4pyscf-cuda11x cupy-cuda11x    # CUDA 11.x
```

### Verify installation

```bash
python3 - << 'EOF'
import pyscf, openmm, rdkit, Bio
print("PySCF   :", pyscf.__version__)
print("OpenMM  :", openmm.__version__)
print("RDKit   :", rdkit.__version__)
print("Biopython:", Bio.__version__)
try:
    import pdbfixer; print("PDBFixer: OK")
except ImportError:
    print("PDBFixer: not found (recommended)")
EOF
```

GPU status is printed on the first line of every Q-MECH run, including the reason if the GPU cannot be used:
```
✓ GPU4PySCF loaded - GPU mode (1 GPU: NVIDIA GeForce RTX 3060 Ti, 8 GB)
```

---

## Input requirements

1. **One PDB file** containing the protein and the ligand. Convert other formats first, e.g. `obabel complex.pdbqt -O complex.pdb`.
2. **The ligand must contain all hydrogens.** Docking outputs such as AutoDock Vina/PDBQT keep only polar hydrogens — add the rest first:
   `obabel lig.pdb -O lig_H.pdb -p 7.4` · UCSF Chimera `AddH` · PyMOL `h_add` · Maestro/MOE.
3. **Ligand net charge** (in this order): 4th command-line argument → PDB formal-charge columns 79–80 → perceived from the 3D structure. Inconsistent charges stop the run and report the consistent value.
4. Protein hydrogens are optional (added by OpenMM at pH 7 if absent or not recognized).

---

## Usage

### Single complex

```bash
python3 qmech_v1.0.py complex.pdb [LIGAND] [OUTDIR] [CHARGE]
```

| Argument | Meaning | Default |
|----------|---------|---------|
| `complex.pdb` | protein + ligand PDB | — |
| `LIGAND` | `AUTO`, `RESNAME` or `RESNAME:CHAIN:RESID` (e.g. `UNK:B:900`) | `AUTO` |
| `OUTDIR` | output folder | `qmech_results` |
| `CHARGE` | ligand net charge (`0`, `-1`, `+1`, …) or `AUTO` | `AUTO` |

Force CPU on a GPU machine: `CUDA_VISIBLE_DEVICES="" python3 qmech_v1.0.py complex.pdb`

### Settings (bottom of `qmech_v1.0.py`)

```python
FUNCTIONAL     = "B3LYP"     # B3LYP | PBE | M06-2X | wB97X-D
BASIS          = "def2-SVP"  # def2-SVPD recommended for Fukui f+ (anion)
SOLVENT        = "water"     # IEF-PCM for DFT/Fukui/MEP; None = gas phase
POCKET_CUTOFF  = 8.0         # Å, whole residues in the MM region
N_THREADS      = 24          # CPU threads
MEP_GRID       = 40          # MEP grid points per dimension
MAX_MEMORY     = "AUTO"      # 85 % of free RAM, or a number in MB
MIN_MEMORY_GB  = 32          # minimum RAM required
PH             = 7.0         # used only if the protein must be re-protonated
INCLUDE_WATERS = False       # protonated crystal waters as TIP3P charges
DPI            = 600
```

### Batch mode (no editing needed)

```bash
bash batch_run_qmech.sh                         # every *.pdb in the current folder
bash batch_run_qmech.sh -d /path/to/pdbs        # PDBs in another folder
bash batch_run_qmech.sh -f complexes.csv        # per-complex ligand / charge
bash batch_run_qmech.sh -F -t GyrB              # + combined figures at the end
nohup bash batch_run_qmech.sh > batch.log 2>&1 &   # long runs in the background
```

| Option | Meaning |
|--------|---------|
| `-d DIR` | folder with the PDB files |
| `-f FILE` | CSV `pdb,ligand,charge` (empty = AUTO, `#` = comment) |
| `-l LIG`, `-c CHARGE` | same ligand / charge for every complex |
| `-g N` / `-g cpu` | GPU index, or force CPU |
| `-k` | skip complexes that already have results (resume) |
| `-F`, `-t TARGET` | run `combined_figures.py` afterwards, protein name for the text |
| `-s PATH` | location of `qmech_v1.0.py` |

Example `complexes.csv`:
```
pdb,ligand,charge
complex_A.pdb,,+1
reference.pdb,80S:A:301,
```

Keep `batch_run_qmech.sh`, `qmech_v1.0.py` and `combined_figures.py` in the same folder.

### Multi-compound figures and results text

Run in the folder that contains the `*_results` directories:

```bash
python3 combined_figures.py                                  # all results found
python3 combined_figures.py --names cmpdA cmpdB ref \
        --labels A B "80S (reference)" --target GyrB --dpi 600
```

Options: `--names` (order), `--labels`, `--target`, `--dir`, `--out`, `--dpi`.

---

## Output

```
<name>_results/
├── fig1_ligand_quantum.png   ← charges, MEP, Fukui, H-bonds (4 panels)
├── fig2_binding_qmmm.png     ← per-residue energies, decomposition, descriptors, HOMO/LUMO
├── report.txt                ← human-readable report incl. WARNINGS section
├── report.json               ← machine-readable results
└── per_residue.csv           ← per-residue E_elec / E_LJ / E_total
<name>.log                    ← full run log (batch mode)
qmech_batch_summary.csv       ← one row per complex (batch mode)
combined_figures/
├── combined_fig1_homo_lumo.png … combined_fig8_mechanism.png
├── comparison_all_complexes.png
└── results_section.txt       ← draft manuscript text (check before use)
```

Always read the **WARNINGS** section of `report.txt` and the **perceived SMILES** in the log.

---

## Computed Parameters

### Global descriptors (Parr 1999 convention)

| Parameter | Formula | Meaning |
|-----------|---------|---------|
| HOMO, LUMO (eV) | Kohn–Sham orbital energies | electron donation / acceptance |
| Gap (eV) | E<sub>LUMO</sub> − E<sub>HOMO</sub> | kinetic stability (smaller = more reactive) |
| I, A (eV) | −E<sub>HOMO</sub>, −E<sub>LUMO</sub> | Koopmans ionization potential, electron affinity |
| χ (eV) | (I + A) / 2 | electronegativity (= −μ) |
| η (eV) | I − A | chemical hardness |
| S (eV⁻¹) | 1 / η | global softness |
| ω (eV) | χ² / 2η | electrophilicity index |
| ΔN<sub>max</sub> | χ / η | maximum electron uptake |
| Vertical IP / EA (eV) | ΔSCF: E(N−1) − E(N), E(N) − E(N+1) | preferred over Koopmans values for ions |

### Fukui functions (Hirshfeld-condensed)

| Index | Meaning |
|-------|---------|
| f⁺ = q(N+1) − q(N) | site most susceptible to **nucleophilic** attack (electrophilic site) |
| f⁻ = q(N) − q(N−1) | site most susceptible to **electrophilic** attack (nucleophilic site) |
| f⁰ | radical attack |
| Δf = f⁺ − f⁻ | dual descriptor (> 0 electrophilic, < 0 nucleophilic) |

### QM/MM energies (kcal/mol)

| Term | Definition |
|------|------------|
| E_elec | frozen gas-phase ligand density × ff14SB charges (per residue) |
| E_pol | polarization of the ligand by the protein field |
| E_int | E(embedded) − E(gas) = E_elec + E_pol |
| E_LJ | Lennard-Jones (ff14SB protein, GAFF-like ligand) |
| E_total | E_int + E_LJ — gas-phase interaction energy, **not ΔG<sub>bind</sub>** |

### Hydrogen bonds
D···A ≤ 3.5 Å, H···A ≤ 2.7 Å, ∠D–H···A ≥ 120°; strong < 2.5 Å, moderate 2.5–3.2 Å, weak > 3.2 Å (Jeffrey 1997).

---

## Interpreting results — limitations

- **QM/MM energies are not binding free energies** (gas phase, single structure, no desolvation/entropy). Binding pockets are often charged, so E_total is dominated by the ligand net charge: **compare only ligands of equal charge**, and rank affinities with a solvent-screened method (e.g. MM-GBSA).
- **Charged ligands:** use `SOLVENT = "water"` (default); gas-phase orbital energies of ions are shifted by several eV. The Domingo ω scale was derived for neutral molecules.
- **Single pose:** all results refer to the structure you provide.
- Ligand LJ parameters are element-based approximations to GAFF.
- If input protein hydrogens are not recognized, OpenMM re-protonates the protein at pH 7 (reported in WARNINGS).
- **Not supported:** open-shell or metal-containing ligands, covalently bound ligands (no link atoms), more than one ligand per run, cofactors/nucleic acids in the MM region (not included), non-standard residues. Use `LIGAND` explicitly if a cofactor is larger than the ligand.

---

## Troubleshooting

| Message | Cause / fix |
|---------|-------------|
| `Ligand is missing hydrogens at … carbon(s)` | add all hydrogens (see Input requirements) |
| `Charge X is not chemically consistent …` | wrong protonation state or charge; use the suggested value |
| `Odd electron count` | hydrogens or net charge wrong |
| `GPU4PySCF not available … libcublas.so.12` | CUDA 12 libraries missing: `pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12` |
| `Q-MECH needs at least 32 GB RAM` | lower `MIN_MEMORY_GB` for small ligands |
| `Chain break … gap ends become charged termini` | missing loop; if also listed as *inside the MM region*, model the loop or report it |
| `SCF STILL NOT CONVERGED` | results for that state are unreliable (batch exit code 2) |

---

## Performance

Measured on an NVIDIA GeForce RTX 3060 Ti (8 GB), 24 CPU threads, 128 GB RAM, for a 45-atom ligand (414 basis functions), gas phase: neutral SCF ≈ 2 min (GPU); complete pipeline ≈ 15 min. With PCM (default) the SCF steps take longer. <!-- add measured PCM timings -->
QM/MM, Hirshfeld and MEP steps run on CPU in both modes; CPU-only runs give identical results.

---

## Citation

If you use Q-MECH in your research, please cite:

```bibtex
@software{dwivedi_qmech_2026,
  author    = {Dwivedi, Vivek Dhar},
  title     = {{Q-MECH: A GPU-Accelerated Open-Source Python Pipeline
               for Quantum Mechanical (DFT and QM/MM) Characterization
               of Protein--Ligand Complexes}},
  year      = {2026},
  version   = {1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

### Please also cite the underlying methods and software

- **PySCF**: Sun et al., *WIREs Comput Mol Sci* 2018, 8, e1340; Sun et al., *J Chem Phys* 2020, 153, 024109
- **GPU4PySCF**: Wu et al., 2024 (see the GPU4PySCF documentation for the current reference)
- **OpenMM 8**: Eastman et al., *J Phys Chem B* 2024, 128, 109
- **AMBER ff14SB**: Maier et al., *J Chem Theory Comput* 2015, 11, 3696
- **B3LYP**: Becke, *J Chem Phys* 1993, 98, 5648; Lee, Yang & Parr, *Phys Rev B* 1988, 37, 785
- **def2-SVP**: Weigend & Ahlrichs, *Phys Chem Chem Phys* 2005, 7, 3297
- **IEF-PCM**: Cancès, Mennucci & Tomasi, *J Chem Phys* 1997, 107, 3032
- **Electrophilicity index**: Parr, Szentpály & Liu, *J Am Chem Soc* 1999, 121, 1922
- **Fukui functions**: Yang & Mortier, *J Am Chem Soc* 1986, 108, 5708
- **Hirshfeld partitioning**: Hirshfeld, *Theor Chim Acta* 1977, 44, 129
- **QM/MM**: Warshel & Levitt, *J Mol Biol* 1976, 103, 227
- **RDKit**: https://www.rdkit.org

---

## Dependencies

| Package | Tested version | Purpose |
|---------|---------------|---------|
| PySCF | 2.14.0 | DFT, PCM, QM/MM |
| GPU4PySCF | 1.8.1 (optional) | GPU acceleration |
| OpenMM | 8.6.1 | ff14SB charges, protonation |
| PDBFixer | 1.12.0 (recommended) | missing atoms, non-standard residues |
| RDKit | 2026.03 | bond perception, descriptors |
| Biopython | 1.88 | PDB parsing |
| NumPy, SciPy, Pandas, Matplotlib | current | numerics, data, figures |

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Author

**Dr. Vivek Dhar Dwivedi**
Raja Shankar Shah University, Chindwara, Madhya Pradesh, India
Computational Drug Design | Molecular Modeling | Bioinformatics

---

## Acknowledgements

The author acknowledges the developers of PySCF, GPU4PySCF, OpenMM, RDKit and Biopython, on which Q-MECH is built.
