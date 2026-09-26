"""
╔══════════════════════════════════════════════════════════════════╗
║   Q-MECH: A GPU-Accelerated Open-Source Python Pipeline for     ║
║   Quantum Mechanical (DFT & QM/MM) Analysis of Molecular        ║
║   Mechanism of Inhibition in Protein–Ligand Complexes           ║
╠══════════════════════════════════════════════════════════════════╣
║   Version  : 1.0                                                 ║
║   Engine   : PySCF 2.14.0 + GPU4PySCF, OpenMM, RDKit            ║
║   Method   : B3LYP/def2-SVP DFT + AMBER ff14SB QM/MM           ║
║   Input    : Protein-Ligand complex PDB file (single input)     ║
║   Output   : HOMO/LUMO, Fukui, MEP, QM/MM energies,            ║
║              H-bonds, Mechanism, 2 figures (8 panels)           ║
╠══════════════════════════════════════════════════════════════════╣
║   Citation:                                                      ║
║   Dwivedi VD (2026). Q-MECH v1.0. Zenodo.                      ║
║   https://doi.org/10.5281/zenodo.XXXXXXX                        ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python3 qmech_v1.0.py Complex_1.pdb LIG Complex_1_results
    python3 qmech_v1.0.py Complex_1.pdb LIG Complex_1_results -1   (4th arg = ligand net charge)
    python3 qmech_v1.0.py Complex_1.pdb LIG:B:401 Complex_1_results  (resname:chain:resid)
    python3 qmech_v1.0.py  (uses settings defined at bottom of script)

Input requirements (any preparation tool: Maestro, Chimera, PyMOL, OpenBabel,
MOE, Discovery Studio, CHARMM-GUI, AMBER/GROMACS, docking outputs, ...):
    * One PDB file with protein + ligand. PDBQT/MOL2/SDF: convert first, e.g.
      obabel complex.pdbqt -O complex.pdb
    * The ligand must carry ALL hydrogens (docking outputs such as AutoDock Vina
      keep only polar H — add the rest first, e.g. obabel lig.pdb -O lig_H.pdb -p 7.4).
    * Ligand net charge: 4th argument, else PDB formal-charge columns 79-80, else
      perceived from the 3D structure. The chosen charge is always cross-checked
      against the structure; inconsistent input stops the run.
    * Protein hydrogens are optional (added by OpenMM at pH 7 if absent or
      not recognised). Protein charges: AMBER ff14SB via OpenMM.

Method references:
    PySCF  : Sun Q et al. J Chem Phys 153, 024109 (2020)
    OpenMM : Eastman P et al. J Phys Chem B 128, 109 (2024)
    ff14SB : Maier JA et al. J Chem Theory Comput 11, 3696 (2015)
    GAFF   : Wang J et al. J Comput Chem 25, 1157 (2004)
    Hirshfeld charges : Hirshfeld FL. Theor Chim Acta 44, 129 (1977)
    Condensed Fukui   : Yang W, Mortier WJ. J Am Chem Soc 108, 5708 (1986)
    omega (ω)         : Parr RG, Szentpaly L, Liu S. J Am Chem Soc 121, 1922 (1999)
    ω scale           : Domingo LR et al. Tetrahedron 58, 4417 (2002)
    H-bond criteria   : Jeffrey GA. An Introduction to Hydrogen Bonding (OUP, 1997)
"""

import os, sys, json, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ── RDKit ──────────────────────────────────────────────────────────
from rdkit import Chem, RDLogger
from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors,
                        rdDetermineBonds, QED)
RDLogger.DisableLog("rdApp.*")

# ── PySCF — GPU or CPU ────────────────────────────────────────────
import pyscf
from pyscf import gto, scf, lib, df as pyscf_df, qmmm as pyscf_qmmm
from pyscf import dft as cpu_dft
from pyscf.data import elements as pyscf_elements
def _preload_pip_cuda_libs():
    """CUDA libraries installed with pip (nvidia-*-cu12, e.g. pulled in by PyTorch)
    live in site-packages/nvidia/*/lib, which the dynamic loader does not search.
    Pre-load them globally so CuPy/GPU4PySCF find them without LD_LIBRARY_PATH."""
    import ctypes, glob, site
    roots = []
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    try:
        roots += site.getsitepackages()
    except Exception:
        pass
    roots += [p for p in sys.path if p.endswith(("site-packages", "dist-packages"))]
    dirs = []
    for r in dict.fromkeys(roots):
        dirs += glob.glob(os.path.join(r, "nvidia", "*", "lib"))
    if not dirs:
        return 0
    order = ["libcudart.so.*", "libnvJitLink.so.*", "libnvrtc.so.*", "libcublasLt.so.*",
             "libcublas.so.*", "libcusparse.so.*", "libcufft.so.*", "libcurand.so.*",
             "libcusolver.so.*"]
    n = 0
    for pat in order:
        for d in dirs:
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                try:
                    ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL); n += 1
                except OSError:
                    pass
                break
    return n

GPU_OK, GPU_REASON = False, ""
try:
    _preload_pip_cuda_libs()
    import cupy
    n_gpu = cupy.cuda.runtime.getDeviceCount()
    if n_gpu == 0:
        raise RuntimeError("CuPy sees 0 GPUs (check CUDA_VISIBLE_DEVICES / driver)")
    from gpu4pyscf import dft
    GPU_OK = True
    _p = cupy.cuda.runtime.getDeviceProperties(0)
    _name = _p["name"].decode() if isinstance(_p["name"], bytes) else _p["name"]
    print(f"  ✓ GPU4PySCF loaded - GPU mode ({n_gpu} GPU: {_name}, "
          f"{_p['totalGlobalMem']/1024**3:.0f} GB)")
except Exception as _e:
    from pyscf import dft
    GPU_OK, GPU_REASON = False, f"{type(_e).__name__}: {_e}"
    print("  ⚠ GPU4PySCF not available - CPU mode")
    print(f"     reason: {GPU_REASON[:300]}")

# ── Biopython ──────────────────────────────────────────────────────
from Bio import PDB as BioPDB

# ── Constants (CODATA 2018) ────────────────────────────────────────
HARTREE_TO_EV   = 27.211386245988
HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANG     = 0.529177210903
ANG_TO_BOHR     = 1.0 / BOHR_TO_ANG
KJ_TO_KCAL      = 1.0 / 4.184

# ── Element lookups ────────────────────────────────────────────────
VDW_ANG = {'C':1.70,'N':1.55,'O':1.52,'S':1.80,'P':1.80,'H':1.20,'F':1.47,
           'Cl':1.75,'Br':1.85,'I':1.98,'B':1.92}
COV_R   = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'S':1.05,
           'Cl':1.02,'Br':1.20,'I':1.39,'B':0.84,'Si':1.11,'Se':1.20}

# ── Protein / solvent / ion residue sets ──────────────────────────
PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HIE","HID","HIP","HSD","HSE","HSP","CYX","CYM","ASH","GLH","LYN",
    "ACE","NME","NMA","MSE",
}
CAP_N = {"ACE"}                 # N-terminal caps (start a segment, neutral)
CAP_C = {"NME","NMA"}           # C-terminal caps (end a segment, neutral)
WATER_RESNAMES = {"HOH","WAT","TIP","TIP3","SOL","H2O","DOD","SPC"}
ION_CHARGES = {                 # formal charges used as MM point charges
    "ZN":2,"MG":2,"CA":2,"MN":2,"CO":2,"NI":2,"CU":2,"CD":2,"FE2":2,"FE":3,
    "NA":1,"K":1,"LI":1,"CS":1,"RB":1,"CL":-1,"BR":-1,"IOD":-1,
    "NA+":1,"K+":1,"CL-":-1,
}
METALS    = {"ZN","MG","CA","MN","CO","NI","CU","CD","FE","FE2"}
SOLVENT   = WATER_RESNAMES | set(ION_CHARGES)
ADDITIVES = {"SO4","PO4","GOL","EDO","ACT","PEG","PG4","PGE","DMS","FMT","MPD",
             "EPE","TRS","BME","IMD","NO3","SCN","CIT","MES","ACY","1PE","P6G"}

# ══════════════════════════════════════════════════════════════════
#  Force-field parameters
#  Protein: AMBER ff14SB charges + LJ taken from OpenMM (amber14-all.xml)
#  Ligand : GAFF-like LJ per element (sigma Å, eps kcal/mol); H typed by partner
# ══════════════════════════════════════════════════════════════════
LIG_LJ = {"C":(3.400,0.0860),"N":(3.250,0.1700),"O":(2.960,0.2100),
          "S":(3.564,0.2500),"P":(3.742,0.2000),"F":(3.118,0.0610),
          "Cl":(3.471,0.2650),"Br":(3.956,0.3200),"I":(4.187,0.4000),
          "H_C":(2.650,0.0157),"H_X":(1.069,0.0157)}

# Covalent warhead SMARTS (first atom = electrophilic centre)
WARHEADS = [
    ("Michael acceptor (enone/acrylamide)", "[CX3;!a]=[CX3;!a]-[CX3]=[OX1]"),
    ("Vinyl sulfone/sulfonamide",           "[CX3;!a]=[CX3;!a]-[SX4](=O)=O"),
    ("Acrylonitrile",                       "[CX3;!a]=[CX3;!a]-C#N"),
    ("Alkynamide/ynone",                    "[CX2]#[CX2]-[CX3]=[OX1]"),
    ("alpha-Haloacetamide/ketone",          "[CH2X4]([Cl,Br,I])-[CX3]=O"),
    ("Epoxide",                             "[CX4]1-[OX2]-[CX4]1"),
    ("Aziridine",                           "[CX4]1-[NX3]-[CX4]1"),
    ("Aldehyde",                            "[CX3H1](=O)[#6]"),
    ("Nitrile (reversible covalent)",       "[CX2]#[NX1]"),
    ("alpha-Ketoamide",                     "[CX3](=O)-[CX3](=O)-[NX3]"),
    ("Boronic acid/ester",                  "[BX3]([OX2])[OX2]"),
    ("Sulfonyl fluoride",                   "[SX4](=O)(=O)F"),
    ("beta-Lactam",                         "[CX3]1(=O)-[CX4]-[CX4]-[NX3]1"),
    ("Isothiocyanate",                      "[CX2](=S)=[NX2]"),
    ("Disulfide",                           "[SX2]-[SX2]"),
]
NUCLEOPHILES = {("CYS","SG"),("CYX","SG"),("SER","OG"),("THR","OG1"),("LYS","NZ"),
                ("TYR","OH"),("HIS","NE2"),("HIS","ND1"),("HIE","NE2"),("HID","ND1")}

WARNINGS = []
def warn(msg):
    WARNINGS.append(msg)
    print(f"  ⚠  {msg}")

def to_np(x):
    """cupy → numpy (no-op for numpy)."""
    if x is None:
        return None
    if hasattr(x, "get") and not isinstance(x, (dict, np.ndarray)):
        x = x.get()
    return np.asarray(x)

TWO_LETTER = {"CL","BR","ZN","MG","MN","FE","CO","NI","CU","CD","NA","LI","CA",
              "SE","SI","AS","AL","AG","AU","PT","PD","HG","PB","SN","MO","CR",
              "BA","SR","CS","RB","BE","TI","ZR","GA","GE","SB","TE","IN","RU","RH"}

def elem_from_atom(atom, resname=""):
    """Element of a Biopython atom: element column first, then the PDB atom name.
    Two-letter symbols (FE, ZN, MN, CL, ...) are recognized, so e.g. FE is not
    read as fluorine. For a monatomic residue the residue name is used as well."""
    el = norm_elem(getattr(atom, "element", "") or "")
    if el and el.upper() != "X":
        return el
    full = getattr(atom, "get_fullname", lambda: atom.get_name())() or atom.get_name()
    nm   = full.strip().upper()
    rn   = str(resname).strip().upper()
    if rn in TWO_LETTER and (nm == rn or nm.rstrip("+-0123456789") == rn):
        return norm_elem(rn)                     # monatomic ion / metal residue
    raw = nm.lstrip("0123456789")
    if rn in PROTEIN_RESNAMES:
        # standard protein atom names: the element is the first letter
        # (CA = alpha carbon, CD1 = carbon, NZ = nitrogen, ...); only MSE SE is Se
        return norm_elem("Se" if (rn == "MSE" and raw.startswith("SE")) else raw[:1])
    two = raw[:2] if not (len(raw) > 2 and raw[2].isalpha()) else ""
    if two in ("CL", "BR"):
        return norm_elem(two)                    # ligand halogens, any justification
    # PDB column convention: single-letter elements are right-justified in the
    # 4-character name field (" CA ", " CD1"), two-letter ones left-justified
    # ("CA  ", "FE  "). A leading blank therefore means a single-letter element,
    # so a ligand carbon named CA or CD1 is not read as calcium or cadmium.
    if full.startswith(" ") and raw:
        return norm_elem(raw[:1])
    if two in TWO_LETTER:
        return norm_elem(two)                    # FE, ZN, MG, ... (left-justified)
    return norm_elem(raw[:1])                    # C, N, O, H, S ...


def norm_elem(el):
    el = "".join(c for c in str(el) if c.isalpha())
    return el[:1].upper() + el[1:].lower() if el else ""


# ── helpers for STEP 1 ─────────────────────────────────────────────
def _formal_charges_from_pdb(pdb_path):
    """{(chain, resseq, atomname): formal charge} from PDB columns 79-80."""
    out = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM","HETATM")) or len(line) < 79:
                continue
            s = line[78:80].strip()
            if not s:
                continue
            sign = -1 if "-" in s else 1
            dig  = "".join(c for c in s if c.isdigit()) or "1"
            try:
                out[(line[21].strip(), int(line[22:26]), line[12:16].strip())] = sign*int(dig)
            except ValueError:
                pass
    return out


def _select_ligand(model, spec):
    """spec: AUTO | RESNAME | RESNAME:CHAIN:RESID  → (residue, chain_id)"""
    cands = []
    for chain in model:
        for res in chain:
            rn = res.get_resname().strip().upper()
            if rn in PROTEIN_RESNAMES or rn in SOLVENT:
                continue
            heavy = [a for a in res if elem_from_atom(a, rn) != "H"]
            cands.append((res, chain.get_id(), rn, len(heavy)))
    if not cands:
        raise ValueError("No non-protein, non-water, non-ion residue found in the PDB.")

    if spec.upper() == "AUTO":
        pool = sorted([c for c in cands if c[2] not in ADDITIVES and c[3] >= 5],
                      key=lambda c: -c[3])
        if not pool:
            raise ValueError(f"No ligand found. Candidates: {sorted({c[2] for c in cands})}. "
                             f"Set ligand_resname manually.")
        if len(pool) > 1:
            warn("Multiple ligand candidates: " +
                 ", ".join(f"{c[2]}:{c[1].strip() or '-'}:{c[0].get_id()[1]}({c[3]} heavy)" for c in pool) +
                 " — using the largest. Use RESNAME:CHAIN:RESID to choose.")
        return pool[0][0], pool[0][1]

    parts = spec.split(":")
    rn = parts[0].upper()
    ch = parts[1] if len(parts) > 1 and parts[1] else None
    ri = int(parts[2]) if len(parts) > 2 and parts[2] else None
    pool = [c for c in cands if c[2] == rn
            and (ch is None or c[1].strip() == ch)
            and (ri is None or c[0].get_id()[1] == ri)]
    if not pool:
        raise ValueError(f"Ligand '{spec}' not found. Candidates: "
                         f"{sorted({f'{c[2]}:{c[1].strip()}:{c[0].get_id()[1]}' for c in cands})}")
    if len(pool) > 1:
        warn(f"{len(pool)} copies of {rn} found — using chain '{pool[0][1].strip() or '-'}' "
             f"resid {pool[0][0].get_id()[1]}. Use RESNAME:CHAIN:RESID to choose another.")
    return pool[0][0], pool[0][1]


def _protein_atoms_openmm(pdb_path, model_id, ph=7.0):
    """All protein atoms (with H) with ff14SB charge, sigma (Å), eps (kcal/mol).
    Chains are split at gaps so OpenMM treats gap ends as termini."""
    from openmm.app import PDBFile, ForceField, Modeller, NoCutoff
    import openmm

    model = BioPDB.PDBParser(QUIET=True).get_structure("p", pdb_path)[model_id]
    ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    seg_map, gaps, art_term, lines = {}, [], set(), []
    serial, nseg = 1, 0
    n_caps = 0
    for chain in model:
        prev, prev_rn, cur = None, None, None
        for res in chain:
            rn = res.get_resname().strip().upper()
            if rn not in PROTEIN_RESNAMES:
                continue
            new_seg = cur is None
            if prev is not None:
                if prev_rn in CAP_C or rn in CAP_N:
                    new_seg = True; n_caps += 1          # capped break: neutral, no warning
                else:
                    dCN = (prev["C"] - res["N"]) if ("C" in prev and "N" in res) else None
                    if dCN is None or dCN > 2.0:
                        new_seg = True
                        gaps.append((chain.id.strip(), prev.get_id()[1], res.get_id()[1], dCN))
                        art_term.add((chain.id.strip(), prev.get_id()[1]))
                        art_term.add((chain.id.strip(), res.get_id()[1]))
            if new_seg:
                if nseg >= len(ids):
                    raise RuntimeError("More than 62 protein segments — too many chain breaks.")
                cur = ids[nseg]; nseg += 1; seg_map[cur] = chain.id.strip()
                if lines:
                    lines.append("TER\n")
            hf, seq, ic = res.get_id()
            for a in res:
                nm = a.get_name(); nm = nm if len(nm) == 4 else " " + nm
                x, y, z = a.get_coord()
                el = elem_from_atom(a, rn).upper()
                lines.append(f"ATOM  {serial % 100000:5d} {nm:<4} {rn:>3} {cur}{seq:4d}{ic:1}   "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {el:>2}\n")
                serial += 1
            prev, prev_rn = res, rn
    lines.append("END\n")
    tmp = Path(f".qmech_protein_{os.getpid()}.pdb")
    tmp.write_text("".join(lines))
    if n_caps:
        print(f"  Capped chain breaks (ACE/NME): {n_caps} — treated as neutral caps")
    for ch, r1, r2, d in gaps:
        why = f"C–N {d:.1f} Å" if d is not None else "backbone C or N atom missing"
        warn(f"Chain break in chain '{ch or '-'}' between residues {r1} and {r2} "
             f"({why}): segments treated separately, gap ends become charged termini.")

    ff = ForceField("amber14-all.xml")
    mk = lambda top: ff.createSystem(top, nonbondedMethod=NoCutoff, constraints=None,
                                     rigidWater=False, removeCMMotion=False)
    pdb = PDBFile(str(tmp))
    top, pos = pdb.topology, pdb.positions
    has_h = any(a.element is not None and a.element.symbol == "H" for a in top.atoms())
    system, how = None, ""
    if has_h:
        try:
            system = mk(top); how = "input hydrogens kept"
        except Exception as e:
            warn(f"OpenMM could not match input protein hydrogens "
                 f"({str(e).splitlines()[0][:90]}); re-protonating at pH {ph}.")
    if system is None:
        try:
            from pdbfixer import PDBFixer
            fx = PDBFixer(filename=str(tmp))
            fx.findMissingResidues(); fx.missingResidues = {}
            fx.findNonstandardResidues(); fx.replaceNonstandardResidues()
            fx.findMissingAtoms(); fx.addMissingAtoms()
            top, pos = fx.topology, fx.positions
            how = "PDBFixer + OpenMM hydrogens"
        except ImportError:
            how = "OpenMM hydrogens"
        m = Modeller(top, pos)
        m.delete([a for a in m.topology.atoms()
                  if a.element is not None and a.element.symbol == "H"])
        try:
            m.addHydrogens(ff, pH=ph)
            top, pos = m.topology, m.positions
            system = mk(top)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"OpenMM/ff14SB could not parameterise the protein ({str(e).splitlines()[0]}). "
                "Fix missing atoms / non-standard residues (install pdbfixer).")
        if has_h:
            warn(f"Protein protonation rebuilt by OpenMM at pH {ph} ({how}); "
                 "protonation states from the input PDB were not kept.")
        else:
            print(f"  Protein had no hydrogens — added by OpenMM at pH {ph} ({how}).")
    tmp.unlink(missing_ok=True)

    nb  = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    xyz = np.array(pos.value_in_unit(openmm.unit.angstrom))
    atoms = []
    for a in top.atoms():
        q, sig, eps = nb.getParticleParameters(a.index)
        r = a.residue
        atoms.append(dict(
            chain=seg_map.get(r.chain.id, r.chain.id), resid=int(r.id),
            icode=(r.insertionCode or "").strip(), resname=r.name, name=a.name,
            elem=a.element.symbol if a.element is not None else "X",
            xyz=xyz[a.index],
            q=q.value_in_unit(openmm.unit.elementary_charge),
            sigma=sig.value_in_unit(openmm.unit.angstrom),
            eps=eps.value_in_unit(openmm.unit.kilojoule_per_mole)*KJ_TO_KCAL))
    return atoms, f"AMBER ff14SB via OpenMM ({how})", art_term


def _alt_path(mol, p, neg, max_len=30):
    """Alternating double/single bond path from positive atom p to a negative
    atom (first bond double, last bond single) in a kekulised molecule."""
    D, S = Chem.BondType.DOUBLE, Chem.BondType.SINGLE
    stack = [(p, D, [p], [])]
    while stack:
        a, need, atoms, bonds = stack.pop()
        if len(bonds) >= max_len:
            continue
        for b in mol.GetAtomWithIdx(a).GetBonds():
            if b.GetBondType() != need:
                continue
            n = b.GetOtherAtomIdx(a)
            if n in atoms:
                continue
            if need == S and n in neg:
                return bonds + [b.GetIdx()], n
            stack.append((n, S if need == D else D, atoms + [n], bonds + [b.GetIdx()]))
    return None


def _clean_resonance(m):
    """Resonance form with the fewest formally charged atoms, obtained by
    recombining separated +/− charges along conjugated paths
    (e.g. [S+]…=C([O-])[O-]  →  s…C(=O)[O-]). Atoms, electrons and net charge
    are unchanged; only the Lewis depiction (and RDKit descriptors) improve."""
    nq = lambda x: sum(1 for a in x.GetAtoms() if a.GetFormalCharge())
    if nq(m) <= 1:
        return m
    rw = Chem.RWMol(m)
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:
        return m
    for _ in range(10):
        neg = {a.GetIdx() for a in rw.GetAtoms() if a.GetFormalCharge() < 0}
        done = False
        for a in rw.GetAtoms():
            if a.GetFormalCharge() <= 0 or not neg:
                continue
            hit = _alt_path(rw, a.GetIdx(), neg)
            if not hit:
                continue
            bonds, n = hit
            for bi in bonds:
                b = rw.GetBondWithIdx(bi)
                b.SetBondType(Chem.BondType.SINGLE if b.GetBondType() == Chem.BondType.DOUBLE
                              else Chem.BondType.DOUBLE)
            a.SetFormalCharge(a.GetFormalCharge() - 1)
            na = rw.GetAtomWithIdx(n); na.SetFormalCharge(na.GetFormalCharge() + 1)
            for x in (a, na):
                x.SetNoImplicit(True); x.SetNumExplicitHs(x.GetNumExplicitHs())
            done = True
            break
        if not done:
            break
    try:
        out = rw.GetMol(); Chem.SanitizeMol(out)
    except Exception:
        return m
    return out if nq(out) < nq(m) else m


def _perceive_ligand(elems, xyz, trial=range(-3, 4)):
    """Chemically valid closed-shell Lewis structures for the ligand as given
    (RDKit xyz2mol), tool-independent. Returns [(charge, n_charged_atoms, smiles)]
    sorted by |charge|, then number of charged atoms. Empty list = the atoms and
    hydrogens do not form a valid molecule (usually missing hydrogens)."""
    block = f"{len(elems)}\n\n" + "\n".join(
        f"{e} {x:.5f} {y:.5f} {z:.5f}" for e, (x, y, z) in zip(elems, xyz))
    Z = sum(pyscf_elements.charge(e) for e in elems)
    out = []
    for c in trial:
        if (Z - c) % 2:
            continue
        try:
            m = Chem.MolFromXYZBlock(block)
            rdDetermineBonds.DetermineBonds(m, charge=c)
            Chem.SanitizeMol(m)
        except Exception:
            continue
        if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
            continue
        bad, n_len = False, 0
        for b in m.GetBonds():            # bond orders vs bond lengths (ranking only)
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if not {m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(j).GetSymbol()} <= {"C","N","O"}:
                continue
            d, t = float(np.linalg.norm(xyz[i]-xyz[j])), b.GetBondType()
            if (t == Chem.BondType.DOUBLE and d > 1.45) or \
               (t == Chem.BondType.AROMATIC and d > 1.47) or \
               (t == Chem.BondType.TRIPLE and d > 1.30):
                n_len += 1
        for a in m.GetAtoms():            # no cumulated double bonds at a bent C/N/O
            if bad or a.GetSymbol() not in ("C","N","O"):
                continue
            mb = [b for b in a.GetBonds() if b.GetBondType() in
                  (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)]
            if len(mb) >= 2:
                i = a.GetIdx(); j, k = mb[0].GetOtherAtomIdx(i), mb[1].GetOtherAtomIdx(i)
                v1, v2 = xyz[j]-xyz[i], xyz[k]-xyz[i]
                if np.degrees(np.arccos(np.clip(v1@v2/np.linalg.norm(v1)/np.linalg.norm(v2), -1, 1))) < 150:
                    bad = True
        if bad:
            continue
        nC = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C" and a.GetFormalCharge())
        if nC:                            # charged carbon only if resonance moves it off C
            try:
                sup = Chem.ResonanceMolSupplier(m, Chem.ALLOW_CHARGE_SEPARATION |
                      Chem.UNCONSTRAINED_ANIONS | Chem.UNCONSTRAINED_CATIONS, 200)
                nC = min(sum(1 for a in r.GetAtoms() if a.GetSymbol() == "C" and a.GetFormalCharge())
                         for r in sup if r is not None)
            except Exception:
                pass
        if nC:
            continue
        m  = _clean_resonance(m)
        nq = sum(1 for a in m.GetAtoms() if a.GetFormalCharge())
        out.append((c, nq, Chem.MolToSmiles(Chem.RemoveHs(m)), n_len))
    out.sort(key=lambda t: (abs(t[0]), t[3], t[1]))
    return [t[:3] for t in out]


def _missing_h_sites(elems, xyz):
    """Carbons whose geometry shows a missing hydrogen (tool-independent):
    ≤1 neighbour, 2 neighbours at a bent angle (<150°), or 3 neighbours in a
    pyramidal (sp3) arrangement. Returns a list of human-readable sites."""
    d = cdist(xyz, xyz)
    sites = []
    for i, e in enumerate(elems):
        if e != "C":
            continue
        nb = [j for j in range(len(elems)) if j != i and
              d[i, j] < 1.25*(COV_R.get(e, 0.8) + COV_R.get(elems[j], 0.8))]
        if len(nb) <= 1:
            sites.append(f"C{i} ({len(nb)} neighbour)")
            continue
        vec = [(xyz[j]-xyz[i])/np.linalg.norm(xyz[j]-xyz[i]) for j in nb]
        ang = lambda a, b: np.degrees(np.arccos(np.clip(a@b, -1, 1)))
        if len(nb) == 2 and ang(vec[0], vec[1]) < 150:
            sites.append(f"C{i} (2 neighbours, {ang(vec[0], vec[1]):.0f}°)")
        elif len(nb) == 3:
            tot = ang(vec[0], vec[1]) + ang(vec[0], vec[2]) + ang(vec[1], vec[2])
            if tot < 340:
                sites.append(f"C{i} (3 neighbours, pyramidal {tot:.0f}°)")
    return sites


# ══════════════════════════════════════════════════════════════════
#  STEP 1 : PDB Parsing & Binding Pocket Extraction
# ══════════════════════════════════════════════════════════════════
def extract_pocket_from_pdb(pdb_path, ligand_resname="AUTO",
                             cutoff=8.0, ligand_charge=None,
                             ph=7.0, include_waters=False):
    print(f"\n{'═'*66}")
    print(f"  STEP 1  |  PDB Parsing & Pocket Extraction")
    print(f"{'═'*66}")
    print(f"  File: {pdb_path}")

    parser = BioPDB.PDBParser(QUIET=True)
    struct = parser.get_structure("mol", pdb_path)
    models = list(struct)
    if len(models) > 1:
        warn(f"PDB has {len(models)} models — only the first model is used.")
    model = models[0]

    # ── Ligand ─────────────────────────────────────────────────────
    lig_res, lig_chain = _select_ligand(model, ligand_resname)
    ligand_resname = lig_res.get_resname().strip().upper()
    lig_resid      = lig_res.get_id()[1]
    print(f"  ✓  Ligand: '{ligand_resname}'  chain '{lig_chain.strip() or '-'}'  resid {lig_resid}")

    lig_names, lig_elems, lig_pos = [], [], []
    for atom in lig_res:
        el = elem_from_atom(atom, ligand_resname)
        try:
            pyscf_elements.charge(el)
        except Exception:
            raise ValueError(f"Unrecognised element '{el}' for ligand atom {atom.get_name()}")
        lig_names.append(atom.get_name().strip())
        lig_elems.append(el)
        lig_pos.append(np.array(atom.get_coord(), dtype=float))
    lig_pos  = np.array(lig_pos)
    lig_cent = lig_pos.mean(axis=0)

    n_h = lig_elems.count("H")
    if n_h == 0:
        raise ValueError(
            "Ligand has NO hydrogens. DFT on a heavy-atom-only ligand describes a different "
            "(radical) molecule. Add hydrogens first with any tool, e.g. "
            "obabel lig.pdb -O lig_H.pdb -p 7.4 | UCSF Chimera: AddH | PyMOL: h_add | Maestro/MOE.")

    # missing hydrogens? (geometry-based, independent of any tool)
    miss = _missing_h_sites(lig_elems, lig_pos)
    if miss:
        raise ValueError(
            f"Ligand is missing hydrogens at {len(miss)} carbon(s): {', '.join(miss[:6])}"
            f"{' …' if len(miss) > 6 else ''}. Docking outputs such as AutoDock Vina/PDBQT keep "
            "only polar H. Add ALL hydrogens (e.g. obabel lig.pdb -O lig_H.pdb -p 7.4, "
            "UCSF Chimera AddH, PyMOL h_add) and re-run.")

    # net charge: argument > PDB formal charges > perceived > 0; cross-checked when possible
    perc   = _perceive_ligand(lig_elems, lig_pos)
    valid  = sorted({c for c, _, _ in perc}, key=abs)
    fc_map = _formal_charges_from_pdb(pdb_path)
    ch_key = lig_chain.strip()
    fc_sum = sum(fc_map.get((ch_key, lig_resid, n), 0) for n in lig_names)
    has_fc = any((ch_key, lig_resid, n) in fc_map for n in lig_names)
    if ligand_charge is not None:
        lig_charge, charge_src = int(ligand_charge), "command line"
        if perc and lig_charge not in valid:
            raise ValueError(
                f"Charge {lig_charge:+d} is not chemically consistent with the ligand as given. "
                f"Consistent charge: {perc[0][0]:+d}  ({perc[0][2]}). Check the protonation state.")
    elif has_fc:
        lig_charge, charge_src = fc_sum, "PDB formal-charge columns"
        if perc and lig_charge not in valid:
            raise ValueError(
                f"PDB formal charges sum to {lig_charge:+d}, but the structure is consistent with "
                f"{perc[0][0]:+d} ({perc[0][2]}). Give the correct charge as the 4th argument.")
    elif perc:
        lig_charge, charge_src = perc[0][0], "perceived from 3D structure"
        if len(valid) > 1:
            warn(f"Structure also admits charge(s) {[c for c in valid if c != lig_charge]}; "
                 f"using {lig_charge:+d}. Pass the charge explicitly if this is wrong.")
    else:
        lig_charge, charge_src = 0, "default (0) — not verified"
    if not perc:
        warn("RDKit could not derive a Lewis structure for this ligand (unusual chemistry or "
             f"distorted geometry), so the net charge {lig_charge:+d} [{charge_src}] could not be "
             "cross-checked. If the ligand is ionised, give its charge as the 4th argument.")
    else:
        best = [p for p in perc if p[0] == lig_charge]
        if best:
            print(f"  Ligand structure (perceived): {best[0][2]}")
    nelec = sum(pyscf_elements.charge(e) for e in lig_elems) - lig_charge
    print(f"  Ligand  : {len(lig_elems)} atoms ({len(lig_elems)-n_h} heavy, {n_h} H) | "
          f"net charge {lig_charge:+d} [{charge_src}] | {nelec} electrons")
    if nelec % 2:
        raise ValueError(
            f"Odd electron count ({nelec}) with charge {lig_charge:+d}: hydrogens or net charge "
            "are wrong. Give the correct ligand charge as the 4th argument.")

    # ── Protein (ff14SB via OpenMM) ───────────────────────────────
    prot, charge_model, art_term = _protein_atoms_openmm(pdb_path, model.id, ph)

    # ── Ions / waters ─────────────────────────────────────────────
    extra, n_w_skip = [], 0
    for chain in model:
        for res in chain:
            rn = res.get_resname().strip().upper()
            if res is lig_res:
                continue
            if rn in ION_CHARGES:
                a = list(res)[0]
                extra.append(dict(chain=chain.get_id().strip(), resid=res.get_id()[1], icode="",
                                  resname=rn, name=a.get_name(), elem=elem_from_atom(a, rn),
                                  xyz=np.array(a.get_coord(), float), q=float(ION_CHARGES[rn]),
                                  sigma=0.0, eps=0.0))
            elif rn in WATER_RESNAMES and include_waters:
                at = list(res)
                if not any(elem_from_atom(a, rn) == "H" for a in at):
                    n_w_skip += 1; continue
                for a in at:
                    el = elem_from_atom(a, rn)
                    extra.append(dict(chain=chain.get_id().strip(), resid=res.get_id()[1], icode="",
                                      resname="HOH", name=a.get_name(), elem=el,
                                      xyz=np.array(a.get_coord(), float),
                                      q=-0.834 if el == "O" else 0.417,
                                      sigma=3.15061 if el == "O" else 0.0,
                                      eps=0.1521 if el == "O" else 0.0))
    if n_w_skip:
        warn(f"{n_w_skip} waters without hydrogens not included (TIP3P needs H).")

    all_mm = prot + extra
    n_prot_res = len({(a["chain"], a["resid"], a["icode"]) for a in prot})
    print(f"  Protein : {len(prot)} atoms (incl. H) | {n_prot_res} residues")
    print(f"  Charges : {charge_model}")

    # ── Distance-based pocket: WHOLE residues within cutoff ────────
    lig_heavy = lig_pos[[i for i, e in enumerate(lig_elems) if e != "H"]]
    groups = {}
    for i, a in enumerate(all_mm):
        groups.setdefault((a["chain"], a["resid"], a["icode"], a["resname"]), []).append(i)
    res_mind = {}
    for k, idx in groups.items():
        hv = [i for i in idx if all_mm[i]["elem"] != "H"] or idx
        res_mind[k] = float(cdist(np.array([all_mm[i]["xyz"] for i in hv]), lig_heavy).min())
    keys = sorted([k for k in groups if res_mind[k] <= cutoff], key=lambda k: (k[0], k[1]))
    pocket_atom_list = [all_mm[i] for k in keys for i in groups[k]]

    pocket_residues = [{"resname": k[3], "resid": k[1], "chain": k[0],
                        "min_dist_A": round(res_mind[k], 2)} for k in keys]

    print(f"\n  Pocket Residues ({len(pocket_residues)}) within {cutoff} Å (whole residues):")
    line = "  "
    for i, r in enumerate(pocket_residues):
        line += f"{r['chain'] or '-'}:{r['resname']}{r['resid']}  "
        if (i+1) % 7 == 0: print(line); line = "  "
    if line.strip(): print(line)

    p_charges = np.array([a["q"] for a in pocket_atom_list])
    print(f"\n  Pocket atoms : {len(pocket_atom_list)}")
    print(f"  MM total charge: {p_charges.sum():+.3f} e")
    if abs(p_charges.sum() - round(p_charges.sum())) > 0.02:
        warn(f"MM net charge {p_charges.sum():+.3f} is not an integer — check residue parameters.")

    near_gap = sorted({f"{k[0] or '-'}:{k[3]}{k[1]}" for k in keys if (k[0], k[1]) in art_term})
    if near_gap:
        warn("Artificial charged termini from chain breaks lie inside the MM region: "
             + ", ".join(near_gap) + ". Consider modelling the missing loop (e.g. MODELLER, SWISS-MODEL, "
             "PDBFixer with SEQRES, Maestro Prime) or report this in the methods.")
    metals = [a for a in pocket_atom_list if a["resname"] in METALS]
    for a in metals:
        warn(f"Metal {a['resname']}{a['resid']} included as a {a['q']:+.0f} point charge "
             "(no charge transfer; the QM region near it may be over-polarised).")

    return {
        "lig_pos"        : lig_pos,
        "lig_elems"      : lig_elems,
        "lig_names"      : lig_names,
        "lig_resname"    : ligand_resname,
        "lig_chain"      : ch_key,
        "lig_resid"      : lig_resid,
        "lig_charge"     : lig_charge,
        "lig_charge_src" : charge_src,
        "lig_centroid"   : lig_cent,
        "mm_coords"      : np.array([a["xyz"] for a in pocket_atom_list]),
        "mm_charges"     : p_charges,
        "mm_sigma"       : np.array([a["sigma"] for a in pocket_atom_list]),
        "mm_eps"         : np.array([a["eps"] for a in pocket_atom_list]),
        "mm_elems"       : [a["elem"] for a in pocket_atom_list],
        "mm_names"       : [f"{a['resname']}{a['resid']}:{a['name']}" for a in pocket_atom_list],
        "mm_atomnames"   : [a["name"] for a in pocket_atom_list],
        "mm_resnames"    : [a["resname"] for a in pocket_atom_list],
        "mm_resids"      : [a["resid"] for a in pocket_atom_list],
        "mm_chains"      : [a["chain"] for a in pocket_atom_list],
        "pocket_residues": pocket_residues,
        "n_pocket_atoms" : len(pocket_atom_list),
        "n_pocket_res"   : len(pocket_residues),
        "charge_model"   : charge_model,
        "metals"         : metals,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 2 : Build PySCF Mol from PDB Coordinates
# ══════════════════════════════════════════════════════════════════
def build_ligand_mol(pocket, basis="def2-SVP", max_memory=80000):
    print(f"\n{'═'*66}")
    print(f"  STEP 2  |  Building Ligand for PySCF")
    print(f"{'─'*66}")

    elems  = pocket["lig_elems"]
    coords = pocket["lig_pos"].astype(float)
    charge = pocket["lig_charge"]
    n_heavy= sum(1 for e in elems if e != 'H')
    print(f"  Heavy atoms : {n_heavy}  →  Basis: {basis}")

    # RDKit mol for connectivity + bond orders (uses the net charge)
    xyz_block = f"{len(elems)}\n\n" + "\n".join(
        f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, (x, y, z) in zip(elems, coords))
    rdkit_mol, smiles = None, None
    try:
        m = Chem.MolFromXYZBlock(xyz_block)
        rdDetermineBonds.DetermineBonds(m, charge=charge)
        Chem.SanitizeMol(m)
        m = _clean_resonance(m)
        rdkit_mol = m
        smiles = Chem.MolToSmiles(Chem.RemoveHs(m))
        print(f"  Perceived structure (CHECK THIS): {smiles}")
    except Exception:
        warn(f"RDKit could not assign bond orders for net charge {charge:+d} — the charge may "
             "be wrong or the geometry distorted. RDKit descriptors/warheads skipped.")

    # neighbour list (RDKit bonds, else covalent radii)
    nbrs = [[] for _ in elems]
    if rdkit_mol is not None:
        for b in rdkit_mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            nbrs[i].append(j); nbrs[j].append(i)
    else:
        d = cdist(coords, coords)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                if d[i, j] < 1.25*(COV_R.get(elems[i], 0.8) + COV_R.get(elems[j], 0.8)):
                    nbrs[i].append(j); nbrs[j].append(i)

    # Build PySCF mol (closed shell; no silent spin change)
    pyscf_mol = gto.Mole()
    pyscf_mol.atom   = [[e, tuple(p)] for e, p in zip(elems, coords)]
    pyscf_mol.basis  = basis
    pyscf_mol.charge = charge
    pyscf_mol.spin   = 0
    pyscf_mol.unit   = 'Angstrom'
    pyscf_mol.verbose     = 3
    pyscf_mol.max_memory  = max_memory
    pyscf_mol.build()

    print(f"  PySCF mol  : {pyscf_mol.natm} atoms  "
          f"| {pyscf_mol.nao} AOs  "
          f"| charge={pyscf_mol.charge:+d}  "
          f"| basis={basis}")

    return {
        "pyscf_mol" : pyscf_mol,
        "rdkit_mol" : rdkit_mol,
        "smiles"    : smiles,
        "nbrs"      : nbrs,
        "coords"    : coords,
        "elems"     : elems,
        "names"     : pocket["lig_names"],
        "basis"     : basis,
        "charge"    : pyscf_mol.charge,
        "n_heavy"   : n_heavy,
    }


# ══════════════════════════════════════════════════════════════════
#  Shared helpers: SCF with convergence check, Hirshfeld, potentials
# ══════════════════════════════════════════════════════════════════
SOLVENTS = {"water": 78.3553, "dmso": 46.826, "methanol": 32.613,
            "ethanol": 24.852, "acetonitrile": 35.688, "chloroform": 4.7113}
SOLV = {"name": None, "eps": None}        # set in run_pipeline

def _make_ks(mol, functional, open_shell=False, gpu=True, pcm=True):
    mod = dft if (gpu and GPU_OK) else cpu_dft
    mf  = mod.UKS(mol) if open_shell else mod.RKS(mol)
    mf.xc = functional
    mf.grids.level = 3
    mf.conv_tol    = 1e-9
    mf.max_cycle   = 200
    if pcm and SOLV["eps"]:
        mf = mf.PCM()
        mf.with_solvent.method = "IEF-PCM"
        mf.with_solvent.eps    = SOLV["eps"]
    return mf

def _run_scf(mol, functional, label, open_shell=False, gpu=True, dm0=None):
    t0 = time.time()
    mf = _make_ks(mol, functional, open_shell, gpu)
    if hasattr(mf, "diis_space"):
        mf.diis_space = 12
    e  = mf.kernel(dm0=dm0) if (dm0 is not None and not (gpu and GPU_OK)) else mf.kernel()
    conv, engine = bool(mf.converged), ("GPU" if (gpu and GPU_OK) else "CPU")
    if not conv:
        warn(f"{label}: SCF not converged (DIIS) — retrying with second-order solver (same basis).")
        try:
            mf = _make_ks(mol, functional, open_shell, gpu=False).newton()
            engine = "CPU-Newton"
        except Exception:
            mf = _make_ks(mol, functional, open_shell, gpu=False)
            mf.level_shift, mf.damp, mf.max_cycle = 0.3, 0.5, 400
            engine = "CPU-levelshift"
        e  = mf.kernel(); conv = bool(mf.converged)
        if not conv:
            warn(f"{label}: SCF STILL NOT CONVERGED — results for this state are unreliable.")
    if SOLV["eps"] and not label.startswith("Gas"):
        engine += f"+PCM({SOLV['name']})"
    print(f"  {label:<22}: E = {float(e):.8f} Ha | converged={conv} | {engine} | {time.time()-t0:.1f}s")
    return mf, float(e), conv

def _dm_total(mf):
    dm = to_np(mf.make_rdm1())
    return dm[0] + dm[1] if dm.ndim == 3 else dm


class Hirshfeld:
    """Hirshfeld partitioning; free-atom densities = spherically averaged
    atomic HF densities (PySCF atom_hf) in the same basis, Becke grid."""
    def __init__(self, mol, grid_level=3):
        from pyscf.dft import gen_grid
        from pyscf.scf import atom_hf
        g = gen_grid.Grids(mol); g.level = grid_level; g.build()
        self.coords, self.weights, self.mol = g.coords, g.weights, mol
        self.aoslice = mol.aoslice_by_atom()
        atm = atom_hf.get_atm_nrhf(mol)
        self.atom_dm = []
        for ia in range(mol.natm):
            s = mol.atom_symbol(ia)
            if s not in atm:
                s = mol.atom_pure_symbol(ia)
            _, _, c, occ = atm[s]
            self.atom_dm.append((c*occ) @ c.conj().T)

    def populations(self, dms, batch=16000):
        from pyscf.dft import numint
        mol  = self.mol
        pops = np.zeros((len(dms), mol.natm))
        for p0 in range(0, len(self.weights), batch):
            c = self.coords[p0:p0+batch]; w = self.weights[p0:p0+batch]
            ao = numint.eval_ao(mol, c, deriv=0)
            rho0 = np.empty((mol.natm, len(w)))
            for ia, (s, D) in enumerate(zip(self.aoslice, self.atom_dm)):
                a = ao[:, s[2]:s[3]]
                rho0[ia] = np.einsum("gi,ij,gj->g", a, D, a)
            wf = rho0 / np.maximum(rho0.sum(0), 1e-30)
            for d, dm in enumerate(dms):
                pops[d] += wf @ (w * numint.eval_rho(mol, ao, dm))
        return pops


def qm_potential(mol, dm, pts_ang, batch=600):
    """Electrostatic potential of the QM region (a.u.) at points given in Å:
    V(r) = Σ Z_A/|R_A-r| - ∫ρ(r')/|r-r'|, analytical integrals (as cubegen.mep)."""
    pts = np.asarray(pts_ang) * ANG_TO_BOHR
    V_nuc = (mol.atom_charges()[None, :] / cdist(pts, mol.atom_coords())).sum(1)
    V_ele = np.empty(len(pts))
    for p0 in range(0, len(pts), batch):
        fakemol = gto.fakemol_for_charges(pts[p0:p0+batch])
        ints    = pyscf_df.incore.aux_e2(mol, fakemol, intor='int3c2e')
        V_ele[p0:p0+batch] = np.einsum('ijp,ij->p', ints, dm)
    return V_nuc - V_ele, V_nuc, V_ele


# ══════════════════════════════════════════════════════════════════
#  STEP 3 : Real DFT — B3LYP/def2-SVP
# ══════════════════════════════════════════════════════════════════
def run_dft_neutral(lig, functional="B3LYP", n_threads=24):
    print(f"\n{'═'*66}")
    print(f"  STEP 3  |  Real DFT  [{functional}/{lig['basis']}]")
    print(f"{'─'*66}")

    lib.num_threads(n_threads)
    mol = lig["pyscf_mol"]
    print(f"  Running {functional}/{lig['basis']} SCF ..."
          + (f"  [IEF-PCM, {SOLV['name']}]" if SOLV["eps"] else "  [gas phase]"))
    print(f"  Threads: {n_threads}  |  Memory: {mol.max_memory} MB")
    mf, energy, converged = _run_scf(mol, functional, "Neutral (N)")

    # HOMO/LUMO (closed shell)
    mo_e   = to_np(mf.mo_energy)
    mo_occ = to_np(mf.mo_occ)
    homo_i = int(np.where(mo_occ > 0)[0].max())
    lumo_i = homo_i + 1

    HOMO_ev = float(mo_e[homo_i]) * HARTREE_TO_EV
    LUMO_ev = float(mo_e[lumo_i]) * HARTREE_TO_EV
    gap_ev  = LUMO_ev - HOMO_ev
    IP_ev   = -HOMO_ev
    EA_ev   = -LUMO_ev
    chi     = (IP_ev + EA_ev) / 2          # electronegativity (= -μ)
    eta     = IP_ev - EA_ev                # chemical hardness (Parr 1999: η = I - A)
    if eta <= 0:
        warn("Non-positive HOMO-LUMO gap — conceptual-DFT descriptors are meaningless.")
        eta = 1e-3
    omega   = chi**2 / (2*eta)             # electrophilicity (Parr 1999: ω = μ²/2η)
    S       = 1.0 / eta                    # global softness (S = 1/η)
    dN_max  = chi / eta                    # max charge transfer (ΔNmax = -μ/η)

    print(f"\n  ── Real Orbital Energies ──────────────────────")
    print(f"  HOMO  (orbital {homo_i}): {HOMO_ev:>9.4f} eV")
    print(f"  LUMO  (orbital {lumo_i}): {LUMO_ev:>9.4f} eV")
    print(f"  HOMO-LUMO gap          : {gap_ev:>9.4f} eV")
    print(f"\n  ── Koopmans DFT Descriptors ───────────────────")
    print(f"  IP  = -E(HOMO) : {IP_ev:>9.4f} eV")
    print(f"  EA  = -E(LUMO) : {EA_ev:>9.4f} eV")
    print(f"  χ (chi)        : {chi:>9.4f} eV")
    print(f"  η (eta)        : {eta:>9.4f} eV")
    print(f"  ω (omega)      : {omega:>9.4f} eV")
    print(f"  S (softness)   : {S:>9.4f} eV⁻¹")

    # Hirshfeld charges (gas phase)
    dm_n  = _dm_total(mf)
    hirsh = Hirshfeld(mol)
    pop_n = hirsh.populations([dm_n])[0]
    print(f"  Σ Hirshfeld population = {pop_n.sum():.3f} (expected {mol.nelectron})")
    hirshfeld_charges = mol.atom_charges().astype(float) - pop_n

    # RDKit descriptors (None if structure could not be perceived)
    rdmol = lig["rdkit_mol"]
    if rdmol is not None:
        mw   = Descriptors.MolWt(rdmol)
        logp = Descriptors.MolLogP(rdmol)
        tpsa = rdMolDescriptors.CalcTPSA(rdmol)
        hbd  = rdMolDescriptors.CalcNumHBD(rdmol)
        hba  = rdMolDescriptors.CalcNumHBA(rdmol)
        qed  = QED.qed(Chem.RemoveHs(rdmol))
        rot  = rdMolDescriptors.CalcNumRotatableBonds(rdmol)
        arom = rdMolDescriptors.CalcNumAromaticRings(rdmol)
    else:
        mw = logp = tpsa = hbd = hba = qed = rot = arom = None
    rnd = lambda v, n: None if v is None else round(v, n)
    toi = lambda v: None if v is None else int(v)

    descriptors = {
        "DFT functional"           : functional,
        "Basis set"                : lig["basis"],
        "Net charge"               : lig["charge"],
        "Solvent model"            : (f"IEF-PCM ({SOLV['name']}, eps={SOLV['eps']})"
                                      if SOLV["eps"] else "gas phase"),
        "SCF converged"            : converged,
        "SCF energy (Ha)"          : round(energy, 8),
        "SCF energy (kcal/mol)"    : round(energy*HARTREE_TO_KCAL, 4),
        "HOMO energy (eV)"         : round(HOMO_ev, 4),
        "LUMO energy (eV)"         : round(LUMO_ev, 4),
        "HOMO-LUMO gap (eV)"       : round(gap_ev,  4),
        "IP = -E(HOMO) (eV)"       : round(IP_ev,   4),
        "EA = -E(LUMO) (eV)"       : round(EA_ev,   4),
        "Electronegativity χ (eV)" : round(chi,     4),
        "Chemical Hardness η (eV)" : round(eta,     4),  # η = IP - EA (Parr 1999)
        "Global Softness S (eV⁻¹)" : round(S,       4),  # S = 1/η
        "Electrophilicity ω (eV)"  : round(omega,   4),  # ω = χ²/2η (Parr 1999)
        "ΔN_max"                   : round(dN_max,  4),  # ΔNmax = χ/η
        "SMILES (perceived)"       : lig["smiles"],
        "MW (Da)"                  : rnd(mw,   2),
        "LogP"                     : rnd(logp, 3),
        "TPSA (Å²)"                : rnd(tpsa, 2),
        "HBD"                      : toi(hbd),
        "HBA"                      : toi(hba),
        "QED"                      : rnd(qed,  3),
        "RotBonds"                 : toi(rot),
        "AromaticRings"            : toi(arom),
    }

    return {
        "mf"               : mf,
        "mol_used"         : mol,
        "energy"           : energy,
        "converged"        : converged,
        "dm_n"             : dm_n,
        "mo_energy"        : mo_e,
        "mo_occ"           : mo_occ,
        "homo_idx"         : homo_i,
        "lumo_idx"         : lumo_i,
        "HOMO_ev"          : HOMO_ev,
        "LUMO_ev"          : LUMO_ev,
        "gap_ev"           : gap_ev,
        "hirsh"            : hirsh,
        "hirshfeld_charges": hirshfeld_charges,
        "descriptors"      : descriptors,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 4 : Real Fukui Indices — N/N+1/N-1 DFT (Hirshfeld condensed)
# ══════════════════════════════════════════════════════════════════
def run_fukui(lig, dft_n, functional="B3LYP"):
    print(f"\n{'═'*66}")
    print(f"  STEP 4  |  Real Fukui Indices  [N / N+1 / N-1 DFT]")
    print(f"{'─'*66}")

    mol_n = dft_n["mol_used"]
    hirsh = dft_n["hirsh"]

    # N+1 (anion) and N-1 (cation), frozen geometry
    mol_p = mol_n.copy(); mol_p.charge = mol_n.charge - 1; mol_p.spin = 1; mol_p.build()
    mf_p, E_p, conv_p = _run_scf(mol_p, functional, "Anion (N+1)", open_shell=True)
    mol_m = mol_n.copy(); mol_m.charge = mol_n.charge + 1; mol_m.spin = 1; mol_m.build()
    mf_m, E_m, conv_m = _run_scf(mol_m, functional, "Cation (N-1)", open_shell=True)
    converged = conv_p and conv_m
    if not converged:
        warn("An ion SCF did not converge — Fukui indices are unreliable.")
    b = lig["basis"].lower()
    if not any(t in b for t in ("+", "aug", "svpd", "tzvpd", "tzvppd", "qzvpd", "qzvppd")):
        warn(f"{lig['basis']} has no diffuse functions; the anion (f+) is described less "
             "accurately (def2-SVPD recommended for final results).")

    P = hirsh.populations([dft_n["dm_n"], _dm_total(mf_p), _dm_total(mf_m)])
    q_n, q_p, q_m = P[0], P[1], P[2]            # electron populations

    f_plus  = q_p - q_n
    f_minus = q_n - q_m
    f_zero  = (f_plus + f_minus) / 2
    delta_f = f_plus - f_minus

    top_fp = int(np.argmax(f_plus))
    top_fm = int(np.argmax(f_minus))

    print(f"\n  ── Real Fukui Indices (Hirshfeld condensed) ───")
    print(f"  Σf+ = {f_plus.sum():.3f}  |  Σf- = {f_minus.sum():.3f}  (both should be 1.000)")
    print(f"  {'Atom':<8} {'Symbol':<6} {'f+':<10} {'f-':<10} "
          f"{'f0':<10} {'Δf':<10}")
    print(f"  {'─'*54}")
    for i in range(mol_n.natm):
        sym = mol_n.atom_symbol(i)
        print(f"  {i:<8} {sym:<6} {f_plus[i]:>+9.4f}  "
              f"{f_minus[i]:>+9.4f}  {f_zero[i]:>+9.4f}  "
              f"{delta_f[i]:>+9.4f}")

    print(f"\n  Most electrophilic: atom {top_fp} "
          f"({mol_n.atom_symbol(top_fp)}) f+={f_plus[top_fp]:+.4f}")
    print(f"  Most nucleophilic : atom {top_fm} "
          f"({mol_n.atom_symbol(top_fm)}) f-={f_minus[top_fm]:+.4f}")

    vIP = (E_m - dft_n["energy"]) * HARTREE_TO_EV
    vEA = (dft_n["energy"] - E_p) * HARTREE_TO_EV
    print(f"  Vertical IP (ΔSCF) = {vIP:.4f} eV | Vertical EA (ΔSCF) = {vEA:.4f} eV")

    return {
        "f_plus"  : f_plus, "f_minus": f_minus,
        "f_zero"  : f_zero, "delta_f": delta_f,
        "q_n":q_n, "q_p":q_p, "q_m":q_m,
        "E_anion" : E_p, "E_cation": E_m,
        "vertical_IP_eV": vIP, "vertical_EA_eV": vEA,
        "converged": converged,
        "top_electrophilic_atom": top_fp,
        "top_nucleophilic_atom" : top_fm,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 5 : Real MEP from DFT Electron Density
# ══════════════════════════════════════════════════════════════════
def compute_real_mep(lig, dft_n, n_pts_per_dim=40):
    print(f"\n{'═'*66}")
    print(f"  STEP 5  |  Real MEP from DFT Electron Density")
    print(f"{'─'*66}")

    mol    = dft_n["mol_used"]
    dm     = dft_n["dm_n"]
    coords = lig["coords"]
    elems  = lig["elems"]

    lo = coords.min(axis=0)-3.0; hi = coords.max(axis=0)+3.0
    xs,ys,zs = (np.linspace(lo[i],hi[i],n_pts_per_dim) for i in range(3))
    gx,gy,gz = np.meshgrid(xs,ys,zs,indexing="ij")
    grid_ang  = np.column_stack([gx.ravel(),gy.ravel(),gz.ravel()])

    # points on a molecular-surface shell: 1.0–1.4 × vdW radius
    vdw_r  = np.array([VDW_ANG.get(e, 1.7) for e in elems])
    ratio  = (cdist(grid_ang, coords) / vdw_r).min(axis=1)
    grid_out = grid_ang[(ratio >= 1.0) & (ratio <= 1.4)]

    print(f"  Grid points (1.0–1.4 × vdW shell): {len(grid_out):,}")
    print(f"  Computing nuclear + electronic MEP (analytical integrals) ...")

    V_mep, V_nuc, V_elec = qm_potential(mol, dm, grid_out)
    Vk = V_mep * HARTREE_TO_KCAL
    near = lambda p: int(np.argmin(np.linalg.norm(coords - p, axis=1)))
    i_min, i_max = int(V_mep.argmin()), int(V_mep.argmax())

    print(f"  MEP range: [{V_mep.min():.5f}, {V_mep.max():.5f}] a.u. "
          f"= [{Vk.min():.2f}, {Vk.max():.2f}] kcal/mol")
    print(f"  Nucleophilic site  (min): near {elems[near(grid_out[i_min])]}{near(grid_out[i_min])}")
    print(f"  Electrophilic site (max): near {elems[near(grid_out[i_max])]}{near(grid_out[i_max])}")

    return {
        "grid"      : grid_out, "mep": V_mep,
        "V_nuc"     : V_nuc,   "V_elec": V_elec,
        "atom_coords": coords,
        "MEP_min"   : float(V_mep.min()),
        "MEP_max"   : float(V_mep.max()),
        "MEP_min_kcal": float(Vk.min()),
        "MEP_max_kcal": float(Vk.max()),
        "MEP_mean"  : float(V_mep.mean()),
        "MEP_std"   : float(V_mep.std()),
        "pct_pos"   : float((V_mep>0).mean()*100),
        "pct_neg"   : float((V_mep<0).mean()*100),
        "nuc_site"  : grid_out[i_min].tolist(),
        "elec_site" : grid_out[i_max].tolist(),
        "nuc_site_atom" : f"{elems[near(grid_out[i_min])]}{near(grid_out[i_min])}",
        "elec_site_atom": f"{elems[near(grid_out[i_max])]}{near(grid_out[i_max])}",
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 6 : Real QM/MM — PySCF Electrostatic Embedding
# ══════════════════════════════════════════════════════════════════
def run_qmmm(lig, pocket, dft_n, functional="B3LYP"):
    print(f"\n{'═'*66}")
    print(f"  STEP 6  |  Real QM/MM  [PySCF electrostatic embedding]")
    print(f"{'─'*66}")

    mol        = lig["pyscf_mol"]
    mm_coords  = pocket["mm_coords"]           # Å (mol.unit = 'Angstrom')
    mm_charges = pocket["mm_charges"]

    print(f"  QM region : {mol.natm} atoms  ({functional}/{lig['basis']})")
    print(f"  MM region : {len(mm_charges)} atoms  ({pocket['charge_model']})")
    print(f"  MM total charge: {mm_charges.sum():+.3f} e")

    # Gas-phase reference with identical CPU settings (GPU4PySCF not used with qmmm)
    if GPU_OK or SOLV["eps"]:
        _s = dict(SOLV); SOLV.update(name=None, eps=None)       # QM/MM is gas-phase
        try:
            mf_gas, E_gas, conv_g = _run_scf(mol, functional, "Gas ref (CPU)", gpu=False)
        finally:
            SOLV.update(_s)
        dm_gas = _dm_total(mf_gas)
    else:
        E_gas, conv_g, dm_gas = dft_n["energy"], dft_n["converged"], dft_n["dm_n"]
        print(f"  {'Gas ref (CPU)':<22}: E = {E_gas:.8f} Ha (reused from STEP 3)")

    mf_base = _make_ks(mol, functional, gpu=False, pcm=False)
    mf_qmmm = pyscf_qmmm.mm_charge(mf_base, mm_coords, mm_charges, unit='Angstrom')
    mf_qmmm.verbose = 3

    print(f"\n  Running embedded QM/MM SCF ...")
    t0 = time.time()
    E_qmmm = float(mf_qmmm.kernel(dm0=dm_gas))
    conv_e = bool(mf_qmmm.converged)
    if not conv_e:
        warn("Embedded SCF not converged (DIIS) — retrying with second-order solver.")
        try:
            mf_qmmm = mf_qmmm.newton()
            E_qmmm = float(mf_qmmm.kernel(dm0=dm_gas)); conv_e = bool(mf_qmmm.converged)
        except Exception as e:
            warn(f"Newton retry failed: {e}")
    if not conv_e:
        warn("Embedded SCF NOT CONVERGED — QM/MM energies unreliable.")
    print(f"  QM/MM SCF done in {time.time()-t0:.1f}s  (converged={conv_e})")
    dm_qmmm = _dm_total(mf_qmmm)

    # Interaction energy = E(embedded) − E(gas): electrostatics + polarization
    E_int_elec = (E_qmmm - E_gas) * HARTREE_TO_KCAL

    # Frozen-density electrostatics per MM atom: q_j · V_QM(r_j) (exact, analytical)
    V_at_mm, _, _ = qm_potential(mol, dm_gas, mm_coords)
    e_elec_atom   = mm_charges * V_at_mm * HARTREE_TO_KCAL
    E_elec_cl     = e_elec_atom.sum()          # frozen gas-density electrostatics
    E_pol         = E_int_elec - E_elec_cl     # polarization (incl. QM distortion)

    # Lennard-Jones: GAFF-like ligand × ff14SB protein, Lorentz–Berthelot
    lig_coords = lig["coords"].astype(float)
    dist_qm_mm = cdist(lig_coords, mm_coords)
    ls, le = [], []
    for i, e in enumerate(lig["elems"]):
        key = e
        if e == "H":
            partner = [lig["elems"][j] for j in lig["nbrs"][i]]
            key = "H_C" if (partner and partner[0] == "C") else "H_X"
        s_, e_ = LIG_LJ.get(key, (3.4, 0.086)); ls.append(s_); le.append(e_)
    sij = (np.array(ls)[:, None] + pocket["mm_sigma"][None, :]) / 2
    eij = np.sqrt(np.array(le)[:, None] * pocket["mm_eps"][None, :])
    sr6 = (sij / dist_qm_mm)**6
    lj_atom = (4*eij*(sr6**2 - sr6)).sum(axis=0)
    E_LJ    = lj_atom.sum()
    E_total = E_int_elec + E_LJ

    hl = np.array([e != "H" for e in lig["elems"]])
    hm = np.array([e != "H" for e in pocket["mm_elems"]])
    d_min = float(dist_qm_mm[np.ix_(hl, hm)].min()) if hm.any() else 99.0
    if d_min < 2.2:
        warn(f"Ligand–protein heavy-atom contact of {d_min:.2f} Å (clash or covalent bond): "
             "LJ and embedding energies will be distorted.")

    # Polarised (embedded) Hirshfeld charges of the ligand
    qm_charges = mol.atom_charges().astype(float) - dft_n["hirsh"].populations([dm_qmmm])[0]

    print(f"\n  ── QM/MM Energy Breakdown (kcal/mol) ───────────")
    print(f"  E_QM embedded  : {E_qmmm*HARTREE_TO_KCAL:>14.3f}  (absolute)")
    print(f"  E_QM gas-phase : {E_gas*HARTREE_TO_KCAL:>14.3f}  (absolute)")
    print(f"  E_elec (frozen): {E_elec_cl:>14.3f}")
    print(f"  E_pol          : {E_pol:>14.3f}")
    print(f"  E_int (QM emb) : {E_int_elec:>14.3f}  = E_elec + E_pol")
    print(f"  E_LJ (vdW)     : {E_LJ:>14.3f}")
    print(f"  E_total QM/MM  : {E_total:>14.3f}  (interaction energy, not ΔG_bind)")

    # Per-residue decomposition (sums exactly to E_elec_cl and E_LJ)
    per_res = {}
    for j in range(len(mm_charges)):
        key = f"{pocket['mm_resnames'][j]}{pocket['mm_resids'][j]}"
        if pocket["mm_chains"][j]:
            key = f"{pocket['mm_chains'][j]}:{key}"
        r = per_res.setdefault(key, {"resname": pocket["mm_resnames"][j],
                                     "resid": pocket["mm_resids"][j],
                                     "E_elec": 0., "E_LJ": 0.})
        r["E_elec"] += e_elec_atom[j]; r["E_LJ"] += lj_atom[j]

    df = pd.DataFrame([
        {"Residue":k,"Resname":v["resname"],"Resid":v["resid"],
         "E_elec":round(v["E_elec"],3),"E_LJ":round(v["E_LJ"],3),
         "E_total":round(v["E_elec"]+v["E_LJ"],3)}
        for k,v in per_res.items()
    ]).sort_values("E_total")

    print(f"\n  Top binding residues:")
    print(f"  {'Residue':<14} {'E_elec':>10} {'E_LJ':>9} "
          f"{'E_total':>10}  kcal/mol")
    print(f"  {'─'*46}")
    for _,r in df.head(5).iterrows():
        print(f"  {r['Residue']:<14} {r['E_elec']:>10.3f} "
              f"{r['E_LJ']:>9.3f} {r['E_total']:>10.3f}")

    return {
        "E_QM_embedded"   : round(E_qmmm*HARTREE_TO_KCAL, 3),
        "E_QM_gas"        : round(E_gas*HARTREE_TO_KCAL, 3),
        "E_int_elec"      : round(E_int_elec, 3),
        "E_elec_classical": round(E_elec_cl, 3),
        "E_pol"           : round(E_pol, 3),
        "E_LJ"            : round(E_LJ, 3),
        "E_total"         : round(E_total, 3),
        "converged"       : bool(conv_e and conv_g),
        "min_heavy_contact_A": round(d_min, 2),
        "note"            : ("E_total = E_int_elec + E_LJ: gas-phase QM/MM interaction energy "
                             "at fixed geometry (no desolvation/entropy), not ΔG_bind"),
        "qm_charges"      : qm_charges,
        "per_residue"     : df,
        "top5"            : df.head(5).to_dict("records"),
        "mf_qmmm"         : mf_qmmm,
        "dm_qmmm"         : dm_qmmm,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 7 : H-Bond Detection (D–H···A geometry)
# ══════════════════════════════════════════════════════════════════
def detect_hbonds(lig, pocket, qmmm, cutoff=3.5, angle_min=120.0, ha_max=2.7):
    print(f"\n{'═'*66}")
    print(f"  STEP 7  |  Protein–Ligand H-Bond Analysis")
    print(f"{'─'*66}")
    print(f"  Criteria: D···A ≤ {cutoff} Å, H···A ≤ {ha_max} Å, ∠D–H···A ≥ {angle_min}°")

    lc, elems, nb = lig["coords"], lig["elems"], lig["nbrs"]
    qm_q  = qmmm["qm_charges"]
    pc, pq, pn = pocket["mm_coords"], pocket["mm_charges"], pocket["mm_names"]
    pe, pres, pat = pocket["mm_elems"], pocket["mm_resnames"], pocket["mm_atomnames"]
    rd = lig["rdkit_mol"]

    # ligand donors (heavy, H) and acceptors
    l_don = [(nb[h][0], h) for h, e in enumerate(elems)
             if e == "H" and nb[h] and elems[nb[h][0]] in ("N", "O")]
    l_acc = []
    for i, e in enumerate(elems):
        if e == "O":
            l_acc.append(i)
        elif e == "N":
            nH = sum(1 for j in nb[i] if elems[j] == "H")
            if rd is not None:
                a = rd.GetAtomWithIdx(i)
                # N conjugated to C=O / C=S (amide, urea, ...) or to S(=O)(=O)
                # (sulfonamide) has its lone pair delocalised: not an acceptor
                amide = any(
                    (n.GetSymbol() == "C" and any(
                        bd.GetBondType() == Chem.BondType.DOUBLE and
                        bd.GetOtherAtom(n).GetSymbol() in ("O", "S") for bd in n.GetBonds()))
                    or (n.GetSymbol() == "S" and sum(
                        1 for bd in n.GetBonds()
                        if bd.GetBondType() == Chem.BondType.DOUBLE and
                        bd.GetOtherAtom(n).GetSymbol() == "O") >= 2)
                    for n in a.GetNeighbors())
                heavy_deg = a.GetDegree() - nH
                sp3 = a.GetHybridization() == Chem.HybridizationType.SP3
                # sp3 amines (primary/secondary/tertiary), aromatic N without H
                # (pyridine-like) and sp/sp2 N with a free lone pair (nitrile, imine)
                if a.GetFormalCharge() <= 0 and not amide and (
                        (sp3 and heavy_deg <= 3) or
                        (a.GetIsAromatic() and nH == 0 and heavy_deg <= 2) or
                        (not sp3 and not a.GetIsAromatic() and nH == 0 and heavy_deg <= 2)):
                    l_acc.append(i)
            elif nH == 0 and len(nb[i]) <= 2:
                l_acc.append(i)

    # protein donors: polar H within 1.25 Å of N/O; acceptors: all O, unprotonated His N
    pol = [j for j, e in enumerate(pe) if e in ("N", "O")]
    hyd = [j for j, e in enumerate(pe) if e == "H"]
    p_don = []
    if hyd and pol:
        dd = cdist(pc[hyd], pc[pol])
        for ih, j in enumerate(hyd):
            k = int(np.argmin(dd[ih]))
            if dd[ih, k] < 1.25:
                p_don.append((pol[k], j))
    has_h = {d for d, _ in p_don}
    p_acc = [j for j, e in enumerate(pe) if e == "O" or
             (e == "N" and pres[j] in ("HIS", "HID", "HIE") and
              pat[j] in ("ND1", "NE2") and j not in has_h)]

    def angle(D, H, A):
        v1, v2 = D - H, A - H
        c = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    def strength(d):          # Jeffrey (1997), D···A distance
        return "Strong" if d < 2.5 else "Moderate" if d <= 3.2 else "Weak"

    hbonds = []
    for D, H in l_don:
        for A in p_acc:
            dDA = float(np.linalg.norm(lc[D]-pc[A])); dHA = float(np.linalg.norm(lc[H]-pc[A]))
            if dDA <= cutoff and dHA <= ha_max:
                ang = angle(lc[D], lc[H], pc[A])
                if ang >= angle_min:
                    hbonds.append({"Type": "Lig(donor) → Prot(acceptor)",
                                   "Lig_atom": f"{elems[D]}{D}", "Prot_atom": pn[A],
                                   "Distance_Å": round(dDA, 2), "H···A_Å": round(dHA, 2),
                                   "Angle_°": round(ang, 1),
                                   "Lig_charge": round(float(qm_q[D]), 4),
                                   "Prot_charge": round(float(pq[A]), 4),
                                   "Strength": strength(dDA)})
    for D, H in p_don:
        for A in l_acc:
            dDA = float(np.linalg.norm(pc[D]-lc[A])); dHA = float(np.linalg.norm(pc[H]-lc[A]))
            if dDA <= cutoff and dHA <= ha_max:
                ang = angle(pc[D], pc[H], lc[A])
                if ang >= angle_min:
                    hbonds.append({"Type": "Lig(acceptor) ← Prot(donor)",
                                   "Lig_atom": f"{elems[A]}{A}", "Prot_atom": pn[D],
                                   "Distance_Å": round(dDA, 2), "H···A_Å": round(dHA, 2),
                                   "Angle_°": round(ang, 1),
                                   "Lig_charge": round(float(qm_q[A]), 4),
                                   "Prot_charge": round(float(pq[D]), 4),
                                   "Strength": strength(dDA)})

    hbonds.sort(key=lambda x: x["Distance_Å"])
    seen_pairs = set()
    hbonds_uniq = []
    for h in hbonds:
        pair = (h["Lig_atom"], h["Prot_atom"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            hbonds_uniq.append(h)
    hbonds = hbonds_uniq

    print(f"  H-bonds detected: {len(hbonds)}")
    for h in hbonds:
        print(f"  [{h['Strength']:<8}] {h['Type']:<30} "
              f"{h['Lig_atom']:<8} ↔ {h['Prot_atom']:<20} "
              f"D···A={h['Distance_Å']} Å  ∠{h['Angle_°']}°")
    return hbonds


# ══════════════════════════════════════════════════════════════════
#  STEP 8 : Mechanism Interpretation (geometry + descriptors)
# ══════════════════════════════════════════════════════════════════
def interpret_mechanism(desc, mep, qmmm, fukui, hbonds, lig, pocket):
    print(f"\n{'═'*66}")
    print(f"  STEP 8  |  Inhibition Mechanism Interpretation")
    print(f"{'─'*66}")

    eta   = desc.get("Chemical Hardness η (eV)")
    S     = desc.get("Global Softness S (eV⁻¹)")
    omega = desc.get("Electrophilicity ω (eV)")
    gap   = desc.get("HOMO-LUMO gap (eV)")
    logp  = desc.get("LogP")
    tpsa  = desc.get("TPSA (Å²)")
    qed   = desc.get("QED")
    n_hb  = len(hbonds)
    top_e = fukui["top_electrophilic_atom"]
    top_n = fukui["top_nucleophilic_atom"]
    el, lc = lig["elems"], lig["coords"]
    pc = pocket["mm_coords"]
    pres, pat = pocket["mm_resnames"], pocket["mm_atomnames"]
    pchain, pid = pocket["mm_chains"], pocket["mm_resids"]
    lab = lambda j: f"{pchain[j]+':' if pchain[j] else ''}{pres[j]}{pid[j]}:{pat[j]}"

    mech = {}

    # Covalent bond present? (ligand heavy atom ≤ 2.1 Å from protein heavy atom)
    heavy_l = [i for i, e in enumerate(el) if e != "H"]
    heavy_p = [j for j, e in enumerate(pocket["mm_elems"])
               if e != "H" and pres[j] not in METALS]
    cov = None
    if heavy_l and heavy_p:
        d = cdist(lc[heavy_l], pc[heavy_p])
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] <= 2.1:
            cov = (heavy_l[i], heavy_p[j], float(d[i, j]))

    # Warhead motifs + nearest nucleophile
    warheads = []
    rd = lig["rdkit_mol"]
    nuc_idx = [j for j in range(len(pres)) if (pres[j], pat[j]) in NUCLEOPHILES]
    if rd is not None:
        for name, sma in WARHEADS:
            for match in rd.GetSubstructMatches(Chem.MolFromSmarts(sma)):
                a = match[0]
                txt = f"{name} at {el[a]}{a} (f+={fukui['f_plus'][a]:.3f})"
                if nuc_idx:
                    dn = np.linalg.norm(pc[nuc_idx] - lc[a], axis=1)
                    k = int(np.argmin(dn))
                    txt += f", nearest nucleophile {lab(nuc_idx[k])} {dn[k]:.2f} Å"
                    warheads.append((txt, float(dn[k])))
                else:
                    warheads.append((txt, 99.0))

    if cov:
        mech["Binding Mode"] = (f"Covalent adduct ({el[cov[0]]}{cov[0]}–{lab(cov[1])} "
                                f"{cov[2]:.2f} Å)")
        warn("Ligand appears covalently bonded to the protein: QM/MM without link atoms "
             "treats it as a separate molecule — energies are not meaningful for this complex.")
    elif any(dw <= 4.5 for _, dw in warheads):
        mech["Binding Mode"] = "Non-covalent pose; warhead positioned for covalent attack (≤4.5 Å)"
    elif warheads:
        mech["Binding Mode"] = "Non-covalent (warhead present but no nucleophile within 4.5 Å)"
    else:
        mech["Binding Mode"] = "Non-covalent"

    mech["Reactivity"] = (f"η = {eta:.2f} eV, S = {S:.3f} eV⁻¹; most electrophilic atom "
                          f"{el[top_e]}{top_e} (f+={fukui['f_plus'][top_e]:.3f}), most "
                          f"nucleophilic atom {el[top_n]}{top_n} (f-={fukui['f_minus'][top_n]:.3f})")

    e_el = abs(qmmm["E_int_elec"]); e_lj = abs(qmmm["E_LJ"])
    share = 100*e_el/(e_el+e_lj) if (e_el+e_lj) > 0 else 0
    kind = ("Electrostatic/polarization-dominated" if share >= 60 else
            "vdW/hydrophobic-dominated" if share <= 40 else "Mixed electrostatic + vdW")
    mech["Dominant Interaction"] = (f"{kind} ({share:.0f}% elec / {100-share:.0f}% LJ by magnitude); "
                                    f"{n_hb} H-bond(s)")

    # Domingo et al. 2002 global electrophilicity scale (ω = μ²/2η, η = εL − εH)
    if   omega > 1.5 : cls = "Strong electrophile"
    elif omega >= 0.8: cls = "Moderate electrophile"
    else             : cls = "Marginal electrophile"
    q_lig = desc.get("Net charge", 0) or 0
    if q_lig and desc.get("Solvent model", "gas phase") == "gas phase":
        mech["Electrophilicity Class"] = (f"Not applicable: ionic ligand (charge {q_lig:+d}) in gas "
                                          f"phase (ω={omega:.2f} eV); use SOLVENT='water'")
    elif q_lig:
        mech["Electrophilicity Class"] = (f"{cls} (ω={omega:.2f} eV, PCM; ionic ligand — Domingo "
                                          f"scale derived for neutral molecules)")
    else:
        mech["Electrophilicity Class"] = f"{cls} (ω={omega:.2f} eV; Domingo 2002 scale)"

    metal_c = []
    for m in pocket["metals"]:
        for i, e in enumerate(el):
            if e in ("N", "O", "S"):
                dm_ = float(np.linalg.norm(lc[i] - m["xyz"]))
                if dm_ <= 2.8:
                    metal_c.append(f"{e}{i}–{m['resname']}{m['resid']} {dm_:.2f} Å")
    mech["Metal Chelation"] = ("Metal coordination: " + "; ".join(metal_c)) if metal_c \
                              else "No metal coordination (≤2.8 Å)"

    mech["Covalent Potential"] = ("; ".join(t for t, _ in warheads) if warheads
                                  else "No known covalent warhead motif")

    mech["Selectivity"] = (f"Not assessable from a single complex (gap={gap:.2f} eV is a "
                           f"kinetic-stability indicator only)")

    if desc.get("MW (Da)") is not None:
        viol = sum([desc["MW (Da)"] > 500, logp > 5, desc["HBD"] > 5, desc["HBA"] > 10])
        mech["Lipinski Rule of 5"] = "✓ Pass" if viol <= 1 else f"✗ Fail ({viol} violations)"
        if viol == 1:
            mech["Lipinski Rule of 5"] += " (1 violation)"
        veber = desc["RotBonds"] <= 10 and tpsa <= 140
        mech["Oral Bioavailability"] = (f"{'Good' if veber else 'Poor'} by Veber rules "
                                        f"(RotB={desc['RotBonds']}, TPSA={tpsa} Å²)")
        mech["Drug-likeness QED"] = (f"High QED={qed:.3f}"     if qed>0.6
                                     else f"Moderate QED={qed:.3f}" if qed>0.4
                                     else f"Low QED={qed:.3f}")
    else:
        mech["Lipinski Rule of 5"] = mech["Oral Bioavailability"] = \
            mech["Drug-likeness QED"] = "N/A (structure not perceived)"

    print(f"\n  ── Mechanism Summary ───────────────────────────")
    for k,v in mech.items():
        print(f"  {k:<24}: {v}")
    return mech


# ══════════════════════════════════════════════════════════════════
#  STEP 9 : Figures (2 × 4-panel, 600 DPI, white background)
# ══════════════════════════════════════════════════════════════════
def visualize(lig, mep_data, qmmm, desc, fukui,
              hbonds, pocket, mechanism, out_dir, dpi=600):
    print(f"\n{'═'*66}")
    print(f"  STEP 9  |  Generating {dpi} DPI Figures")
    print(f"{'─'*66}")

    BG  = "#FFFFFF"; PAN = "#F4F6F9"; TXT = "#1a1a2e"
    SUB = "#4a4a6a"; CYN = "#1565C0"; RED = "#C62828"
    GRN = "#2E7D32"; YLW = "#F57F17"; CMAP = "RdBu_r"

    plt.rcParams.update({
        "font.family"    : "DejaVu Sans",
        "font.size"      : 11,
        "axes.linewidth" : 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    def sa(ax, title, sub=""):
        ax.set_facecolor(PAN)
        for sp in ax.spines.values():
            sp.set_color("#cccccc"); sp.set_linewidth(0.8)
        ax.tick_params(colors=TXT, labelsize=10, length=4, width=0.8)
        ax.set_title(f"{title}\n{sub}" if sub else title,
                     color=TXT, fontsize=12, fontweight="bold", pad=10)
        ax.xaxis.label.set_color(SUB); ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_color(SUB); ax.yaxis.label.set_fontsize(10)
        ax.grid(True,color="#e2e2e2",lw=0.5,ls="--",alpha=0.8)
        ax.set_axisbelow(True)

    def add_leg(ax, **kw):
        leg = ax.legend(fontsize=9, framealpha=0.95,
                        labelcolor=TXT, **kw)
        leg.get_frame().set_facecolor(PAN)
        leg.get_frame().set_edgecolor("#cccccc")

    coords = lig["coords"]
    elems  = lig["elems"]
    fp     = fukui["f_plus"]
    fm     = fukui["f_minus"]
    mc     = qmmm["qm_charges"]
    bonds  = [(i, j) for i in range(len(elems)) for j in lig["nbrs"][i] if j > i]

    # project onto the ligand's principal plane (clearer than raw X–Y)
    c0 = coords.mean(0); _, _, vt = np.linalg.svd(coords - c0)
    proj  = lambda X: (np.asarray(X) - c0) @ vt[:2].T
    depth = lambda X: (np.asarray(X) - c0) @ vt[2]
    P = proj(coords)

    # ── Figure 1 : Ligand Analysis ────────────────────────────────
    fig1, ax1 = plt.subplots(2,2,figsize=(20,18))
    fig1.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.40,wspace=0.36,
                        left=0.07,right=0.97,top=0.92,bottom=0.07)

    # A: Charge Map
    axA = ax1[0,0]
    lim = max(float(np.abs(mc).max()), 1e-3)
    scA = axA.scatter(P[:,0],P[:,1],c=mc,cmap=CMAP,vmin=-lim,vmax=lim,
                      s=[90 if e=="H" else 300 for e in elems],
                      edgecolors="#555",lw=0.8,zorder=3)
    for i,j in bonds:
        axA.plot(P[[i,j],0],P[[i,j],1],color="#666",lw=1.5,zorder=2)
    for i,(el,pos,q) in enumerate(zip(elems,P,mc)):
        if el == "H":
            continue
        axA.annotate(f"{el}{i}\n{q:+.3f}",
                     (pos[0],pos[1]),fontsize=8,color=TXT,
                     ha="center",va="bottom",fontweight="bold",
                     xytext=(0,8),textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.15",
                               fc="white",ec="#ddd",alpha=0.85,lw=0.5))
    cb0=plt.colorbar(scA,ax=axA,fraction=0.046,pad=0.03)
    cb0.set_label("Hirshfeld Charge in pocket (e)",color=SUB,fontsize=10)
    cb0.ax.yaxis.set_tick_params(color=TXT,labelsize=9)
    plt.setp(cb0.ax.yaxis.get_ticklabels(),color=TXT)
    cb0.outline.set_edgecolor("#cccccc")
    axA.set_aspect("equal"); axA.set_xlabel("PC1 (Å)"); axA.set_ylabel("PC2 (Å)")
    sa(axA,"A.  DFT Hirshfeld Charge Map (QM/MM-polarised)",
       f"B3LYP/{lig['basis']} | SCF = {desc.get('SCF energy (Ha)',0):.4f} Ha")

    # B: MEP on molecular-surface shell
    axB = ax1[0,1]
    G,V = mep_data["grid"], mep_data["mep"]*HARTREE_TO_KCAL
    o   = np.argsort(depth(G)); Gp = proj(G)[o]; Vo = V[o]
    vl  = np.percentile(np.abs(V),97)
    scB=axB.scatter(Gp[:,0],Gp[:,1],c=Vo,cmap=CMAP,s=8,alpha=0.8,vmin=-vl,vmax=vl)
    for i,j in bonds:
        axB.plot(P[[i,j],0],P[[i,j],1],color="#222",lw=1.2,zorder=5)
    Pp = proj(pocket["mm_coords"])
    axB.scatter(Pp[:,0],Pp[:,1],c=GRN,s=20,marker="x",
                lw=1.2,alpha=0.35,zorder=3,label="MM pocket atoms")
    axB.set_xlim(Gp[:,0].min()-1,Gp[:,0].max()+1); axB.set_ylim(Gp[:,1].min()-1,Gp[:,1].max()+1)
    cb1=plt.colorbar(scB,ax=axB,fraction=0.046,pad=0.03)
    cb1.set_label("MEP (kcal/mol)",color=SUB,fontsize=10)
    cb1.ax.yaxis.set_tick_params(color=TXT,labelsize=9)
    plt.setp(cb1.ax.yaxis.get_ticklabels(),color=TXT)
    cb1.outline.set_edgecolor("#cccccc")
    add_leg(axB)
    axB.set_aspect("equal"); axB.set_xlabel("PC1 (Å)"); axB.set_ylabel("PC2 (Å)")
    sa(axB,"B.  Real DFT MEP (1.0–1.4 × vdW surface)",
       f"Min={mep_data['MEP_min_kcal']:.1f} (near {mep_data['nuc_site_atom']})  "
       f"Max={mep_data['MEP_max_kcal']:.1f} (near {mep_data['elec_site_atom']}) kcal/mol")

    # C: Fukui (heavy atoms)
    axC = ax1[1,0]
    hv  = [i for i,e in enumerate(elems) if e != "H"]
    idx = np.arange(len(hv))
    axC.bar(idx-0.22,fp[hv],0.40,color=RED,alpha=0.85,
            label="f⁺ (site for nucleophilic attack, N+1)",edgecolor="none")
    axC.bar(idx+0.22,fm[hv],0.40,color=CYN,alpha=0.85,
            label="f⁻ (site for electrophilic attack, N-1)",edgecolor="none")
    ti=fukui["top_electrophilic_atom"]; ni=fukui["top_nucleophilic_atom"]
    if ti in hv:
        k=hv.index(ti)
        axC.annotate(f"Max f⁺\n{elems[ti]}{ti}",
                     xy=(k-0.22,fp[ti]),xytext=(k+1.5,fp[ti]+0.02),
                     fontsize=9,color=RED,fontweight="bold",
                     arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
    if ni in hv:
        k=hv.index(ni)
        axC.annotate(f"Max f⁻\n{elems[ni]}{ni}",
                     xy=(k+0.22,fm[ni]),xytext=(k+1.5,fm[ni]+0.02),
                     fontsize=9,color=CYN,fontweight="bold",
                     arrowprops=dict(arrowstyle="->",color=CYN,lw=1.2))
    axC.set_xlabel("Atom index"); axC.set_ylabel("Fukui index (Hirshfeld)")
    axC.set_xticks(idx)
    axC.set_xticklabels([f"{elems[i]}{i}" for i in hv],
                        rotation=55,fontsize=8,ha="right")
    add_leg(axC)
    sa(axC,"C.  Real Fukui Reactivity Indices",
       "Calculated from N, N+1, N-1 DFT states (heavy atoms)")

    # D: H-bonds
    axD = ax1[1,1]
    if hbonds:
        labs=[f"{h['Lig_atom']} ↔ {h['Prot_atom']}" for h in hbonds]
        dsts=[h["Distance_Å"] for h in hbonds]
        col=[GRN if h["Strength"]!="Weak" else YLW for h in hbonds]
        bars=axD.barh(labs,dsts,color=col,edgecolor="#aaa",lw=0.4,height=0.55)
        for bar,h in zip(bars,hbonds):
            axD.text(bar.get_width()+0.03,
                     bar.get_y()+bar.get_height()/2,
                     f"{h['Distance_Å']:.2f} Å, {h['Angle_°']:.0f}°",va="center",fontsize=9,
                     color=TXT,fontweight="bold")
        axD.axvline(3.5,color=RED,lw=1.2,ls="--",label="3.5 Å cutoff")
        axD.axvline(3.2,color=GRN,lw=1.2,ls=":",label="3.2 Å (moderate/weak)")
        axD.set_xlim(0,4.3); axD.invert_yaxis()
        axD.set_xlabel("Donor–Acceptor Distance (Å)")
        add_leg(axD)
    else:
        axD.text(0.5,0.5,"No H-bonds detected",
                 ha="center",va="center",color=SUB,
                 fontsize=14,transform=axD.transAxes)
    sa(axD,f"D.  Protein–Ligand H-Bonds ({len(hbonds)})",
       "D–H···A: D···A ≤ 3.5 Å, H···A ≤ 2.7 Å, angle ≥ 120°")

    fig1.suptitle(
        f"Figure 1 — Ligand Quantum Analysis  ·  "
        f"HOMO={desc.get('HOMO energy (eV)',0):.4f} eV  ·  "
        f"LUMO={desc.get('LUMO energy (eV)',0):.4f} eV  ·  "
        f"Gap={desc.get('HOMO-LUMO gap (eV)',0):.4f} eV",
        color=TXT,fontsize=14,fontweight="bold",y=0.96)
    p1 = out_dir/"fig1_ligand_quantum.png"
    fig1.savefig(p1,dpi=dpi,bbox_inches="tight",
                 facecolor="white",edgecolor="none")
    plt.close(fig1)
    print(f"  ✓ Figure 1 → {p1.name}")

    # ── Figure 2 : Binding Analysis ───────────────────────────────
    fig2, ax2 = plt.subplots(2,2,figsize=(20,18))
    fig2.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.42,wspace=0.38,
                        left=0.10,right=0.97,top=0.92,bottom=0.07)

    # E: Per-residue
    axE = ax2[0,0]
    df  = qmmm["per_residue"].head(14)
    col = [RED if v<0 else CYN for v in df["E_total"]]
    bars= axE.barh(df["Residue"],df["E_total"],color=col,
                   edgecolor="#aaa",lw=0.4,height=0.65)
    for bar,val in zip(bars,df["E_total"]):
        xp=bar.get_width()
        axE.text(xp+(0.2 if xp>=0 else -0.2),
                 bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}",va="center",
                 ha=("left" if xp>=0 else "right"),
                 fontsize=8,color=TXT)
    axE.axvline(0,color="#444",lw=1.0); axE.invert_yaxis()
    axE.legend(handles=[Patch(color=RED,label="Favorable"),
                         Patch(color=CYN,label="Unfavorable")],
               fontsize=9,framealpha=0.95,labelcolor=TXT
               ).get_frame().set_facecolor(PAN)
    axE.set_xlabel("E_elec + E_LJ (kcal/mol)")
    sa(axE,"E.  Per-Residue QM/MM Interaction",
       f"E_total = {qmmm['E_total']:.2f} kcal/mol (incl. E_pol {qmmm['E_pol']:.2f})")

    # F: Electrostatic vs vdW (grouped bars; stacking mixed signs is misleading)
    axF = ax2[0,1]
    df10= qmmm["per_residue"].head(12)
    y   = np.arange(len(df10))
    axF.barh(y-0.2,df10["E_elec"],0.4,color=RED,alpha=0.85,
             label="E_elec",edgecolor="none")
    axF.barh(y+0.2,df10["E_LJ"],0.4,color=CYN,alpha=0.85,
             label="E_LJ (vdW)",edgecolor="none")
    axF.set_yticks(y); axF.set_yticklabels(df10["Residue"],fontsize=9); axF.invert_yaxis()
    axF.set_xlabel("Energy (kcal/mol)")
    axF.axvline(0,color="#444",lw=1.0)
    add_leg(axF,loc="lower right")
    sa(axF,"F.  Electrostatic vs vdW Decomposition",
       f"E_elec={qmmm['E_elec_classical']:.2f}  E_pol={qmmm['E_pol']:.2f}  "
       f"E_LJ={qmmm['E_LJ']:.2f}  kcal/mol")

    # G: DFT Radar
    axG = ax2[1,0]; axG.remove()
    axG = fig2.add_subplot(2,2,3,polar=True)
    axG.set_facecolor(PAN)
    keys = ["IP = -E(HOMO) (eV)","EA = -E(LUMO) (eV)",
            "Chemical Hardness η (eV)",
            "Electrophilicity ω (eV)","Global Softness S (eV⁻¹)"]
    lbls = ["IP","EA","Hardness η","Electrophilicity ω","Softness S"]
    vals = [abs(desc.get(k) or 0.0) for k in keys]
    vmax = max([v for v in vals if v > 0] or [1.0]) + 1e-9
    vn   = [v/vmax for v in vals]+[vals[0]/vmax]
    N    = len(keys)
    angs = [n/N*2*np.pi for n in range(N)]+[0]
    axG.plot(angs,vn,color=GRN,lw=2.5,zorder=3)
    axG.fill(angs,vn,color=GRN,alpha=0.25,zorder=2)
    axG.scatter(angs[:-1],vn[:-1],color=GRN,s=80,
                zorder=4,edgecolors="white",lw=1.0)
    for ang,vv,lbl,k in zip(angs[:-1],vn[:-1],lbls,keys):
        axG.annotate(f"{desc.get(k,0):.2f}",
                     xy=(ang,vv),xytext=(ang,vv+0.14),
                     ha="center",va="center",fontsize=9,
                     color=TXT,fontweight="bold")
    axG.set_xticks(angs[:-1])
    axG.set_xticklabels(lbls,color=TXT,fontsize=9,fontweight="bold")
    axG.set_yticklabels([]); axG.set_ylim(0,1.35)
    axG.spines["polar"].set_color("#cccccc")
    axG.yaxis.grid(True,color="#ddd",lw=0.6)
    axG.xaxis.grid(True,color="#ddd",lw=0.6)
    axG.set_title(
        f"G.  DFT Reactivity Descriptors (|value| / max)\n"
        f"Gap = {desc.get('HOMO-LUMO gap (eV)',0):.4f} eV  |  "
        f"η = {desc.get('Chemical Hardness η (eV)',0):.4f} eV",
        color=TXT,fontsize=12,fontweight="bold",pad=24)

    # H: HOMO/LUMO diagram
    axH = ax2[1,1]
    HOMO = desc.get("HOMO energy (eV)",0)
    LUMO = desc.get("LUMO energy (eV)",0)
    gap  = desc.get("HOMO-LUMO gap (eV)",0)
    axH.barh(["HOMO","LUMO"],[HOMO,LUMO],
             color=[RED,CYN],edgecolor="#555",height=0.4)
    axH.axvline(0,color="#333",lw=0.8,ls="--")
    axH.annotate("",xy=(LUMO,0.62),xytext=(HOMO,0.62),
                 arrowprops=dict(arrowstyle="<->",color=GRN,lw=2.5))
    axH.text((HOMO+LUMO)/2,0.72,f"Gap = {gap:.4f} eV",
             ha="center",fontsize=12,color=GRN,fontweight="bold")
    for val,lbl in [(HOMO,"HOMO"),(LUMO,"LUMO")]:
        axH.text(val+0.05,["HOMO","LUMO"].index(lbl),
                 f"{val:.4f} eV",va="center",fontsize=11,
                 color=TXT,fontweight="bold")
    axH.set_xlabel("Orbital Energy (eV)")
    axH.set_yticks([0,1])
    axH.set_yticklabels(["HOMO","LUMO"],fontsize=13,fontweight="bold")
    sa(axH,"H.  Real HOMO/LUMO Orbital Energies",
       f"IP={desc.get('IP = -E(HOMO) (eV)',0):.4f} eV  |  "
       f"EA={desc.get('EA = -E(LUMO) (eV)',0):.4f} eV  |  "
       f"χ={desc.get('Electronegativity χ (eV)',0):.4f} eV")

    fig2.suptitle(
        f"Figure 2 — QM/MM Binding Analysis  ·  "
        f"E_int(QM/MM) = {qmmm['E_total']:.2f} kcal/mol  ·  "
        f"H-bonds = {len(hbonds)}  ·  "
        f"Pocket = {pocket['n_pocket_res']} residues",
        color=TXT,fontsize=14,fontweight="bold",y=0.96)
    p2 = out_dir/"fig2_binding_qmmm.png"
    fig2.savefig(p2,dpi=dpi,bbox_inches="tight",
                 facecolor="white",edgecolor="none")
    plt.close(fig2)
    print(f"  ✓ Figure 2 → {p2.name}")
    return str(p1), str(p2)


# ══════════════════════════════════════════════════════════════════
#  STEP 10 : Report
# ══════════════════════════════════════════════════════════════════
def save_report(pdb_path, pocket, desc, mep, qmmm,
                fukui, hbonds, mech, lig, out_dir):
    ts  = datetime.now().isoformat()
    SEP = "═"*68; sep = "─"*68

    data = {
        "timestamp"         : ts,
        "pipeline"          : "Q-MECH v1.0",
        "citation"          : "Dwivedi VD (2026). Q-MECH v1.0. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX",
        "pdb_file"          : str(pdb_path),
        "ligand_resname"    : pocket["lig_resname"],
        "ligand_chain"      : pocket["lig_chain"],
        "ligand_resid"      : pocket["lig_resid"],
        "ligand_net_charge" : pocket["lig_charge"],
        "ligand_charge_source": pocket["lig_charge_src"],
        "DFT_method"        : desc.get("DFT functional","B3LYP"),
        "basis_set"         : lig["basis"],
        "MM_charge_model"   : pocket["charge_model"],
        "MM_total_charge"   : round(float(pocket["mm_charges"].sum()), 3),
        "pocket_residues"   : pocket["pocket_residues"],
        "n_pocket_atoms"    : pocket["n_pocket_atoms"],
        "DFT_descriptors"   : desc,
        "MEP_features"      : {k:v for k,v in mep.items()
                                if k not in("grid","mep","atom_coords",
                                            "V_nuc","V_elec")},
        "Fukui_indices"     : {
            "method"  : "Hirshfeld-condensed, frozen geometry",
            "f_plus"  : fukui["f_plus"].tolist(),
            "f_minus" : fukui["f_minus"].tolist(),
            "delta_f" : fukui["delta_f"].tolist(),
            "top_electrophilic_atom": int(fukui["top_electrophilic_atom"]),
            "top_nucleophilic_atom" : int(fukui["top_nucleophilic_atom"]),
            "vertical_IP_eV": round(fukui["vertical_IP_eV"], 4),
            "vertical_EA_eV": round(fukui["vertical_EA_eV"], 4),
            "converged": fukui["converged"],
        },
        ("Hirshfeld_charges_pcm" if SOLV["eps"] else "Hirshfeld_charges_gas"):
            [round(float(x),4) for x in fukui.get("q_gas", [])],
        "Hirshfeld_charges_embedded": [round(float(x),4) for x in qmmm["qm_charges"]],
        "QMMM_energies"     : {k:v for k,v in qmmm.items()
                                if k not in("per_residue","top5",
                                            "mf_qmmm","dm_qmmm","qm_charges")},
        "top5_residues"     : qmmm["top5"],
        "per_residue"       : qmmm["per_residue"].to_dict("records"),
        "hydrogen_bonds"    : hbonds,
        "mechanism"         : mech,
        "warnings"          : WARNINGS,
    }

    jp = out_dir/"report.json"
    with open(jp,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False,default=str)
    qmmm["per_residue"].to_csv(out_dir/"per_residue.csv", index=False)

    tp = out_dir/"report.txt"
    with open(tp,"w",encoding="utf-8") as f:
        f.write(f"{SEP}\n  Q-MECH v1.0 — REAL DFT + QM/MM INHIBITION REPORT\n{SEP}\n")
        f.write(f"  Date    : {ts}\n")
        f.write(f"  PDB     : {pdb_path}\n")
        f.write(f"  Ligand  : {pocket['lig_resname']} (chain '{pocket['lig_chain'] or '-'}', "
                f"resid {pocket['lig_resid']}) | net charge {pocket['lig_charge']:+d} "
                f"[{pocket['lig_charge_src']}]\n")
        f.write(f"  SMILES  : {lig['smiles']}\n")
        f.write(f"  Method  : {desc.get('DFT functional','B3LYP')}/"
                f"{lig['basis']} (PySCF real DFT) | {desc.get('Solvent model','gas phase')}\n")
        f.write(f"  MM      : {pocket['charge_model']} | {pocket['n_pocket_atoms']} atoms | "
                f"net {pocket['mm_charges'].sum():+.3f} e\n")
        f.write(f"  Cite    : Dwivedi VD (2026). Q-MECH v1.0. Zenodo.\n"
                f"            https://doi.org/10.5281/zenodo.XXXXXXX\n\n")

        f.write(f"{sep}\n  WARNINGS ({len(WARNINGS)})\n{sep}\n")
        for w in WARNINGS:
            f.write(f"  • {w}\n")
        f.write("\n")

        f.write(f"{sep}\n  BINDING POCKET ({pocket['n_pocket_res']} residues)\n{sep}\n")
        for r in pocket["pocket_residues"]:
            f.write(f"  {r['chain'] or '-'}:{r['resname']}{r['resid']}   "
                    f"(min dist {r['min_dist_A']} Å)\n")
        f.write("\n")

        f.write(f"{sep}\n  REAL DFT ORBITAL ENERGIES ({desc.get('DFT functional')}/{lig['basis']})\n{sep}\n")
        for k in ["Solvent model","SCF converged","SCF energy (Ha)","HOMO energy (eV)","LUMO energy (eV)",
                  "HOMO-LUMO gap (eV)","IP = -E(HOMO) (eV)",
                  "EA = -E(LUMO) (eV)","Electronegativity χ (eV)",
                  "Chemical Hardness η (eV)","Electrophilicity ω (eV)",
                  "Global Softness S (eV⁻¹)","ΔN_max"]:
            f.write(f"  {k:<38}: {desc.get(k,'')}\n")
        f.write(f"  {'Vertical IP (ΔSCF) (eV)':<38}: {fukui['vertical_IP_eV']:.4f}\n")
        f.write(f"  {'Vertical EA (ΔSCF) (eV)':<38}: {fukui['vertical_EA_eV']:.4f}\n")
        f.write("  Conventions: I=-E(HOMO), A=-E(LUMO), χ=(I+A)/2, η=I-A, S=1/η, "
                "ω=χ²/2η, ΔNmax=χ/η (Parr 1999)\n\n")

        f.write(f"{sep}\n  REAL FUKUI INDICES (N/N+1/N-1 DFT, Hirshfeld)\n{sep}\n")
        f.write(f"  {'Atom':<8} {'f+':<12} {'f-':<12} {'Δf':<12}\n")
        f.write(f"  {'─'*46}\n")
        for i,(fp,fm,df0) in enumerate(zip(fukui["f_plus"],
                                            fukui["f_minus"],
                                            fukui["delta_f"])):
            f.write(f"  {lig['elems'][i]}{i:<6} {fp:>+10.4f}  "
                    f"{fm:>+10.4f}  {df0:>+10.4f}\n")
        f.write("\n")

        f.write(f"{sep}\n  QM/MM ENERGIES (PySCF embedded; kcal/mol)\n{sep}\n")
        f.write(f"  E_elec (frozen density) : {qmmm['E_elec_classical']:>12.3f}\n")
        f.write(f"  E_pol (polarization)    : {qmmm['E_pol']:>12.3f}\n")
        f.write(f"  E_int = E_emb − E_gas   : {qmmm['E_int_elec']:>12.3f}\n")
        f.write(f"  E_LJ (vdW)              : {qmmm['E_LJ']:>12.3f}\n")
        f.write(f"  E_total QM/MM           : {qmmm['E_total']:>12.3f}\n")
        f.write(f"  ({qmmm['note']})\n\n")

        f.write(f"{sep}\n  TOP BINDING RESIDUES\n{sep}\n")
        f.write(f"  {'Residue':<14}{'E_elec':>10}{'E_LJ':>9}{'E_total':>10}  kcal/mol\n")
        for r in qmmm["top5"]:
            f.write(f"  {r['Residue']:<14}{r['E_elec']:>10.3f}"
                    f"{r['E_LJ']:>9.3f}{r['E_total']:>10.3f}\n")
        f.write("\n")

        f.write(f"{sep}\n  H-BONDS ({len(hbonds)})\n{sep}\n")
        for h in hbonds:
            f.write(f"  [{h['Strength']:<8}] {h['Type']:<30} "
                    f"Lig:{h['Lig_atom']:<8} Prot:{h['Prot_atom']:<20} "
                    f"D···A={h['Distance_Å']} Å  H···A={h['H···A_Å']} Å  ∠{h['Angle_°']}°\n")
        f.write("\n")

        f.write(f"{sep}\n  MECHANISM\n{sep}\n")
        for k,v in mech.items():
            f.write(f"  {k:<24}: {v}\n")
        f.write(f"\n{SEP}\n")

    print(f"  ✓ Reports → {tp.name}, {jp.name}, per_residue.csv")
    return tp


def _system_memory_gb():
    """(total_GB, available_GB) — psutil if installed, else /proc/meminfo, else sysconf."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total/1024**3, vm.available/1024**3
    except Exception:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = float(v.split()[0]) / 1024**2          # kB → GB
        return info["MemTotal"], info.get("MemAvailable", info["MemTotal"]*0.7)
    except Exception:
        pass
    try:
        tot = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        return tot, tot*0.7
    except Exception:
        return None, None


def resolve_memory(max_memory="AUTO", min_gb=32, use_fraction=0.85):
    """PySCF memory in MB. AUTO = use_fraction × currently available RAM.
    Stops if the machine has less than min_gb of RAM."""
    total, avail = _system_memory_gb()
    if total is None:
        warn("Could not detect system RAM — using 32000 MB.")
        return 32000 if str(max_memory).upper() == "AUTO" else int(max_memory)
    print(f"  RAM       : {total:.1f} GB total | {avail:.1f} GB available")
    if total < min_gb * 0.97:            # 0.97: firmware/kernel reserve a little RAM
        raise RuntimeError(
            f"Q-MECH needs at least {min_gb} GB RAM; this machine has {total:.1f} GB. "
            f"(To run anyway on a small ligand, lower MIN_MEMORY_GB in the settings.)")
    if avail < min_gb * 0.97:
        warn(f"Only {avail:.1f} GB RAM is free right now (other programs are using memory); "
             "the run may be slower or fail. Close other jobs if possible.")
    if str(max_memory).upper() == "AUTO":
        mb = int(avail * use_fraction * 1024)
        print(f"  PySCF memory: {mb} MB (AUTO = {int(use_fraction*100)}% of available RAM)")
        return mb
    mb = int(max_memory)
    if mb > avail * 1024:
        warn(f"MAX_MEMORY = {mb} MB is more than the free RAM ({avail*1024:.0f} MB) — "
             "the run may be killed by the system.")
    return mb


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
def run_pipeline(pdb_path, ligand_resname="AUTO",
                 functional="B3LYP", pocket_cutoff=8.0,
                 n_threads=24, mep_grid_pts=40,
                 out_dir=None, ligand_charge=None,
                 basis="def2-SVP", max_memory="AUTO", min_memory_gb=32,
                 ph=7.0, include_waters=False, dpi=600, solvent=None):

    WARNINGS.clear()
    if solvent in (None, "", "none", "gas", "None"):
        SOLV.update(name=None, eps=None)
    else:
        eps = SOLVENTS.get(str(solvent).lower())
        if eps is None:
            try:
                eps = float(solvent); solvent = f"eps={eps}"
            except ValueError:
                raise ValueError(f"Unknown solvent '{solvent}'. Use one of {list(SOLVENTS)} "
                                 "or a dielectric constant number.")
        SOLV.update(name=str(solvent).lower(), eps=eps)
    t_start = time.time()
    print(f"\n{'╔'+'═'*66+'╗'}")
    print(f"║{'Q-MECH v1.0 — Quantum Mechanistic Inhibition Pipeline':^66}║")
    print(f"{'╚'+'═'*66+'╝'}")
    print(f"  PDB       : {pdb_path}")
    print(f"  Functional: {functional}/{basis}  |  DFT solvent: "
          f"{'IEF-PCM ' + SOLV['name'] if SOLV['eps'] else 'gas phase'}")
    print(f"  Threads   : {n_threads}  |  {'GPU' if GPU_OK else 'CPU'}")
    print(f"  Start     : {datetime.now().strftime('%H:%M:%S')}")

    if out_dir is None:
        out_dir = Path("qmech_results")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(n_threads)
    max_memory = resolve_memory(max_memory, min_memory_gb)

    pocket  = extract_pocket_from_pdb(pdb_path, ligand_resname, pocket_cutoff,
                                      ligand_charge, ph, include_waters)
    lig     = build_ligand_mol(pocket, basis, max_memory)
    if pocket["lig_charge"] and not SOLV["eps"]:
        warn(f"Ionic ligand (charge {pocket['lig_charge']:+d}) in gas phase: orbital energies and "
             "ω are shifted by the net charge and not comparable with neutral ligands. "
             "Set SOLVENT = 'water' for comparable descriptors.")
    dft_n   = run_dft_neutral(lig, functional, n_threads)
    fukui   = run_fukui(lig, dft_n, functional)
    fukui["q_gas"] = dft_n["hirshfeld_charges"]
    mep_dat = compute_real_mep(lig, dft_n, mep_grid_pts)
    qmmm    = run_qmmm(lig, pocket, dft_n, functional)
    hbonds  = detect_hbonds(lig, pocket, qmmm)
    mech    = interpret_mechanism(dft_n["descriptors"], mep_dat,
                                   qmmm, fukui, hbonds, lig, pocket)
    visualize(lig, mep_dat, qmmm, dft_n["descriptors"],
              fukui, hbonds, pocket, mech, out_dir, dpi)
    save_report(pdb_path, pocket, dft_n["descriptors"],
                mep_dat, qmmm, fukui, hbonds, mech, lig, out_dir)

    ok = dft_n["converged"] and fukui["converged"] and qmmm["converged"]
    total = time.time()-t_start
    print(f"\n{'╔'+'═'*66+'╗'}")
    print(f"║{'PIPELINE COMPLETE' + ('' if ok else ' (WITH SCF WARNINGS)'):^66}║")
    print(f"║  Total time : {total/60:.1f} min{'':<46}║")
    print(f"║  Warnings   : {len(WARNINGS):<52}║")
    print(f"║  Output     : {str(out_dir.resolve())[-52:]:<52}║")
    print(f"{'╚'+'═'*66+'╝'}\n")

    return {"pocket":pocket,"ligand":lig,"dft":dft_n,
            "fukui":fukui,"mep":mep_dat,"qmmm":qmmm,
            "hbonds":hbonds,"mechanism":mech,"ok":ok}


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    import sys

    # Command-line usage:
    # python3 qmech_v1.0.py Complex_1.pdb LIG Complex_1_results [charge]
    if len(sys.argv) >= 2:
        PDB_FILE       = sys.argv[1]
        LIGAND_RESNAME = sys.argv[2] if len(sys.argv) >= 3 else "AUTO"
        OUT_DIR        = sys.argv[3] if len(sys.argv) >= 4 else "qmech_results"
        CHG            = sys.argv[4] if len(sys.argv) >= 5 else "AUTO"
    else:
        # ┌────────────────────────────────────────────────────┐
        # │  Edit these settings for your complex:             │
        # └────────────────────────────────────────────────────┘
        PDB_FILE       = "Complex_1.pdb"   # ← your PDB file
        LIGAND_RESNAME = "AUTO"            # ← or "LIG", "UNK", "UNK:B:900"
        OUT_DIR        = "Complex_1_results"
        CHG            = "AUTO"            # ← ligand net charge ("0", "-1", "+1"); AUTO = from PDB / structure

    LIGAND_CHARGE  = None if str(CHG).upper() == "AUTO" else int(CHG)

    FUNCTIONAL     = "B3LYP"     # B3LYP | PBE | M06-2X | wB97X-D
    BASIS          = "def2-SVP"  # def2-SVPD recommended for Fukui f+ (anion)
    POCKET_CUTOFF  = 8.0         # Å, whole residues in MM region
    N_THREADS      = 24          # CPU threads
    MEP_GRID       = 40          # 40=standard, 60=fine
    MAX_MEMORY     = "AUTO"      # "AUTO" = 85% of free RAM, or a number in MB
    MIN_MEMORY_GB  = 32          # minimum RAM required to run
    PH             = 7.0         # used only if protein must be re-protonated
    INCLUDE_WATERS = False       # True: protonated crystal waters as TIP3P charges
    DPI            = 600
    SOLVENT        = "water"     # DFT/Fukui/MEP in IEF-PCM; None = gas phase. QM/MM stays gas-phase

    try:
        res = run_pipeline(
            pdb_path       = PDB_FILE,
            ligand_resname = LIGAND_RESNAME,
            functional     = FUNCTIONAL,
            pocket_cutoff  = POCKET_CUTOFF,
            n_threads      = N_THREADS,
            mep_grid_pts   = MEP_GRID,
            out_dir        = OUT_DIR,
            ligand_charge  = LIGAND_CHARGE,
            basis          = BASIS,
            max_memory     = MAX_MEMORY,
            min_memory_gb  = MIN_MEMORY_GB,
            ph             = PH,
            include_waters = INCLUDE_WATERS,
            dpi            = DPI,
            solvent        = SOLVENT,
        )
        sys.exit(0 if res["ok"] else 2)
    except (ValueError, RuntimeError) as e:
        print(f"\n  ✗ ERROR: {e}\n")
        sys.exit(1)
