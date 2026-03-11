"""
Per-residue energy scoring for protein structures using openmm.

This module calculates per-residue energy contributions from an energy-minimized
protein structure using the Amber ff14SB force field. The input structure MUST
be energy minimized before scoring for meaningful results.

Functions:
    score_residues: Main entry point - loads PDB, calculates per-residue energies, writes TSV
    prepare_structure: Fix missing atoms using pdbfixer
    minimize_structure: Energy minimize using openmm LocalEnergyMinimizer
"""

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    print("Error: openmm not installed. Run: conda install -c conda-forge openmm pdbfixer numpy", file=sys.stderr)
    sys.exit(1)

try:
    from pdbfixer import PDBFixer
except ImportError:
    PDBFixer = None


# Standard amino acid residue names
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Common protonation variants
    "HIE", "HID", "HIP", "CYX", "ASH", "GLH", "LYN",
}

# Backbone atom names (to exclude when calculating side chain energies)
BACKBONE_ATOMS = {
    "N", "CA", "C", "O", "OXT",  # Heavy backbone atoms
    "H", "H1", "H2", "H3", "HA", "HA2", "HA3",  # Backbone hydrogens
}


def prepare_structure(
    input_pdb: str,
    output_pdb: Optional[str] = None,
    add_hydrogens: bool = True,
    ph: float = 7.0,
    remove_heterogens: bool = True,
    keep_water: bool = False,
    add_missing_residues: bool = False,
) -> str:
    """
    Fix missing atoms and optionally add hydrogens using PDBFixer.

    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path for output PDB file (default: input_pdb with '_fixed' suffix)
        add_hydrogens: Whether to add missing hydrogens
        ph: pH for protonation state assignment
        remove_heterogens: Remove non-protein molecules (ligands, ions, etc.)
        keep_water: Keep water molecules (only applies if remove_heterogens=True)
        add_missing_residues: Add missing residues from gaps (WARNING: this renumbers
            residues starting from 1, which will change PDB residue numbering)

    Returns:
        Path to the fixed PDB file

    Raises:
        ImportError: If pdbfixer is not installed
        FileNotFoundError: If input PDB file does not exist
    """
    if PDBFixer is None:
        raise ImportError("pdbfixer is required. Install with: conda install -c conda-forge pdbfixer")

    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError("Input PDB file not found: {}".format(input_pdb))

    if output_pdb is None:
        output_pdb = str(input_path.with_suffix("")) + "_fixed.pdb"

    print("Fixing structure: {}".format(input_pdb))

    fixer = PDBFixer(filename=input_pdb)

    # Remove heterogens (ligands, ions, etc.) if requested
    if remove_heterogens:
        print("Removing heterogens (keep_water={})...".format(keep_water))
        fixer.removeHeterogens(keepWater=keep_water)

    # Find missing residues (required for findMissingAtoms to work)
    fixer.findMissingResidues()

    # Clear missing residues if we don't want to add them (preserves original numbering)
    if not add_missing_residues:
        fixer.missingResidues = {}
    else:
        print("Adding missing residues (WARNING: this renumbers residues)...")

    # Find and add missing atoms within existing residues
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Save residue IDs BEFORE adding hydrogens (which can renumber them)
    saved_residue_ids = {}  # (chain_id, residue_index_in_chain) -> residue_id
    for chain in fixer.topology.chains():
        for idx, residue in enumerate(chain.residues()):
            saved_residue_ids[(chain.id, idx)] = residue.id

    # Add hydrogens at specified pH
    if add_hydrogens:
        fixer.addMissingHydrogens(ph)

    # Restore residue IDs after adding hydrogens (which may renumber them)
    for chain in fixer.topology.chains():
        for idx, residue in enumerate(chain.residues()):
            key = (chain.id, idx)
            if key in saved_residue_ids:
                residue.id = saved_residue_ids[key]

    # Write fixed structure, preserving original residue IDs
    with open(output_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    print("Fixed structure written to: {}".format(output_pdb))
    return output_pdb


def minimize_structure(
    input_pdb: str,
    output_pdb: Optional[str] = None,
    forcefield: str = "amber14-all",
    max_iterations: int = 1000,
    tolerance: float = 10.0,
) -> str:
    """
    Energy minimize a protein structure using openmm.

    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path for output PDB file (default: input_pdb with '_minimized' suffix)
        forcefield: Force field to use ('amber14-all' or 'charmm36')
        max_iterations: Maximum minimization iterations
        tolerance: Energy tolerance in kJ/mol/nm

    Returns:
        Path to the minimized PDB file

    Raises:
        FileNotFoundError: If input PDB file does not exist
        ValueError: If force field is not supported
    """
    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError("Input PDB file not found: {}".format(input_pdb))

    if output_pdb is None:
        output_pdb = str(input_path.with_suffix("")) + "_minimized.pdb"

    print("Loading structure: {}".format(input_pdb))
    pdb = app.PDBFile(input_pdb)

    # Set up force field
    if forcefield == "amber14-all":
        ff = app.ForceField("amber14-all.xml")
    elif forcefield == "charmm36":
        ff = app.ForceField("charmm36.xml")
    else:
        raise ValueError("Unsupported force field: {}. Use 'amber14-all' or 'charmm36'".format(forcefield))

    print("Creating system with {} force field...".format(forcefield))

    # Create system without periodic boundaries (vacuum)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )

    # Create integrator (required even for minimization)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    # Create simulation
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Get initial energy
    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print("Initial energy: {:.2f} kJ/mol".format(initial_energy))

    # Minimize
    print("Minimizing (max {} iterations, tolerance {} kJ/mol/nm)...".format(max_iterations, tolerance))
    simulation.minimizeEnergy(
        maxIterations=max_iterations,
        tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometer,
    )

    # Get final energy
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print("Final energy: {:.2f} kJ/mol".format(final_energy))
    print("Energy change: {:.2f} kJ/mol".format(final_energy - initial_energy))

    # Write minimized structure, preserving original residue IDs
    positions = state.getPositions()
    with open(output_pdb, "w") as f:
        app.PDBFile.writeFile(pdb.topology, positions, f, keepIds=True)

    print("Minimized structure written to: {}".format(output_pdb))
    return output_pdb


def _get_residue_key(residue):
    """Get a unique key for a residue (chain_id, residue_id, residue_name)."""
    chain_id = residue.chain.id if residue.chain.id else "A"
    # Use residue.id (PDB numbering) instead of residue.index (0-based topology index)
    return (chain_id, residue.id, residue.name)


def _build_atom_to_residue_map(topology):
    """Build a mapping from atom index to residue key."""
    atom_to_residue = {}
    for atom in topology.atoms():
        atom_to_residue[atom.index] = _get_residue_key(atom.residue)
    return atom_to_residue


def score_residues(
    input_pdb: str,
    output_tsv: Optional[str] = None,
    forcefield: str = "amber14-all",
    include_waters: bool = False,
    include_ligands: bool = False,
    source_chains: Optional[str] = None,
    target_chains: Optional[str] = None,
    interface_only: bool = True,
    interface_distance: float = 4.0,
) -> str:
    """
    Calculate per-residue energy contributions from a protein structure.

    IMPORTANT: The input structure MUST be energy minimized for meaningful results.
    Use minimize_structure() first if the structure has not been minimized.

    The energy decomposition strategy:
    - Bonded terms (bonds, angles, torsions): Attributed to residue containing the central atom
    - Non-bonded terms (electrostatics, LJ): Split 50/50 between interacting residues

    For interface analysis:
    - Use source_chains to specify which chains to score (e.g., "C" for chain C only)
    - Use target_chains to specify which chains to count interactions with (e.g., "A")
    - With interface_only=True (default), only residues within interface_distance of
      target_chains are included

    Args:
        input_pdb: Path to input PDB file (must be energy minimized)
        output_tsv: Path for output TSV file (default: input_pdb with '_scores.tsv' suffix)
        forcefield: Force field to use ('amber14-all' or 'charmm36')
        include_waters: Include water molecules in scoring
        include_ligands: Include ligand molecules in scoring
        source_chains: Only score residues from these chains (e.g., "C" or "AB").
            If None, score all chains.
        target_chains: Only count non-bonded interactions with these chains (e.g., "A").
            If None, count interactions with all chains.
        interface_only: If True and target_chains is set, only include residues within
            interface_distance of target chain atoms. Default True.
        interface_distance: Distance cutoff (Angstroms) for interface residue detection.
            Default 8.0 A.

    Returns:
        Path to the output TSV file

    Raises:
        FileNotFoundError: If input PDB file does not exist
        ValueError: If force field is not supported
    """
    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError("Input PDB file not found: {}".format(input_pdb))

    if output_tsv is None:
        output_tsv = str(input_path.with_suffix("")) + "_scores.tsv"

    # Convert chain strings to sets for faster lookup
    source_chain_set = set(source_chains) if source_chains else None
    target_chain_set = set(target_chains) if target_chains else None

    print("Loading structure: {}".format(input_pdb))
    pdb = app.PDBFile(input_pdb)

    # Build residue list and atom mapping for ALL residues first
    all_residue_keys = []
    residue_info = {}
    atom_to_residue = {}
    atom_to_chain = {}
    skipped_residues = set()

    for residue in pdb.topology.residues():
        res_key = _get_residue_key(residue)
        chain_id = res_key[0]

        # Skip waters unless requested
        if residue.name in ("HOH", "WAT", "TIP3") and not include_waters:
            skipped_residues.add(res_key)
            for atom in residue.atoms():
                atom_to_residue[atom.index] = res_key
                atom_to_chain[atom.index] = chain_id
            continue

        # Skip non-standard residues (ligands) unless requested
        if residue.name not in STANDARD_RESIDUES:
            if not include_ligands:
                warnings.warn("Skipping non-standard residue: {} {} {}".format(
                    res_key[0], res_key[1], residue.name))
                skipped_residues.add(res_key)
                for atom in residue.atoms():
                    atom_to_residue[atom.index] = res_key
                    atom_to_chain[atom.index] = chain_id
                continue

        all_residue_keys.append(res_key)
        residue_info[res_key] = {
            "chain_id": chain_id,
            "residue_id": res_key[1],  # PDB residue number
            "residue_name": res_key[2],
        }

        for atom in residue.atoms():
            atom_to_residue[atom.index] = res_key
            atom_to_chain[atom.index] = chain_id

    # Filter to source chains if specified
    if source_chain_set:
        residue_keys = [k for k in all_residue_keys if k[0] in source_chain_set]
        print("Filtering to source chains: {} ({} residues)".format(
            source_chains, len(residue_keys)))
    else:
        residue_keys = all_residue_keys

    # Build mapping of atom index to element and name for filtering
    atom_elements = {}
    atom_names = {}
    for atom in pdb.topology.atoms():
        elem = atom.element.symbol if atom.element else "X"
        atom_elements[atom.index] = elem
        atom_names[atom.index] = atom.name

    # Helper function to check if atom is a side chain heavy atom
    def is_sidechain_heavy(atom_idx):
        elem = atom_elements.get(atom_idx, "X")
        name = atom_names.get(atom_idx, "")
        return elem != "H" and name not in BACKBONE_ATOMS

    # Identify interface residues if target_chains is specified and interface_only=True
    interface_residues = set()
    if target_chain_set and interface_only:
        print("Finding interface residues with side chain atoms within {:.1f} A of target chains: {}...".format(
            interface_distance, target_chains))

        # Get positions
        positions = np.array([[p.x, p.y, p.z] for p in pdb.positions]) * 10  # nm to Angstrom

        # Build list of target chain heavy atom indices (including backbone)
        # Source uses side chain only, but we consider distance to ANY target heavy atom
        target_atoms = [i for i, chain in atom_to_chain.items()
                        if chain in target_chain_set and atom_elements.get(i, "X") != "H"]

        # For each source residue, check if any SIDE CHAIN heavy atom is within interface_distance of target
        for res_key in residue_keys:
            if res_key[0] in target_chain_set:
                # Target chain residues are always included
                interface_residues.add(res_key)
                continue

            # Get SIDE CHAIN heavy atoms for this residue
            res_atoms = [i for i, rk in atom_to_residue.items()
                         if rk == res_key and is_sidechain_heavy(i)]

            # Check minimum distance to any target side chain heavy atom
            for src_atom in res_atoms:
                for tgt_atom in target_atoms:
                    dist = np.linalg.norm(positions[src_atom] - positions[tgt_atom])
                    if dist <= interface_distance:
                        interface_residues.add(res_key)
                        break
                if res_key in interface_residues:
                    break

        # Filter to interface residues only
        residue_keys = [k for k in residue_keys if k in interface_residues]
        print("Found {} interface residues".format(len(residue_keys)))

    if not residue_keys:
        raise ValueError("No residues to score after filtering")

    print("Scoring {} residues ({} skipped)".format(len(residue_keys), len(skipped_residues)))

    # Set up force field
    if forcefield == "amber14-all":
        ff = app.ForceField("amber14-all.xml")
    elif forcefield == "charmm36":
        ff = app.ForceField("charmm36.xml")
    else:
        raise ValueError("Unsupported force field: {}. Use 'amber14-all' or 'charmm36'".format(forcefield))

    print("Creating system with {} force field...".format(forcefield))

    # Create system without periodic boundaries (vacuum)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )

    # Create integrator and simulation
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # Initialize energy accumulator for each residue
    residue_energies = defaultdict(float)

    # In interface mode, only calculate inter-chain non-bonded interactions
    interface_mode = target_chain_set is not None

    if interface_mode:
        print("Interface mode: calculating only inter-chain non-bonded interactions...")
    else:
        print("Decomposing energy by force type...")

    # Process each force in the system
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        force_name = force.__class__.__name__

        # In interface mode, skip bonded terms (only calculate inter-chain non-bonded)
        if interface_mode:
            if isinstance(force, (openmm.HarmonicBondForce, openmm.HarmonicAngleForce,
                                  openmm.PeriodicTorsionForce)):
                continue

        # Create a copy of the system with only this force for energy evaluation
        test_system = openmm.System()
        for i in range(system.getNumParticles()):
            test_system.addParticle(system.getParticleMass(i))

        # Clone the force
        force_copy = _clone_force(force)
        if force_copy is None:
            continue

        test_system.addForce(force_copy)

        # Attribute energy based on force type
        if isinstance(force, openmm.HarmonicBondForce):
            _attribute_bond_energy(force, simulation.context, atom_to_residue,
                                   residue_energies, skipped_residues)

        elif isinstance(force, openmm.HarmonicAngleForce):
            _attribute_angle_energy(force, simulation.context, atom_to_residue,
                                    residue_energies, skipped_residues)

        elif isinstance(force, openmm.PeriodicTorsionForce):
            _attribute_torsion_energy(force, simulation.context, atom_to_residue,
                                      residue_energies, skipped_residues)

        elif isinstance(force, openmm.NonbondedForce):
            _attribute_nonbonded_energy(force, simulation.context, atom_to_residue,
                                        residue_energies, skipped_residues, pdb.positions,
                                        atom_to_chain, target_chain_set, atom_names)

        elif isinstance(force, openmm.CMMotionRemover):
            # Skip center of mass motion remover - not an energy term
            pass

        else:
            # For other forces, try to get total energy and distribute evenly
            print("  Warning: Unhandled force type: {}".format(force_name))

    # Write output TSV
    print("Writing results to: {}".format(output_tsv))
    with open(output_tsv, "w") as f:
        f.write("chain_id\tresidue_id\tresidue_name\ttotal_energy\n")

        for res_key in residue_keys:
            info = residue_info[res_key]
            energy = residue_energies[res_key]
            f.write("{}\t{}\t{}\t{:.4f}\n".format(
                info["chain_id"],
                info["residue_id"],
                info["residue_name"],
                energy,
            ))

    # Print summary statistics
    energies = [residue_energies[k] for k in residue_keys]
    print("\nEnergy summary:")
    print("  Total: {:.2f} kJ/mol".format(sum(energies)))
    print("  Mean per residue: {:.2f} kJ/mol".format(np.mean(energies)))
    print("  Std dev: {:.2f} kJ/mol".format(np.std(energies)))
    print("  Min: {:.2f} kJ/mol (residue {})".format(
        min(energies), residue_keys[np.argmin(energies)]))
    print("  Max: {:.2f} kJ/mol (residue {})".format(
        max(energies), residue_keys[np.argmax(energies)]))

    return output_tsv


def _clone_force(force):
    """Create a copy of a force object."""
    # This is a simplified clone - for full functionality, use XML serialization
    try:
        xml = openmm.XmlSerializer.serialize(force)
        return openmm.XmlSerializer.deserialize(xml)
    except Exception:
        return None


def _attribute_bond_energy(force, context, atom_to_residue, residue_energies, skipped):
    """Attribute harmonic bond energy to residues (to residue of first atom)."""
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    for i in range(force.getNumBonds()):
        p1, p2, length, k = force.getBondParameters(i)

        # Get residues for both atoms
        res1 = atom_to_residue.get(p1)
        res2 = atom_to_residue.get(p2)

        if res1 in skipped and res2 in skipped:
            continue

        # Calculate bond energy
        r = np.linalg.norm(positions[p1] - positions[p2])
        r0 = length.value_in_unit(unit.nanometer)
        k_val = k.value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2)
        energy = 0.5 * k_val * (r - r0)**2

        # Attribute to first non-skipped residue (or split if both valid)
        if res1 not in skipped and res2 not in skipped:
            if res1 == res2:
                residue_energies[res1] += energy
            else:
                residue_energies[res1] += energy / 2
                residue_energies[res2] += energy / 2
        elif res1 not in skipped:
            residue_energies[res1] += energy
        elif res2 not in skipped:
            residue_energies[res2] += energy


def _attribute_angle_energy(force, context, atom_to_residue, residue_energies, skipped):
    """Attribute harmonic angle energy to the central atom's residue."""
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    for i in range(force.getNumAngles()):
        p1, p2, p3, angle0, k = force.getAngleParameters(i)

        # Central atom is p2
        res_central = atom_to_residue.get(p2)

        if res_central in skipped:
            continue

        # Calculate angle energy
        v1 = positions[p1] - positions[p2]
        v2 = positions[p3] - positions[p2]

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        angle0_val = angle0.value_in_unit(unit.radian)
        k_val = k.value_in_unit(unit.kilojoules_per_mole / unit.radian**2)
        energy = 0.5 * k_val * (angle - angle0_val)**2

        residue_energies[res_central] += energy


def _attribute_torsion_energy(force, context, atom_to_residue, residue_energies, skipped):
    """Attribute periodic torsion energy to the central bond's residues."""
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    for i in range(force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)

        # Central atoms are p2 and p3
        res2 = atom_to_residue.get(p2)
        res3 = atom_to_residue.get(p3)

        if res2 in skipped and res3 in skipped:
            continue

        # Calculate dihedral angle
        b1 = positions[p2] - positions[p1]
        b2 = positions[p3] - positions[p2]
        b3 = positions[p4] - positions[p3]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        if n1_norm < 1e-10 or n2_norm < 1e-10:
            continue

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        dihedral = np.arctan2(y, x)

        # Calculate torsion energy
        phase_val = phase.value_in_unit(unit.radian)
        k_val = k.value_in_unit(unit.kilojoules_per_mole)
        energy = k_val * (1 + np.cos(periodicity * dihedral - phase_val))

        # Attribute to central residues
        if res2 not in skipped and res3 not in skipped:
            if res2 == res3:
                residue_energies[res2] += energy
            else:
                residue_energies[res2] += energy / 2
                residue_energies[res3] += energy / 2
        elif res2 not in skipped:
            residue_energies[res2] += energy
        elif res3 not in skipped:
            residue_energies[res3] += energy


def _attribute_nonbonded_energy(force, context, atom_to_residue, residue_energies,
                                 skipped, positions, atom_to_chain=None, target_chain_set=None,
                                 atom_names=None):
    """Attribute non-bonded energy (LJ + electrostatics) split 50/50 between interacting residues.

    If target_chain_set is provided (interface mode):
    - Only interactions between source and target chains are counted
    - Source chain: only SIDE CHAIN atoms are considered (backbone excluded)
    - Target chain: ALL atoms are considered (side chain + backbone)
    - The energy is attributed only to the source chain residue
    """
    pos = np.array([[p.x, p.y, p.z] for p in positions]) * 10  # Convert to Angstrom for calculation

    # Get parameters for all atoms
    num_particles = force.getNumParticles()
    charges = []
    sigmas = []
    epsilons = []

    for i in range(num_particles):
        charge, sigma, epsilon = force.getParticleParameters(i)
        charges.append(charge.value_in_unit(unit.elementary_charge))
        sigmas.append(sigma.value_in_unit(unit.nanometer))
        epsilons.append(epsilon.value_in_unit(unit.kilojoules_per_mole))

    charges = np.array(charges)
    sigmas = np.array(sigmas)
    epsilons = np.array(epsilons)

    # Get exceptions (1-4 interactions with modified parameters)
    exceptions = {}
    for i in range(force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
        key = (min(p1, p2), max(p1, p2))
        exceptions[key] = {
            "chargeProd": chargeProd.value_in_unit(unit.elementary_charge**2),
            "sigma": sigma.value_in_unit(unit.nanometer),
            "epsilon": epsilon.value_in_unit(unit.kilojoules_per_mole),
        }

    # Constants for energy calculation
    ONE_4PI_EPS0 = 138.935456  # kJ*nm/(mol*e^2)

    # In interface mode, only consider side chain atoms
    interface_mode = target_chain_set is not None

    # Calculate pairwise energies
    if interface_mode:
        print("  Calculating side chain interactions with target chains {}...".format(
            "".join(sorted(target_chain_set))))
    else:
        print("  Calculating non-bonded interactions...")

    for i in range(num_particles):
        res_i = atom_to_residue.get(i)
        if res_i in skipped:
            continue

        chain_i = atom_to_chain.get(i) if atom_to_chain else None

        # In interface mode, skip backbone atoms for source chain (non-target) atoms
        # Target chain atoms include backbone
        if interface_mode and atom_names:
            name_i = atom_names.get(i, "")
            i_in_target = chain_i in target_chain_set if target_chain_set else False
            if name_i in BACKBONE_ATOMS and not i_in_target:
                continue

        for j in range(i + 1, num_particles):
            res_j = atom_to_residue.get(j)
            if res_j in skipped:
                continue

            chain_j = atom_to_chain.get(j) if atom_to_chain else None

            # In interface mode, skip backbone atoms for source chain (non-target) atoms
            # Target chain atoms include backbone
            if interface_mode and atom_names:
                name_j = atom_names.get(j, "")
                j_in_target = chain_j in target_chain_set if target_chain_set else False
                if name_j in BACKBONE_ATOMS and not j_in_target:
                    continue

            # If target_chain_set is specified, filter interactions
            if interface_mode and atom_to_chain:
                i_in_target = chain_i in target_chain_set
                j_in_target = chain_j in target_chain_set

                # Skip if neither atom is in target chains
                if not i_in_target and not j_in_target:
                    continue

                # Skip if both atoms are in target chains (intra-target interaction)
                if i_in_target and j_in_target:
                    continue

            key = (i, j)

            # Check if this is an exception (1-2, 1-3, or modified 1-4)
            if key in exceptions:
                exc = exceptions[key]
                if exc["epsilon"] == 0 and exc["chargeProd"] == 0:
                    # This is a 1-2 or 1-3 exclusion, skip
                    continue

                # Use exception parameters for 1-4 interactions
                r = np.linalg.norm(pos[i] - pos[j]) / 10  # Convert to nm
                if r < 0.01:  # Avoid division by zero
                    continue

                # Electrostatic energy
                e_elec = ONE_4PI_EPS0 * exc["chargeProd"] / r

                # LJ energy
                if exc["epsilon"] > 0 and exc["sigma"] > 0:
                    sig_r = exc["sigma"] / r
                    e_lj = 4 * exc["epsilon"] * (sig_r**12 - sig_r**6)
                else:
                    e_lj = 0

                energy = e_elec + e_lj
            else:
                # Regular non-bonded interaction
                r = np.linalg.norm(pos[i] - pos[j]) / 10  # Convert to nm
                if r < 0.01:  # Avoid division by zero
                    continue

                # Electrostatic energy
                e_elec = ONE_4PI_EPS0 * charges[i] * charges[j] / r

                # LJ energy (using Lorentz-Berthelot combining rules)
                sigma_ij = (sigmas[i] + sigmas[j]) / 2
                epsilon_ij = np.sqrt(epsilons[i] * epsilons[j])

                if epsilon_ij > 0 and sigma_ij > 0:
                    sig_r = sigma_ij / r
                    e_lj = 4 * epsilon_ij * (sig_r**12 - sig_r**6)
                else:
                    e_lj = 0

                energy = e_elec + e_lj

            # Attribute energy based on whether we're filtering by target chains
            if target_chain_set and atom_to_chain:
                # Attribute full energy to the non-target chain residue
                i_in_target = chain_i in target_chain_set
                j_in_target = chain_j in target_chain_set

                if not i_in_target:
                    residue_energies[res_i] += energy
                if not j_in_target:
                    residue_energies[res_j] += energy
            else:
                # Normal mode: split 50/50 between the two residues
                if res_i == res_j:
                    residue_energies[res_i] += energy
                else:
                    residue_energies[res_i] += energy / 2
                    residue_energies[res_j] += energy / 2


def main():
    """Command-line interface for residue scoring."""
    parser = argparse.ArgumentParser(
        description="Score per-residue energies for a protein structure using openmm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score all residues (input must be minimized)
  python score_residues.py score protein_minimized.pdb

  # Score interface: chain C residues interacting with chain A
  python score_residues.py score protein.pdb --source-chains C --target-chains A

  # Score all chain C residues (not just interface) against chain A
  python score_residues.py score protein.pdb --source-chains C --target-chains A --all-residues

  # Use custom interface distance cutoff (default: 8 Angstroms)
  python score_residues.py score protein.pdb --source-chains C --target-chains A --interface-distance 10

  # Prepare structure (fix missing atoms, add hydrogens)
  python score_residues.py prepare protein.pdb -o protein_fixed.pdb

  # Minimize structure
  python score_residues.py minimize protein_fixed.pdb -o protein_minimized.pdb

  # Full pipeline
  python score_residues.py prepare protein.pdb -o protein_fixed.pdb
  python score_residues.py minimize protein_fixed.pdb -o protein_minimized.pdb
  python score_residues.py score protein_minimized.pdb -o scores.tsv
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Score command
    score_parser = subparsers.add_parser("score", help="Calculate per-residue energies")
    score_parser.add_argument("input_pdb", help="Input PDB file (must be minimized)")
    score_parser.add_argument("-o", "--output", help="Output TSV file")
    score_parser.add_argument(
        "--forcefield",
        choices=["amber14-all", "charmm36"],
        default="amber14-all",
        help="Force field to use (default: amber14-all)",
    )
    score_parser.add_argument(
        "--include-waters",
        action="store_true",
        help="Include water molecules in scoring",
    )
    score_parser.add_argument(
        "--include-ligands",
        action="store_true",
        help="Include ligand molecules in scoring",
    )
    score_parser.add_argument(
        "--source-chains",
        type=str,
        default=None,
        help="Only score residues from these chains (e.g., 'C' or 'AB')",
    )
    score_parser.add_argument(
        "--target-chains",
        type=str,
        default=None,
        help="Only count interactions with these chains (e.g., 'A' for interface with chain A)",
    )
    score_parser.add_argument(
        "--all-residues",
        action="store_true",
        help="Include all residues, not just interface residues (disables interface_only)",
    )
    score_parser.add_argument(
        "--interface-distance",
        type=float,
        default=4.0,
        help="Heavy atom distance cutoff (Angstroms) for interface detection (default: 4.0)",
    )

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Fix missing atoms with pdbfixer")
    prepare_parser.add_argument("input_pdb", help="Input PDB file")
    prepare_parser.add_argument("-o", "--output", help="Output PDB file")
    prepare_parser.add_argument(
        "--no-hydrogens",
        action="store_true",
        help="Do not add missing hydrogens",
    )
    prepare_parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH for protonation state assignment (default: 7.0)",
    )
    prepare_parser.add_argument(
        "--keep-heterogens",
        action="store_true",
        help="Keep heterogens (ligands, ions) instead of removing them",
    )
    prepare_parser.add_argument(
        "--keep-water",
        action="store_true",
        help="Keep water molecules (only applies if heterogens are removed)",
    )
    prepare_parser.add_argument(
        "--add-missing-residues",
        action="store_true",
        help="Add missing residues from gaps (WARNING: renumbers residues from 1)",
    )

    # Minimize command
    minimize_parser = subparsers.add_parser("minimize", help="Energy minimize structure")
    minimize_parser.add_argument("input_pdb", help="Input PDB file")
    minimize_parser.add_argument("-o", "--output", help="Output PDB file")
    minimize_parser.add_argument(
        "--forcefield",
        choices=["amber14-all", "charmm36"],
        default="amber14-all",
        help="Force field to use (default: amber14-all)",
    )
    minimize_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum minimization iterations (default: 1000)",
    )
    minimize_parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Energy tolerance in kJ/mol/nm (default: 10.0)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "score":
        score_residues(
            args.input_pdb,
            output_tsv=args.output,
            forcefield=args.forcefield,
            include_waters=args.include_waters,
            include_ligands=args.include_ligands,
            source_chains=args.source_chains,
            target_chains=args.target_chains,
            interface_only=not args.all_residues,
            interface_distance=args.interface_distance,
        )

    elif args.command == "prepare":
        prepare_structure(
            args.input_pdb,
            output_pdb=args.output,
            add_hydrogens=not args.no_hydrogens,
            ph=args.ph,
            remove_heterogens=not args.keep_heterogens,
            keep_water=args.keep_water,
            add_missing_residues=args.add_missing_residues,
        )

    elif args.command == "minimize":
        minimize_structure(
            args.input_pdb,
            output_pdb=args.output,
            forcefield=args.forcefield,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
        )


if __name__ == "__main__":
    main()