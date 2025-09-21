"""
Detect urea -- aromatic stacking interactions using MDAnalysis.
Also computes per-residue occupancies.
"""

import MDAnalysis as mda
import numpy as np
import pandas as pd
from math import degrees, acos
from tqdm import tqdm

topology = "./Neuropeptide1FVN375_Urea/1FVN_80MU/npt.gro" # .gro or .top
trajectory = "./Neuropeptide1FVN375_Urea/1FVN_80MU/md_cntr.xtc" # .xtc file
urea_resname = "Urea"
near_cutoff = 5.0        # Å : only consider urea molecules within this distance from protein
face_dist_cutoff = 5.0   # Å : centroid - urea distance cutoff for stacking (4.5-5.5 Å)
face_angle_cutoff = 30.0 # degrees : max angle for face-parallel
t_angle_min = 60.0       # degrees : min angle for T-shaped
t_angle_max = 120.0      # degrees : max angle for T-shaped


u = mda.Universe(topology, trajectory)

def centroid(positions):
    return np.mean(positions, axis=0)

def plane_normal(positions):
    v1 = positions[1] - positions[0]
    for i in range(2, positions.shape[0]):
        v2 = positions[i] - positions[0]
        n = np.cross(v1, v2)
        if np.linalg.norm(n) > 1e-6:
            return n / np.linalg.norm(n)
    return np.array([0,0,1])

AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "TRP": ["CD2", "CE2", "CE3", "CZ3", "CH2", "CZ2"]
}

UREA_O = "O"
UREA_C = "C"
UREA_NS = ["N1", "N2", "N"]

# Collect aromatic residues
arom_resnames = list(AROMATIC_RING_ATOMS.keys())
arom_sel_str = "protein and (resname " + " ".join(arom_resnames) + ")"
arom_residues = u.select_atoms(arom_sel_str).residues

arom_info = []
for res in arom_residues:
    atomnames = AROMATIC_RING_ATOMS.get(res.resname, [])
    available = [a for a in atomnames if a in res.atoms.names]
    if len(available) >= 3:
        arom_info.append((res, available))

# Track occupancy per residue
occupancy = {
    (res.resid, res.resname): {"face": 0, "T": 0, "total": 0}
    for res, _ in arom_info
}

n_frames = len(u.trajectory)

for ts in tqdm(u.trajectory, total=n_frames, desc="Frames"):
    near_urea = u.select_atoms(f"resname {urea_resname} and around {near_cutoff} protein")
    if len(near_urea) == 0:
        continue
    near_urea_res = near_urea.residues

    # Precompute aromatic centroids & normals
    arom_centroids = []
    for res, atom_list in arom_info:
        ring_atoms = res.atoms.select_atoms(" or ".join([f"name {n}" for n in atom_list]))
        if len(ring_atoms) < 3:
            continue
        pos = ring_atoms.positions
        c = centroid(pos)
        nrm = plane_normal(pos)
        arom_centroids.append((res, c, nrm))

    # For each urea, compute its O pos and normal
    for ures in near_urea_res:
        u_atoms = ures.atoms
        selO = u_atoms.select_atoms(f"name {UREA_O}")
        if len(selO) == 0:
            continue
        uO = selO[0].position

        Ns = []
        for nname in UREA_NS:
            sel = u_atoms.select_atoms(f"name {nname}")
            if len(sel) > 0:
                Ns.extend([a.position for a in sel])
            if len(Ns) >= 2:
                break
        if len(Ns) < 2:
            continue

        Csel = u_atoms.select_atoms(f"name {UREA_C}")
        Cpos = Csel[0].position if len(Csel) > 0 else np.mean(np.array(Ns + [uO]), axis=0)
        unrm = np.cross(Ns[0] - Cpos, Ns[1] - Cpos)
        if np.linalg.norm(unrm) == 0:
            unrm = np.array([0,0,1])
        else:
            unrm /= np.linalg.norm(unrm)

        for res, c, nrm in arom_centroids:
            d = np.linalg.norm(c - uO)
            if d > face_dist_cutoff:
                continue
            angle_norm = degrees(acos(abs(np.clip(np.dot(nrm, unrm), -1.0, 1.0))))
            vec = uO - c
            angle_vec = degrees(acos(abs(np.clip(np.dot(nrm, vec/np.linalg.norm(vec)), -1.0, 1.0))))

            resid_key = (res.resid, res.resname)
            if angle_norm <= face_angle_cutoff:
                occupancy[resid_key]["face"] += 1
                occupancy[resid_key]["total"] += 1
            elif t_angle_min <= angle_vec <= t_angle_max:
                occupancy[resid_key]["T"] += 1
                occupancy[resid_key]["total"] += 1

# Convert to per-frame occupancy (fraction)
df_occ = []
for (resid, resname), counts in occupancy.items():
    df_occ.append({
        "resid": resid,
        "resname": resname,
        "occupancy_face": counts["face"] / n_frames,
        "occupancy_T": counts["T"] / n_frames,
        "occupancy_total": counts["total"] / n_frames
    })

df_occ = pd.DataFrame(df_occ)
df_occ.to_csv("residue_occupancies.csv", index=False)

print("Done. Residue-wise occupancies saved to residue_occupancies.csv")

