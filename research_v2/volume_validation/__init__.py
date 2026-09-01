"""Volume validation package for RockSeg V2 — Experiment E5.

Validates the 2.5D-to-volume estimation pipeline using the Čapek et al. (2025)
dataset of 868 rock fragments with watertight 3D meshes (OBJ format).

Pipeline:
    3D mesh (ground truth)  ->  simulated 2.5D surface  ->  shape descriptors
    ->  volume estimation (box / ellipsoid / 2.5D / shape-aware)
    ->  comparison with reference volume
"""
