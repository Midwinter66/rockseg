"""Quick visualization of external stone shapes from cached features.

Generates schematic 2.5D views (footprint + height profile) for representative
external stones based on their 12 shape features. Also compares T01 vs L01
feature distributions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "datasets" / "t01_l01_scaled_10mm" / "cache"
OUT_DIR = ROOT / "output" / "external_stone_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all(dataset_id):
    """Load all cached samples for a dataset."""
    folder = CACHE_DIR / dataset_id
    samples = []
    for p in sorted(folder.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        if d.get("status") == "success":
            samples.append(d)
    return samples


def schematic_footprint(C, AR, eq_diam_m, n=200):
    """Generate a schematic footprint (ellipse) from circularity and aspect ratio.
    
    This is approximate - real stones have irregular shapes, but ellipse gives
    a rough idea of the overall proportions.
    """
    # From AR = a/b -> a = b*AR
    # Area = pi*a*b = pi*b^2*AR
    # eq_diam = 2*sqrt(area/pi) = 2*b*sqrt(AR)
    # -> b = eq_diam / (2*sqrt(AR))
    # -> a = eq_diam * sqrt(AR) / 2
    b = eq_diam_m / (2 * np.sqrt(AR))
    a = eq_diam_m * np.sqrt(AR) / 2
    
    theta = np.linspace(0, 2*np.pi, n)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return x, y, a, b


def schematic_height_profile(H_mean_norm, H_std_norm, H_p25_norm, H_p75_norm, 
                              H_skew_norm, h_max_m, n=100):
    """Generate a schematic height profile from height distribution stats.
    
    Uses a skewed distribution approximation. This is very approximate but
    gives a sense of flat vs peaked vs skewed.
    """
    # Simple approach: create a height field that matches the statistics
    # Use a 1D profile along the major axis
    x = np.linspace(0, 1, n)
    
    # Create a bell-ish curve modulated by skew
    # Base: cosine-like dome
    base = np.cos((x - 0.5) * np.pi) * 0.5 + 0.5
    
    # Apply skew by warping x
    if H_skew_norm < 0:
        # Negative skew = top is flat, bottom is steep (more mass at top)
        warp = x ** (1 + abs(H_skew_norm) * 0.5)
    else:
        # Positive skew = top is peaked, bottom is wide
        warp = x ** (1 / (1 + H_skew_norm * 0.5))
    
    profile = np.cos((warp - 0.5) * np.pi) * 0.5 + 0.5
    
    # Scale to match H_std approximately
    # Normalize so that mean and std roughly match
    p_mean = profile.mean()
    p_std = profile.std()
    
    target_std = H_std_norm
    if p_std > 0:
        profile = (profile - p_mean) * (target_std / p_std) + H_mean_norm
    
    # Clip to [0, 1]
    profile = np.clip(profile, 0, 1)
    
    return profile * h_max_m


def schematic_height_map(C, AR, eq_diam_m, H_mean_norm, H_std_norm, 
                          H_p25_norm, H_p75_norm, H_skew_norm, fill_ratio,
                          grid_res=0.01):
    """Generate a schematic 2.5D height map."""
    _, _, a, b = schematic_footprint(C, AR, eq_diam_m)
    
    nx = max(10, int(2 * a / grid_res))
    ny = max(10, int(2 * b / grid_res))
    
    y, x = np.mgrid[-b:b:ny*1j, -a:a:nx*1j]
    
    # Elliptical mask
    r2 = (x/a)**2 + (y/b)**2
    mask = r2 <= 1.0
    
    # Radial distance from center (normalized 0-1)
    r_norm = np.sqrt(np.clip(r2, 0, 1))
    
    # Height profile: dome shape with skew
    base_h = np.cos(r_norm * np.pi / 2)  # 1 at center, 0 at edge
    
    # Apply skew along x-axis (major axis)
    x_norm = np.clip(x / a, -1, 1) * 0.5 + 0.5  # 0 to 1
    if H_skew_norm < 0:
        warp = x_norm ** (1 + abs(H_skew_norm) * 0.3)
    else:
        warp = x_norm ** (1 / (1 + H_skew_norm * 0.3))
    skew_mod = np.cos((warp - 0.5) * np.pi) * 0.5 + 0.5
    
    # Combine: radial dome modulated by skew
    h = base_h * (0.7 + 0.3 * skew_mod)
    
    # Adjust to match fill_ratio approximately
    # fill_ratio = V / (footprint_area * h_max)
    # For a half-ellipsoid, fill_ratio = 2/3 ≈ 0.667
    # Scale h to match fill_ratio
    current_fill = h[mask].mean()  # approximate
    if current_fill > 0:
        scale = H_mean_norm / current_fill
        h = h * scale
    
    h = h * mask
    h[~mask] = np.nan
    
    return h


def plot_stone_schematic(ax_top, ax_side, s, title):
    """Plot one stone: top view footprint + side view height profile."""
    eq_d = 2 * np.sqrt(s["V_2_5D"] / (1000**3) / (np.pi * s["fill_ratio"] * 0.667))  # rough
    # Better: compute from footprint area implied by fill_ratio and V_2_5D
    # V_2_5D = fill_ratio * footprint_area * h_max
    # h_max not directly available, estimate from features
    # Actually let's just normalize to unit size for shape comparison
    
    # Use unit eq_diam for shape comparison
    eq_diam = 1.0
    
    h_map = schematic_height_map(
        s["C"], s["AR"], eq_diam,
        s["H_mean_norm"], s["H_std_norm"],
        s["H_p25_norm"], s["H_p75_norm"], s["H_skew_norm"],
        s["fill_ratio"],
        grid_res=0.01
    )
    
    # Top view
    im = ax_top.imshow(h_map, origin="lower", cmap="YlOrRd", 
                        interpolation="bilinear", vmin=0)
    ax_top.set_title(title, fontsize=9)
    ax_top.set_aspect("equal")
    ax_top.axis("off")
    
    # Side view (profile along major axis)
    profile = schematic_height_profile(
        s["H_mean_norm"], s["H_std_norm"],
        s["H_p25_norm"], s["H_p75_norm"], s["H_skew_norm"],
        h_max_m=1.0
    )
    ax_side.plot(np.linspace(0, 1, len(profile)), profile, "b-", linewidth=1.5)
    ax_side.fill_between(np.linspace(0, 1, len(profile)), profile, alpha=0.3, color="blue")
    ax_side.set_title("Height profile", fontsize=8)
    ax_side.set_xlabel("Normalized distance", fontsize=7)
    ax_side.set_ylabel("Norm. height", fontsize=7)
    ax_side.tick_params(labelsize=6)
    ax_side.set_ylim(0, 1.1)
    
    return im


def main():
    print("Loading external stone data...")
    t01 = load_all("T01")
    l01 = load_all("L01")
    print(f"  T01: {len(t01)} samples")
    print(f"  L01: {len(l01)} samples")
    
    # --- Feature distribution comparison ---
    features = ["C", "AR", "solidity", "compactness", "eq_diam_ratio",
                "H_mean_norm", "H_std_norm", "H_p25_norm", "H_p75_norm", 
                "H_skew_norm", "fill_ratio", "ellipsoid_ratio", "y_ratio"]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, feat in enumerate(features):
        ax = axes[idx]
        t_vals = [s[feat] for s in t01 if feat in s]
        l_vals = [s[feat] for s in l01 if feat in s]
        
        bins = 25
        all_vals = t_vals + l_vals
        vmin, vmax = min(all_vals), max(all_vals)
        
        ax.hist(t_vals, bins=bins, range=(vmin, vmax), alpha=0.6, 
                label=f"T01 (n={len(t_vals)})", color="steelblue", density=True)
        ax.hist(l_vals, bins=bins, range=(vmin, vmax), alpha=0.6,
                label=f"L01 (n={len(l_vals)})", color="coral", density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    
    # Hide unused axes
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("External Stone Shape Features: T01 vs L01", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "feature_distributions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    
    # --- Summary statistics table ---
    print("\n=== Feature Summary ===")
    print(f"{'Feature':<20} {'T01 mean':>10} {'T01 std':>10} {'L01 mean':>10} {'L01 std':>10}")
    print("-" * 65)
    for feat in features:
        t_vals = [s[feat] for s in t01 if feat in s]
        l_vals = [s[feat] for s in l01 if feat in s]
        print(f"{feat:<20} {np.mean(t_vals):>10.4f} {np.std(t_vals):>10.4f} "
              f"{np.mean(l_vals):>10.4f} {np.std(l_vals):>10.4f}")
    
    # --- Representative stone schematics ---
    print("\nGenerating representative stone visualizations...")
    
    # Pick 6 stones from T01: small y_ratio, medium, large y_ratio (diverse shapes)
    t01_sorted = sorted(t01, key=lambda s: s["y_ratio"])
    picks_idx = [0, len(t01_sorted)//4, len(t01_sorted)//2, 
                  3*len(t01_sorted)//4, len(t01_sorted)-1]
    # Also pick one with high AR and one with low C
    t01_by_ar = sorted(t01, key=lambda s: s["AR"])
    
    selected = [
        t01_sorted[0],  # lowest y_ratio (most "spiky")
        t01_by_ar[-1],  # highest AR (most elongated)
        t01_sorted[len(t01_sorted)//2],  # median
        t01_by_ar[0],   # lowest AR (most round)
        t01_sorted[-1],  # highest y_ratio (most "blocky")
        sorted(t01, key=lambda s: s["C"])[0],  # least circular
    ]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    
    for idx, s in enumerate(selected[:6]):
        ax_top = axes[0, idx]
        ax_side = axes[1, idx]
        
        sid = s["sample_id"]
        y = s["y_ratio"]
        ar = s["AR"]
        c = s["C"]
        
        plot_stone_schematic(ax_top, ax_side, s, f"{sid}\ny={y:.3f}  AR={ar:.2f}  C={c:.2f}")
    
    fig.suptitle("T01 Representative Stones (Schematic) — shape diversity", 
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "t01_representative_stones.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    
    # Same for L01
    l01_sorted = sorted(l01, key=lambda s: s["y_ratio"])
    l01_by_ar = sorted(l01, key=lambda s: s["AR"])
    
    selected_l = [
        l01_sorted[0],
        l01_by_ar[-1],
        l01_sorted[len(l01_sorted)//2],
        l01_by_ar[0],
        l01_sorted[-1],
        sorted(l01, key=lambda s: s["C"])[0],
    ]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    
    for idx, s in enumerate(selected_l[:6]):
        ax_top = axes[0, idx]
        ax_side = axes[1, idx]
        
        sid = s["sample_id"]
        y = s["y_ratio"]
        ar = s["AR"]
        c = s["C"]
        
        plot_stone_schematic(ax_top, ax_side, s, f"{sid}\ny={y:.3f}  AR={ar:.2f}  C={c:.2f}")
    
    fig.suptitle("L01 Representative Stones (Schematic) — shape diversity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "l01_representative_stones.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    
    # --- y_ratio comparison (most important for volume) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t_y = [s["y_ratio"] for s in t01]
    l_y = [s["y_ratio"] for s in l01]
    
    bins = np.linspace(min(t_y + l_y), max(t_y + l_y), 30)
    ax.hist(t_y, bins=bins, alpha=0.6, label=f"T01 (n={len(t_y)})", 
            color="steelblue", density=True)
    ax.hist(l_y, bins=bins, alpha=0.6, label=f"L01 (n={len(l_y)})",
            color="coral", density=True)
    
    ax.axvline(np.mean(t_y), color="steelblue", linestyle="--", alpha=0.8,
               label=f"T01 mean={np.mean(t_y):.3f}")
    ax.axvline(np.mean(l_y), color="coral", linestyle="--", alpha=0.8,
               label=f"L01 mean={np.mean(l_y):.3f}")
    
    ax.set_xlabel("y_ratio = V_true / V_2.5D", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Correction Ratio Distribution: T01 vs L01", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    out = OUT_DIR / "y_ratio_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
