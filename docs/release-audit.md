# Release audit and paper alignment

## Keep

- Five original per-scene NPZ archives and the combined Mix5 NPZ used by
  the paper experiments. The combined archive has 120k/40k/40k samples; each scene has
  24k/8k/8k.
- Original fixed Shenzhen test NPZ with 40k valid UE-BS links.
- Original Shenzhen few-shot NPZ files so the local paper experiments remain
  reproducible; corrected alternatives use a `_disjoint` filename suffix.
- The generator modules, the relevant YAML configs, and the custom XML/PLY
  scenes required to regenerate the channels.

## Rewrite or omit

- Replace the repository's old README and PLAN: they describe six training
  scenes including a synthetic street canyon and an older Shenzhen setup.
- Omit `configs/SHENZHEN.yaml` and `csi_SHENZHEN_3GHz_32x1024.npz`: these are
  a five-BS, depth-5, 500 m, 150k-target earlier Shenzhen variant, not the
  held-out ten-BS benchmark.
- Omit `csi_zjuBS0_*.npz`, the simple street canyon, Frobenius-renormalized
  copies, raw generation logs, unused renders, `.blend` working files, old ZIPs,
  the virtual environment, and checkpoints. None are required by the cited
  Mix5 and held-out Shenzhen protocol.
- Historical `3GHz` filenames remain for loader compatibility; metadata and
  documentation explicitly say 3.5 GHz.

## Findings requiring paper or release decision

1. The author confirmed that Shenzhen experiments used all 40k fixed test
   links. Correct the main-text reference to 20k; the appendix and local
   experiment scripts already use 40k.
2. Original Shenzhen few-shot train/validation files split *links*, so the
   same UE coordinate can occur in both subsets. A different sampling seed
   also leaves UE coordinates shared with the fixed test. This release
   retains exact original files and supplies separately marked corrected
   copies. For example, the original 12,800-link file has 1,565 UE
   coordinates shared between train and validation; 280 links are removed
   from its corrected copy because their UE coordinates occur in the fixed
   test. The corrected copy has 10,016 train and 2,504 validation links.
   Results from corrected copies need new evaluation.
3. `scattering_coefficient=0.1` is the exact Sionna RT configuration field.
   If the paper calls it `sigma_diff`, define that symbol as this field.
   The Shenzhen 25–50 m filter applies to rooftop height above terrain;
   the BS antenna is mounted another 2.5 m above the selected roof. If the
   paper says the BS antenna height itself is strictly 25–50 m, refine that
   wording to match the implementation.
4. The bundled JSON metadata has been made path-independent. XML/PLY runtime
   scenes use relative mesh paths; TOML files retain historical path settings
   and are provenance records until adjusted for a local Blender setup.
5. OSM attribution and data rights need to accompany scene meshes. The
   provenance and reuse rights of terrain imported through Blosm have not
   been established from the files in this repository. Confirm before
   public redistribution. The owner selected MIT for original code and
   CC BY 4.0 for the generated CSI archives; these grants do not override
   third-party scene geometry rights.

## Source references

- Sionna scene documentation: https://nvlabs.github.io/sionna/rt/api/scene.html
- OpenStreetMap copyright: https://www.openstreetmap.org/copyright
- OSM attribution guidance: https://osmfoundation.org/wiki/Licence/Attribution_Guidelines
- Blosm documentation: https://github.com/vvoovv/blosm/wiki/Documentation
