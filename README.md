# Three.js Colorectal Visualization Engine

## Features
- Stage 0-IV progression follows mucosa -> submucosa -> muscle -> lymph nodes -> distant metastasis (per stages.png)
- Entire colorectal anatomy scene (colon + rectum)
- Mucosal folds and lesion heat overlay on tissue wall
- Multi-class cells (`colonocyte`, `goblet`, `stem`, `tumor`) with membrane, nucleus, organelles
- Biologic connection rules (epithelial adherens neighborhoods, crypt-local stem links, tumor-dominant links)
- Real spatial mode projects loaded cell coordinates onto the colorectal wall geometry (not isolated cluster space)
- Edge color/opacity encodes estimated spread risk from energy level + energy gradient + tumor adjacency
- Dataset playback from `data/simulation.json`
- Camera presets and fly-through
- Frame and video export

## Run

```powershell
python -m http.server 8000
```

Open `http://localhost:8000`.

## Controls

- `Space`: pause/resume
- `R`: reset
- `1`: overview camera
- `2`: lumen fly-through camera
- `3`: rectum closeup camera
- `P`: export current frame as PNG
- `V`: start/stop live recording to WebM
- `B`: auto-record one full treatment cycle to WebM

## Data Input

Load order:
1. `data/real_cells.csv` + optional `data/real_contacts.csv` (real spatial mode)
2. `data/simulation.json` (playback mode)
3. Synthetic fallback

### Real Spatial Files

`data/real_cells.csv` columns:
- `cell_id` (string)
- `x`, `y`, `z` (float; `z` optional, defaults to 0)
- `cell_type` (`colonocyte|goblet|stem|tumor` or free text mapped to these)
- `energy` (float 0..1.2)
- `energy_post` (optional float; used for step 1)
- `risk_score` (optional fallback if `energy` missing)

`data/real_contacts.csv` columns:
- `src_id`
- `dst_id`
- `weight` (optional; currently ignored for rendering thickness)

Notes:
- If contacts file is missing, biologic rule-based edges are generated.
- If real files exist, engine shows `Mode: real spatial contacts (mapped to colorectum)`.

Schema:

```json
{
  "nodes": [
    {
      "id": 0,
      "position": [x, y, z],
      "energy": 0.82,
      "cellType": "tumor"
    }
  ],
  "edges": [[0, 1], [1, 2]],
  "steps": [
    { "step": 0, "action": 1, "energies": [0.82, 0.76, 0.71] },
    { "step": 1, "action": 2, "energies": [0.73, 0.68, 0.63] }
  ]
}
```

`cellType` values:
- `colonocyte`
- `goblet`
- `stem`
- `tumor`

Connection realism:
- Auto-generated edges are no longer generic nearest-neighbor lines.
- They are constrained by tissue topology (axial/ring neighborhood on mucosa), crypt band locality, and cell type.
- If you provide `edges` in data, those are still used directly.

## Export Notes

- PNG exports download one image at current view/time.
- WebM recording captures animated frames from the canvas.
- `B` records one complete treatment cycle (dataset length, or synthetic 160 steps).

# Thermodynamic-Graph-NN-for-RL-Optimized-Cancer-Treatment
