# Viewer: React Rewrite and Improvements

**Goal:** turn `specret view` from a single hand-written page into a React application with a real map. Minerals and gases from every sensor (EMIT, AVIRIS, AVIRIS-NG, HyTES, field spectra) should be explorable in one place: layers, evidence, comparisons with reference products, background jobs, shareable views and exports. The viewer stays local and offline-first, like the rest of the tool.

**What success means:**
- **One map for everything:**
  - mineral groups, confidence, unmixing fractions and gas enhancement as layers;
  - EMIT and published reference products, toggled and compared on the same map;
  - pan and zoom from a whole scene down to single pixels.
- **Every answer shows its evidence:**
  - for a pixel: its spectrum against the matches, continuum-removed features and per-window fit;
  - for a plume: its fitted gases and confusers, the residual against the gas signatures, both rate estimates and the uncertainty budget.
- **Long operations run as jobs** (mapping, plume detection, LWIR flights) that can be queued and watched from the UI.
- **A view can be shared** as a URL, and everything shown can be exported.
- **Ordinary use needs no Node.** `uv run specret view` keeps working.

## Decisions (2026-11-01)

| Question | Decision | Effect |
|---|---|---|
| Frontend stack | **React + TypeScript, built with Vite; the built assets are committed** | Source lives in `src/specret/viewer/web/`. `npm run build` writes to `src/specret/viewer/static/app/`, which is committed, so Python users never need Node; only frontend work does. A check fails when committed assets fall behind the source. |
| Map | **MapLibre GL JS with local raster tiles** | The server renders XYZ tiles from our COGs, so everything works offline. An online satellite or street basemap is an explicit opt-in toggle, off by default. Plumes and selections are vector layers. |
| Scope | **All four improvement areas** | Gas across all sensors with comparisons, mineral exploration, spectra and evidence views, and jobs, sharing and exports. Phases V0–V5 below. |

---

## 1. Today

- **Server:** `viewer/app.py` (FastAPI, 14 endpoints).
- **Page:** `viewer/static/index.html`, about 400 lines of vanilla JavaScript and Plotly, all inline.
- **What it shows:**
  - a mineral-group map as one PNG of the whole orthorectified scene; clicking it identifies the pixel and plots the spectrum against matches;
  - a lat/lon search;
  - drag-and-drop identification of field spectra;
  - CSV export of matches and spectra;
  - a gas panel: an EMIT enhancement PNG with plume outlines, a plume table and plume details;
  - background mapping and detection jobs with progress.

What limits it:

| Limit | Effect |
|---|---|
| Whole-scene PNGs (2261 × 1887 for EMIT) scaled to the panel | No real zoom. Gas maps are noise at panel size, and plume outlines are invisible until you scroll. |
| No basemap, no geographic context | Hard to tell where a plume or mineral patch is |
| Separate mineral and gas panels, one scene at a time | No overlays: you can't see carbonate under a plume, or our map beside EMIT's |
| HyTES and AVIRIS-NG are missing | LWIR flights and the second VSWIR sensor are CLI-only |
| State lives only in the page | Views can't be shared, and a reload loses the selection |
| One large inline script | Hard to grow or test, with no types against the API |

## 2. Architecture

```
src/specret/viewer/
  app.py                 FastAPI: JSON API, tile endpoint, serves the built app
  tiles.py               XYZ tiles from COGs (rio-tiler), colormaps, legends
  layers.py              layer registry: what exists per scene, how to render it
  static/app/            built React app (committed; produced by `npm run build`)
  web/                   React + TypeScript source (Vite)
    package.json, vite.config.ts, tsconfig.json
    src/
      api/               generated TypeScript types from FastAPI's OpenAPI schema + fetch hooks
      state/             view state (zustand), mirrored to the URL
      map/               MapLibre map, layer manager, legends, swipe compare, selection tools
      panels/            Pixel, Plume, Scene summary, Jobs, Layers, Field spectra
      charts/            spectrum, continuum-removed and fit-residual plots (Plotly basic bundle)
      components/        shared UI pieces: tables, badges, dialogs
```

- **API contract:** FastAPI already publishes OpenAPI. `openapi-typescript` generates the TypeScript types, so frontend and backend can't drift silently. Endpoints get Pydantic response models where they lack them.
- **Data fetching:** TanStack Query handles caching, retries and job polling.
- **State:**
  - the active scene, enabled layers and their opacity, the map view, and the selected pixel or plume are kept in the URL query string, which makes sharing work;
  - preferences such as theme and the basemap opt-in go to `localStorage`.
- **Tiles:**
  - `GET /api/tiles/{scene}/{layer}/{z}/{x}/{y}.png` is rendered with `rio-tiler` from the existing COGs. They are already tiled with overviews, so tiles are cheap.
  - Categorical layers (mineral group) use their colour tables; continuous layers (confidence, ppm·m) use named colormaps with stated ranges.
  - Rotated UTM grids (AVIRIS-NG) and the EMIT lat/lon grid are both reprojected by rio-tiler.
- **Vector layers:** plume outlines and source points come as GeoJSON (`plumes.geojson`, which already exists), and box selections as GeoJSON polygons.
- **Offline first:** all JavaScript, CSS and fonts are bundled, with no CDN. The only online request is the opt-in basemap.
- **Theme:** keep today's light/dark tokens, ported to CSS variables in the app.
- **Dependencies:**
  - Python: `rio-tiler`.
  - JavaScript: `react`, `react-dom`, `maplibre-gl`, `@tanstack/react-query`, `zustand`, `plotly.js-basic-dist-min`, and `openapi-typescript` (dev only).
  - Tests: `vitest` with `@testing-library/react`.

## 3. Phases

### Phase V0: Scaffold and parity (1 week)

**Status (2026-11-01): done.**
- **App:** React 19 + TypeScript + Vite in `viewer/web/`, built into `viewer/static/app/` (committed).
- **Served:** FastAPI serves the app at `/` with gzip, and the old page at `/legacy`.
- **Parity, checked in the browser:**
  - mineral map with pan, zoom and fit;
  - pixel click, lat/lon search and field upload, each identified with the chart and matches;
  - CSV exports;
  - mapping jobs;
  - EMIT gas panel: plume table, plume details, source marker and detection jobs.
- **Tests:** 7 Vitest tests; pytest checks the build is complete and matches its sources (a source hash).
- **Packages:** installed with pnpm, because this machine's global npm shim is broken (`web/README.md`).
- **Type gap:** response bodies are typed by hand until endpoints get Pydantic models, which come with the V1 layer API.

- Vite + React + TypeScript app in `viewer/web/`, built into `viewer/static/app/`. `app.py` serves it at `/`. The old page stays at `/legacy` until parity is confirmed, then is removed.
- Generated API types and a small typed fetch layer.
- Feature parity with today:
  - scene list and mapping jobs;
  - pixel identification with the spectrum plot and matches table;
  - lat/lon search;
  - field-spectrum drop;
  - CSV exports;
  - the EMIT gas panel.

  At this stage the map can be the existing PNG inside a simple pan/zoom view; MapLibre arrives in V1.
- **Build hygiene:**
  - `npm run build` and `npm test`;
  - a Python test that the committed `static/app/index.html` exists and references bundles that exist;
  - a check (`scripts/check_viewer_build.py`, wired into pytest) that fails when `web/` sources are newer than the committed build;
  - `node_modules` is gitignored.

**Exit:**
- Every current workflow works in the React app.
- The existing viewer tests pass against the new API.
- `uv run specret view` needs no Node.

### Phase V1: Map and layers (1–2 weeks)

- **Map:** MapLibre with a local tile endpoint (`viewer/tiles.py`) and a layer registry (`viewer/layers.py`) that lists, per scene, what can be shown:

  | Kind | Layers |
  |---|---|
  | Mineral | group (categorical), confidence, score, margin |
  | Gas | CH4 and CO2 enhancement (ppm·m), plume mask, plume confidence |
  | Reference | EMIT L2B MIN groups, EMIT L2B CH4ENH and CO2ENH, published AVIRIS-NG CH4 maps |
  | Context | true-colour RGB built from the radiance or reflectance (computed once and cached as a COG) |

- **Layer panel:** on/off, opacity, order, colormap and stretch for continuous layers, and a legend per layer.
- **Basemap:** none by default (offline). An opt-in toggle adds an online satellite or OSM basemap and says so.
- **Clicking the map:** a lat/lon click goes to the scene under it. The pixel panel shows minerals, and where a gas layer is on, the enhancement there too.
- **Scene switching:** a scene picker zooms to the scene's footprint. Scene footprints are drawn as outlines, so you can see what's loaded.

**Exit:**
- An EMIT scene pans and zooms smoothly to single pixels.
- Plume outlines are visible at every zoom.
- The mineral map and the CH4 enhancement can be overlaid with opacity.
- Everything works with the network unplugged.

### Phase V2: Mineral exploration (1–2 weeks)

- **Group filter:** show only chosen groups (for example carbonates or kaolin), with the rest greyed.
- **Confidence mask:** a slider hides pixels below a chosen calibrated probability; the "likely unknown" class gets its own colour.
- **Unmixing layers:** `map-scene --unmix` (new) stores the top endmembers and fractions per pixel, as `unmix.npz` plus COG layers. The viewer shows fraction maps per endmember group, plus the fit residual.
- **Box or polygon selection:** a summary of the area: the group composition (`scene_summary` with a bbox, extended to polygons), the median confidence, and a histogram of scores. The selection can be exported as GeoJSON.
- **Agreement view:** our group map against EMIT L2B MIN groups, as a swipe or side-by-side with an agreement layer (agree / disagree / no reference), reusing `eval/emit_l2bmin.py`.

**Exit:**
- The unmixing layers exist and render for a mapped scene.
- A box selection over Cuprite gives the same composition as the `scene_summary` MCP tool.
- The agreement view reproduces the Phase 3 agreement numbers for the Mountain Pass validation scene.

### Phase V3: Spectra and evidence (1–2 weeks)

- **Pixel panel:**
  - the spectrum against the top matches, as now;
  - a continuum-removed inset per diagnostic window;
  - per-window evidence cards: band centre, depth and shape fit, query against library;
  - the unmixing decomposition: endmember spectra scaled by fraction, their sum, and the residual.
- **Compare mode:** pin up to 5 pixels (or field spectra) and overlay them, with a table of their verdicts. Useful for checking that a patch is consistent.
- **Plume panel:** shows the evidence behind a plume.
  - **Plots:**
    - the plume's mean log-radiance residual, with the fitted gas and confuser terms and the remainder;
    - each candidate gas's signature, with the windows shaded;
    - the shifted-mask null spread as a band.
  - **Rates:** both estimates (IME and cross-sectional flux) with the uncertainty budget as a bar.
  - **Context:** the wind vector drawn on the map at the source.
- **Backend for the plume panel:** detection stores per-candidate fit spectra (`plume_spectra.npz`: ȳ, null mean and standard deviation, fitted terms, bands), so the panel needs no rerun. `explain_plume` returns them.

**Exit:**
- Every value in the plume panel traces to stored outputs, with no recomputation.
- The evidence shown for a pixel matches `explain_match` from MCP.

### Phase V4: Gas across sensors and comparisons (2 weeks)

- **AVIRIS-NG:** `detect-plumes` on an `ang…` line writes the same outputs as EMIT, and the viewer lists AVIRIS-NG lines beside EMIT scenes. The published Duren et al. CH4 map is a reference layer.
- **HyTES:**
  - `detect-plumes-lwir` also writes map-grid COGs (enhancement per gas, the ΔT layer, and the low-contrast mask), gridded from the flight's per-pixel lat/lon;
  - plume outlines are written as polygons, not just points.

  The viewer shows the LWIR gases (NH3, SO2, CH4, N2O, C2H4) with the thermal-contrast layer, so a reader can see where detection was impossible.
- **Compare view:**
  - a swipe or side-by-side of our enhancement against EMIT CH4ENH or CO2ENH, or against the published AVIRIS-NG map, using the same colour stretch;
  - a scatter of the two over the visible area, with r and the Deming slope from `eval/ghg_compare.py`.
- **Persistence:** with several overpasses of one area loaded, plumes are badged with their repeat count from `gas-persistence`, and a timeline lists the overpasses that saw each source.
- **Plume table:**
  - sort and filter by gas, confidence, rate, sensor and whether a confuser was fitted;
  - both rate estimates and the wind source;
  - clicking a row flies the map to the plume.

**Exit:**
- An EMIT scene, an AVIRIS-NG line and a HyTES flight each show their gases, plumes and evidence in the same UI.
- The compare view reproduces the G1 and G5 agreement numbers for the visible scene.

### Phase V5: Jobs, sharing and exports (1 week)

- **Jobs panel:**
  - one queue for every long operation: map scene (with or without unmixing), detect plumes (EMIT or AVIRIS-NG), the LWIR flight run, and persistence across selected scenes;
  - progress, elapsed time, logs, and cancel;
  - jobs run one at a time on the GPU, and the queue survives page reloads (it lives in the server process).
- **"Set up" from the UI:** `doctor` items show as a checklist. Items with `specret` fix steps get a "Run" button that queues the job. Downloads that need credentials stay CLI-only and show the command to run.
- **Sharing:** the URL holds the scene, layers, opacity, view and selection, and a "Copy link" button copies it. Anyone running the viewer on the same project data sees the same view.
- **Exports:**
  - the map view as PNG, with legend and scale bar;
  - the plume table as CSV or GeoJSON;
  - the selection summary as CSV;
  - pixel spectra and matches as CSV, as now;
  - a plume "evidence card" as a self-contained HTML file with its plots and numbers.

**Exit:**
- Every CLI pipeline step a user needs for a new scene can be started from the UI and watched to completion.
- A copied link restores the exact view after a server restart.

## 4. Testing

| Level | What | How |
|---|---|---|
| API | Every endpoint, tiles included (status, content type, tile size, empty-tile behaviour) | pytest + FastAPI TestClient against the synthetic workspace fixture already used by `test_phase3.py` |
| Components | Layer panel, plume table, pixel panel, URL state round-trip | Vitest + Testing Library, run by `npm test` |
| Build | Committed bundle exists and matches the sources | `scripts/check_viewer_build.py` in pytest |
| End to end | Load a scene, toggle layers, click a pixel, open a plume, run a job | Checked in the browser preview after each phase, with screenshots in the phase write-up. Playwright can be added later if regressions appear. |

## 5. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| The committed build drifts from the source | The viewer runs old code | A build-freshness check in pytest; rebuild before every commit that touches `web/` |
| Bundle size (MapLibre about 800 kB, Plotly basic about 1 MB) | Slow first load | Gzip/brotli from FastAPI; Plotly is loaded only when a chart opens; cached with long max-age, since file names are content-hashed |
| Tile rendering cost on big rasters | A sluggish map | COG overviews already exist; tiles are cached in memory per layer and colormap; HyTES and AVIRIS-NG get overviews when written |
| HyTES is swath data with no map grid | Can't be tiled directly | Gridding to COGs at detection time (V4), with nearest-neighbour from the per-pixel lat/lon |
| Online basemap breaks "everything local" | Privacy or offline surprises | Off by default; an explicit toggle with a note that it fetches from the internet |
| Long jobs block the server | An unresponsive UI | Jobs run in worker threads with one GPU job at a time; the API stays responsive |

## 6. Open questions

1. **Which online basemap for the opt-in:** OSM raster tiles (free, needs attribution) or a satellite source such as Esri World Imagery (check its terms). The default is OSM.
2. **Should the legacy page be removed at the end of V0, or kept until V1?** The default is to remove it once parity is confirmed.
3. **Multi-user use:** today the viewer is single-user on localhost. If it will be shared on a network, jobs need per-user queues and the API needs authentication, which is out of scope for this plan.
