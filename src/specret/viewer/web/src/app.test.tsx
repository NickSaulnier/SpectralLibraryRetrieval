import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { route } from "./api/client";
import type { Identification, Job, PlumeRow } from "./api/types";
import { initialLayers, mergeLayers, moveLayer, tileQuery } from "./map/layerState";
import { selectionGeojson, summaryCsv } from "./panels/AreaPanel";
import { fmt, parseLatLon, toCsv } from "./lib";
import { PlumesPanel } from "./panels/PlumesPanel";
import { matchesCsv, ResultPanel, spectraCsv } from "./panels/ResultPanel";
import { addPin, MAX_PINS, pinLabel } from "./panels/PixelEvidence";
import { budgetShares } from "./panels/PlumeEvidence";
import { compass, filterSort, NO_FILTER, sceneDate, Timeline, type PlumeFilter, type SortKey } from "./panels/PlumesPanel";
import { lineThrough, roundBounds } from "./panels/CompareStats";
import { Checklist, newlyFinished } from "./panels/JobsPanel";
import { evidenceCardHtml, metresPerPixel, plumesCsv, plumesGeojson, scaleBar } from "./exports";
import { applyShared, decodeView, encodeView } from "./viewState";
import { windArrow } from "./map/windArrow";
import { gapped } from "./charts/Plot";

// Charts are drawn by Plotly, which jsdom cannot render; record what they are given instead
const plotted: unknown[][] = [];
vi.mock("plotly.js-basic-dist-min", () => ({
  default: { react: (_el: unknown, traces: unknown[]) => void plotted.push(traces) },
}));

describe("helpers", () => {
  it("quotes CSV fields that need it", () => {
    expect(toCsv([["a", 'say "hi"', "x,y", null, 3]])).toBe('a,"say ""hi""","x,y",,3\n');
  });

  it("parses lat, lon in common forms and rejects the rest", () => {
    expect(parseLatLon("37.53, -117.22")).toEqual({ lat: 37.53, lon: -117.22 });
    expect(parseLatLon("37.53 -117.22")).toEqual({ lat: 37.53, lon: -117.22 });
    expect(parseLatLon("37.53")).toBeNull();
    expect(parseLatLon("137, 20")).toBeNull();
  });

  it("formats missing numbers as a dash", () => {
    expect(fmt(null)).toBe("–");
    expect(fmt(Number.NaN)).toBe("–");
    expect(fmt(1234.567, 1)).toBe((1234.6).toLocaleString());
  });

  it("fills typed route templates and escapes parameters", () => {
    expect(route("/api/gas/{scene_id}/plume/{plume_id}", { scene_id: "a b", plume_id: 3 })).toBe(
      "/api/gas/a%20b/plume/3",
    );
  });

});

const result: Identification = {
  sensor: "emit",
  summary: { verdict: "calcite", best_material: "Calcite", best_group: "calcite", p_group_correct: 0.8 },
  matches: [
    { spectrum_id: "t:1", material_name: "Calcite", group: "calcite", score: 0.91, purity_flag: "a", spectrum: [0.5, 0.4],
      features: [{ window: "carbonate_2340", weight: 0.12, score: 0.8, query_depth: 0.11, query_center_nm: 2338,
                   cand_depth: 0.15, cand_center_nm: 2341, shape_r: 0.93 }] },
  ],
  wavelengths_nm: [2300, 2340],
  query: [0.45, 0.35],
  windows: {},
  pixel: { row: 3, col: 4, x: 10, y: 11, lat: 37.5, lon: -117.2 },
};

const layer = (name: string, kind = "continuous") => ({
  id: `S.${name}`, scene: "S", name, title: name, group: "G", kind, units: "", colormap: "magma",
  rescale: [0, 10], legend: [], ready: true, buildable: false, note: "",
});

describe("layer state", () => {
  it("starts with the context and the main product visible", () => {
    const ls = initialLayers([layer("rgb", "rgb"), layer("mineral_group", "categorical"), layer("ch4_enh")]);
    expect(ls.map((l) => [l.name, l.visible])).toEqual([["rgb", true], ["mineral_group", true], ["ch4_enh", false]]);
    expect(ls[2].stretch).toEqual([0, 10]);
    const gasOnly = initialLayers([layer("rgb", "rgb"), layer("ch4_enh")]);
    expect(gasOnly[1].visible).toBe(true);
  });

  it("keeps the user's settings when layer metadata refreshes", () => {
    const user = moveLayer(initialLayers([layer("a"), layer("b")]), "S.a", 1).map((l) =>
      l.name === "a" ? { ...l, visible: true, opacity: 0.3, stretch: [1, 2] as [number, number] } : l,
    );
    const fresh = [layer("a"), { ...layer("b"), legend: [{ code: 1, label: "x", rgb: [1, 2, 3] }] }];
    const merged = mergeLayers(fresh, user);
    expect(merged.map((l) => l.name)).toEqual(["b", "a"]);
    expect(merged[1]).toMatchObject({ visible: true, opacity: 0.3, stretch: [1, 2] });
    expect(merged[0].legend).toHaveLength(1);
    expect(mergeLayers([{ ...layer("z"), scene: "T" }], user)[0].scene).toBe("T"); // new scene: reset
  });

  it("builds tile queries from the layer's settings", () => {
    const [cat, cont] = initialLayers([layer("mineral_group", "categorical"), layer("ch4_enh")]);
    expect(tileQuery(cat)).toEqual({ groups: undefined, min_p: undefined });
    expect(tileQuery({ ...cat, keep: [3, 7], minP: 0.5 })).toEqual({ groups: "3,7", min_p: 0.5 });
    expect(tileQuery({ ...cat, keep: [] })).toEqual({ groups: "", min_p: undefined });
    expect(tileQuery({ ...cont, stretch: [0, 1500], cmap: "magma" })).toEqual({ rescale: "0,1500", colormap: "magma" });
  });

  it("reorders layers within bounds", () => {
    const ls = initialLayers([layer("a"), layer("b"), layer("c")]);
    expect(moveLayer(ls, "S.a", 1).map((l) => l.name)).toEqual(["b", "a", "c"]);
    expect(moveLayer(ls, "S.c", 1)).toBe(ls);
    expect(moveLayer(ls, "S.a", -1)).toBe(ls);
  });
});

describe("exports", () => {
  it("writes matches and spectra CSVs", () => {
    const m = matchesCsv(result, "scene_r3_c4").split("\n");
    expect(m[0]).toContain("material_name");
    expect(m[1]).toContain("scene_r3_c4,3,4,37.5,-117.2,calcite,0.8,1,t:1,Calcite,calcite,0.9100,a");
    expect(spectraCsv(result).split("\n").slice(0, 3)).toEqual([
      "wavelength_nm,query,Calcite (t:1)",
      "2300,0.45,0.5",
      "2340,0.35,0.4",
    ]);
  });
});

describe("PlumesPanel", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("lists plumes and shows evidence for the selected one", async () => {
    const responses: Record<string, unknown> = {
      "/api/gas/S1/plumes": {
        scene_id: "S1",
        plumes: [
          { plume_id: 1, gas: "CH4", confidence: 0.97, source_lat: 31.5, source_lon: -103.2, n_px: 40,
            max_alpha_ppmm: 900, q_kg_h: 1200, sigma_q_kg_h: 400, u10: 2.1, wind_source: "HRRR",
            confusers_fitted: "confuser:calcite", carbonate_fraction: 0.4 },
        ],
        summary: { scene_id: "S1", plumes: 1, candidates: 9, CH4: { plumes: 1, total_q_kg_h: 1200 } },
      },
      "/api/gas/S1/plume/1": {
        plume_id: 1, gas: "CH4", confidence: 0.97, verdict: "CH4", fitted_terms: { CH4: 600 },
        evidence: { gas_z: 8.2 }, evidence_meaning: { gas_z: "z-score of the gas" }, carbonate_fraction: 0.4,
        rate: { ime_kg: 50, length_m: 400, u10: 2.1, u_eff: 1.6, wind_source: "HRRR", q_kg_h: 1200, sigma_q_kg_h: 400 },
        relative_uncertainty: { wind: 0.3 },
        cross_sectional_flux: { q_csf_kg_h: 800, transects: 7, csf_iqr_rel: 0.5 },
        wind: { wind_u: 1.5, wind_v: 1.5 },
        spectra: { units: "", wavelengths_nm: [1650, 1660], windows: [[1600, 1700]], observed: [-0.01, -0.02],
                   null_sd: [0.003, 0.003], baseline: [0, 0], terms: { CH4: [-0.01, -0.019] }, model: [-0.01, -0.019],
                   remainder: [0, -0.001], signatures_per_ppmm: { CH4: [1e-5, 2e-5], CO2: [1e-7, 0] } },
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(responses[url]), { status: 200 })));
    const onSelect = vi.fn();
    const ui = (selected: number | null) => (
      <QueryClientProvider client={client}>
        <PlumesPanel gasScene="S1" sensor="EMIT" analyzed selected={selected} onSelect={onSelect} onChanged={() => {}} />
      </QueryClientProvider>
    );
    const client = new QueryClient();
    const { rerender } = render(ui(null));
    expect(await screen.findByText(/1 plumes from 9 candidates/)).toBeInTheDocument();
    expect(screen.getByText("calcite")).toBeInTheDocument();
    fireEvent.click(screen.getByText("CH4", { selector: "td" }));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ plume_id: 1 }));
    rerender(ui(1));
    await waitFor(() => expect(screen.getByText(/z-score of the gas/)).toBeInTheDocument());
    expect(screen.getByText(/carbonate under plume 40%/)).toBeInTheDocument();
    expect(screen.getByText(/towards NE/)).toBeInTheDocument();
    expect(screen.getByLabelText("IME 1,200 kg/h")).toBeInTheDocument();
    expect(screen.getByLabelText("Cross-section 800 kg/h")).toBeInTheDocument();
    expect(screen.getByText(/7 transects, IQR 50%/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Plume fit" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Gas signatures" })).toBeInTheDocument();
    await waitFor(() => expect(plotted.flatMap((t) => (t as { name?: string }[]).map((x) => x.name))).toContain("CH4 (fitted)"));
  });

  it("offers detection when the scene is not analyzed", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PlumesPanel gasScene="S2" sensor="EMIT" analyzed={false} selected={null} onSelect={() => {}} onChanged={() => {}} />
      </QueryClientProvider>,
    );
    expect(screen.getByRole("button", { name: "Detect plumes" })).toBeInTheDocument();
  });
});

describe("area exports", () => {
  const poly: [number, number][] = [[-117.25, 37.5], [-117.15, 37.5], [-117.15, 37.6], [-117.25, 37.6]];
  const summary = {
    area_km2: 98.0,
    min_p: 0.3,
    minerals: { pixels: 100, groups: [{ group: "white_mica", pixels: 60, fraction: 0.6, median_p: 0.7 }] },
    layers: [{ id: "S.ch4_enh", title: "CH4 enhancement", units: "ppm·m", pixels: 90, mean: 1, median: 2,
               p05: -5, p95: 9, hist_edges: [0, 1], hist_counts: [3] }],
    plumes: { count: 1, by_gas: { CH4: { count: 1, total_q_kg_h: 500 } }, ids: [4] },
  };

  it("writes a closed GeoJSON polygon with the summary", () => {
    const fc = JSON.parse(selectionGeojson(poly, summary));
    const ring = fc.features[0].geometry.coordinates[0];
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual(ring[4]);
    expect(fc.features[0].properties.area_km2).toBe(98);
  });

  it("writes the summary as CSV", () => {
    const lines = summaryCsv(summary).trim().split("\n");
    expect(lines).toEqual([
      "section,name,pixels,fraction_or_median,p05,p95,units",
      "mineral,white_mica,60,0.6000,,,",
      "layer,CH4 enhancement,90,2,-5,9,ppm·m",
      "plumes,CH4,1,500,,,kg/h",
    ]);
  });
});

describe("V3 evidence", () => {
  it("keeps at most five pins, newest last, one per name", () => {
    let pins = [] as ReturnType<typeof addPin>;
    for (let i = 0; i < 7; i++) pins = addPin(pins, { name: `p${i}`, result });
    expect(pins.map((p) => p.name)).toEqual(["p2", "p3", "p4", "p5", "p6"]);
    expect(pins).toHaveLength(MAX_PINS);
    expect(addPin(pins, { name: "p3", result }).map((p) => p.name)).toEqual(["p2", "p4", "p5", "p6", "p3"]);
    expect([pinLabel("20230524T105835_2314407_002_r703_c525"), pinLabel("field 1")]).toEqual(["r703 c525", "field 1"]);
  });

  it("breaks spectral lines across missing bands", () => {
    const g = gapped([1700, 1710, 1960, 1970]);
    expect(g.x).toEqual([1700, 1710, null, 1960, 1970]);
    expect(g.y([1, 2, 3, 4])).toEqual([1, 2, null, 3, 4]);
  });

  it("splits the rate variance between its sources", () => {
    const b = budgetShares({ wind: 0.3, retrieval: 0.3, mask: 0.0, background: 0.1 });
    expect(b.map((x) => x.name)).toEqual(["wind", "retrieval", "background"]);
    expect(b.reduce((s, x) => s + x.share, 0)).toBeCloseTo(1);
    expect(b[0].share).toBeCloseTo(0.09 / 0.19);
  });

  it("draws the wind arrow downwind of the source", () => {
    const fc = windArrow({ lat: 0, lon: 10, u: 2, v: 0 }, 300);
    const [shaft, head] = fc.features.map((f) => (f.geometry as GeoJSON.LineString).coordinates);
    expect(shaft[0]).toEqual([10, 0]);
    expect(shaft[1][0]).toBeCloseTo(10 + 600 / 111_320);
    expect(shaft[1][1]).toBeCloseTo(0);
    expect(head[1]).toEqual(shaft[1]);
    expect(head[0][0]).toBeLessThan(shaft[1][0]);
    expect(windArrow({ lat: 0, lon: 0, u: 0, v: 0 }).features).toHaveLength(0);
    expect([compass(0, 1), compass(1, 0), compass(-1, -1)]).toEqual(["N", "E", "SW"]);
  });

  it("shows a match's per-window evidence and the continuum-removed inset", async () => {
    plotted.length = 0;
    const withProfiles = {
      ...result,
      profiles: [{ window: "carbonate_2340", wavelengths_nm: [2300, 2340], query: [0, 0.11], matches: [[0, 0.15]],
                   depth: [0.11, 0.15], center_nm: [2338, 2341] }],
    };
    render(<ResultPanel status="" result={withProfiles} name="px" onPin={() => {}} />);
    expect(screen.getByText("carbonate-2340")).toBeInTheDocument();
    expect(screen.getByText("2,338 → 2,341 nm")).toBeInTheDocument();
    expect(screen.getByText("0.93")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Pin" })).toBeInTheDocument();
    await waitFor(() => expect(plotted.length).toBeGreaterThanOrEqual(2));
    const inset = plotted.find((t) => (t as { name: string }[])[0].name === "pixel" && t.length === 2) as { y: number[] }[];
    expect(inset[0].y).toEqual([1, 0.89]); // continuum-removed: 1 - depth
  });
});

describe("V4 gas across sensors", () => {
  const row = (over: Partial<PlumeRow>): PlumeRow => ({
    plume_id: 1, gas: "CH4", confidence: 0.9, source_lat: 31.5, source_lon: -103.2, n_px: 10, max_alpha_ppmm: 500,
    q_kg_h: 100, sigma_q_kg_h: 50, u10: 2, wind_source: "HRRR", confusers_fitted: null, carbonate_fraction: null,
    ...over,
  });

  it("filters and sorts plumes, missing values last", () => {
    const rows = [
      row({ plume_id: 1, q_kg_h: 100, sensor: "EMIT" }),
      row({ plume_id: 2, q_kg_h: null, sensor: "HyTES", gas: "NH3" }),
      row({ plume_id: 3, q_kg_h: 900, sensor: "EMIT", confusers_fitted: "confuser:calcite", confidence: 0.4 }),
    ];
    const ids = (f: Partial<PlumeFilter>, key: SortKey = "q_kg_h", desc = true) =>
      filterSort(rows, { ...NO_FILTER, ...f }, key, desc).map((r) => r.plume_id);
    expect(ids({})).toEqual([3, 1, 2]);
    expect(ids({}, "q_kg_h", false)).toEqual([1, 3, 2]);
    expect(ids({ sensor: "EMIT" })).toEqual([3, 1]);
    expect(ids({ gas: "NH3" })).toEqual([2]);
    expect(ids({ confusers: "with" })).toEqual([3]);
    expect(ids({ confusers: "without" })).toEqual([1, 2]);
    expect(ids({ minConfidence: 0.5 })).toEqual([1, 2]);
  });

  it("dates scenes from all three sensors", () => {
    expect(sceneDate("20230608T162322_2315911_014")).toBe("2023-06-08");
    expect(sceneDate("ang20161001t205456")).toBe("2016-10-01");
    expect(sceneDate("2014-07-06.194610.SaltonSea.Line2")).toBe("2014-07-06");
  });

  it("lists the overpasses over a source in date order", () => {
    const p = row({ covered_by: ["20240101T000000_2400000_001", "20220101T000000_2200000_001"],
                    seen_in: ["20240101T000000_2400000_001"] });
    render(<Timeline plume={p} scene="20230608T162322_2315911_014" />);
    expect(screen.getByText(/seen in 1 of 2/)).toBeInTheDocument();
    const items = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(items).toEqual(["2022-01-01 not seen", "2023-06-08 this scene", "2024-01-01 seen"]);
  });

  it("draws comparison lines through the origin and rounds the view", () => {
    expect(lineThrough(0.5, [-100, 300])).toEqual({ x: [-100, 300], y: [-50, 150] });
    expect(roundBounds([-103.123456, 31.987654, -103.0, 32.0])).toEqual([-103.1235, 31.9877, -103, 32]);
  });
});

describe("V5 sharing, exports and jobs", () => {
  const ls = () =>
    initialLayers([layer("rgb", "rgb"), layer("mineral_group", "categorical"), layer("ch4_enh"), layer("co2_enh")]);

  it("round-trips a view through the URL", () => {
    const layers = ls().map((l) =>
      l.name === "ch4_enh"
        ? { ...l, visible: true, opacity: 0.55, stretch: [0, 900] as [number, number], cmap: "rdbu_r" }
        : l.name === "mineral_group"
          ? { ...l, keep: [3, 7], minP: 0.4 }
          : l,
    );
    const area: [number, number][] = [[-117.3, 37.5], [-117.2, 37.5], [-117.2, 37.6]];
    const q = encodeView({ scene: "S", tab: "plumes", camera: { lon: -117.25, lat: 37.55, zoom: 11.234 }, plume: 4,
                           point: { lat: 37.51, lon: -117.21 }, layers, area });
    expect(q).toContain("s=S");
    expect(q).toContain("v=-117.25,37.55,11.23");
    const v = decodeView("?" + q);
    expect(v).toMatchObject({ scene: "S", tab: "plumes", plume: 4, point: { lat: 37.51, lon: -117.21 }, area });
    expect(v.layers).toEqual([
      { name: "rgb", opacity: 1 },
      { name: "mineral_group", opacity: 0.8, keep: [3, 7], minP: 0.4 },
      { name: "ch4_enh", opacity: 0.55, stretch: [0, 900], cmap: "rdbu_r" },
    ]);
    // Applied to fresh layers: same visibility and settings, the shared ones on top in order
    const back = applyShared(ls(), v.layers!);
    expect(back.map((l) => [l.name, l.visible])).toEqual([
      ["co2_enh", false], ["rgb", true], ["mineral_group", true], ["ch4_enh", true],
    ]);
    expect(back[3]).toMatchObject({ opacity: 0.55, stretch: [0, 900], cmap: "rdbu_r" });
    expect(back[2]).toMatchObject({ keep: [3, 7], minP: 0.4 });
    expect(decodeView("?v=1,2&pt=x&a=1,2;3,4")).toEqual({});
  });

  it("sizes the scale bar to a round length", () => {
    expect(metresPerPixel(0, 0)).toBeCloseTo(40075016.686 / 512);
    expect(scaleBar(10, 150)).toEqual({ metres: 1000, px: 100, label: "1 km" });
    expect(scaleBar(1, 150)).toEqual({ metres: 100, px: 100, label: "100 m" });
    expect(scaleBar(3, 150).metres).toBe(200);
  });

  const plumeRow: PlumeRow = { plume_id: 2, gas: "CH4", confidence: 0.9, source_lat: 31.5, source_lon: -103.2, n_px: 5,
    max_alpha_ppmm: 800, q_kg_h: 400, sigma_q_kg_h: 100, u10: 2, wind_source: "HRRR", confusers_fitted: "confuser:calcite",
    carbonate_fraction: 0.3, overpasses_covering: 2, overpasses_seen: 1 };

  it("exports the plume table as CSV and GeoJSON", () => {
    const lines = plumesCsv([plumeRow], "S1").trim().split("\n");
    expect(lines[0].split(",").slice(0, 4)).toEqual(["scene_id", "sensor", "plume_id", "gas"]);
    expect(lines[1].startsWith("S1,,2,CH4,0.9,31.5,-103.2")).toBe(true);
    const gj = JSON.parse(plumesGeojson([plumeRow, { ...plumeRow, source_lat: null }], "S1"));
    expect(gj.features).toHaveLength(1);
    expect(gj.features[0].geometry.coordinates).toEqual([-103.2, 31.5]);
    expect(gj.features[0].properties).toMatchObject({ scene_id: "S1", q_kg_h: 400, overpasses_seen: 1 });
  });

  it("writes a self-contained evidence card", () => {
    const e = {
      plume_id: 2, gas: "CH4", confidence: 0.9, verdict: "CH4", fitted_terms: { CH4: 600 }, evidence: { gas_z: 8 },
      evidence_meaning: { gas_z: "z <of> the gas" }, carbonate_fraction: 0.3,
      rate: { ime_kg: 50, length_m: 400, u10: 2, u_eff: 1.6, wind_source: "HRRR", q_kg_h: 400, sigma_q_kg_h: 100 },
      relative_uncertainty: { wind: 0.3 }, cross_sectional_flux: { q_csf_kg_h: 350, transects: 5, csf_iqr_rel: 0.4 },
    };
    const html = evidenceCardHtml(e, plumeRow, "S1", [{ title: "Plume fit", src: "data:image/svg+xml,%3Csvg%3E" }]);
    expect(html).toContain("<title>Plume 2 (CH4) – S1</title>");
    expect(html).toContain("400 ± 100 kg/h");
    expect(html).toContain("350 kg/h, 5 transects, IQR 40%");
    expect(html).toContain("1 of 2 other overpasses");
    expect(html).toContain("z &lt;of&gt; the gas");
    expect(html).toContain('src="data:image/svg+xml,%3Csvg%3E"');
    expect(html).not.toMatch(/<script|https?:\/\//);
  });

  it("tells which jobs just finished", () => {
    const j = (id: string, state: Job["state"]) => ({ job_id: id, state }) as Job;
    expect(newlyFinished([j("a", "running"), j("b", "queued"), j("c", "done")],
                         [j("a", "done"), j("b", "running"), j("c", "done")])).toEqual(["a"]);
  });

  it("offers Run only for steps the viewer can do", () => {
    const onRun = vi.fn();
    render(
      <Checklist
        onRun={onRun}
        items={[
          { key: "a", level: "required", name: "Canonical library", ok: false, detail: "missing", hint: null,
            steps: [{ argv: ["ingest"], command: "specret ingest", runnable: true }] },
          { key: "b", level: "optional", name: "EMIT radiance", ok: false, detail: "none", hint: null,
            steps: [{ argv: ["download", "emit-gas"], command: "specret download emit-gas", runnable: false }] },
          { key: "c", level: "recommended", name: "Split", ok: true, detail: "ok", hint: null, steps: [] },
        ]}
      />,
    );
    const runs = screen.getAllByRole("button", { name: "Run" });
    expect(runs).toHaveLength(1);
    fireEvent.click(runs[0]);
    expect(onRun).toHaveBeenCalledWith(["ingest"]);
    expect(screen.getByText(/run it in a terminal/)).toBeInTheDocument();
  });
});
