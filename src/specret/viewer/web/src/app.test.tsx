import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { route } from "./api/client";
import type { Identification } from "./api/types";
import { initialLayers, mergeLayers, moveLayer } from "./map/layerState";
import { fmt, parseLatLon, toCsv } from "./lib";
import { PlumesPanel } from "./panels/PlumesPanel";
import { matchesCsv, spectraCsv } from "./panels/ResultPanel";

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
    { spectrum_id: "t:1", material_name: "Calcite", group: "calcite", score: 0.91, purity_flag: "a", spectrum: [0.5, 0.4] },
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
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(responses[url]), { status: 200 })));
    const onSelect = vi.fn();
    const ui = (selected: number | null) => (
      <QueryClientProvider client={client}>
        <PlumesPanel gasScene="S1" analyzed selected={selected} onSelect={onSelect} onChanged={() => {}} />
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
  });

  it("offers detection when the scene is not analyzed", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PlumesPanel gasScene="S2" analyzed={false} selected={null} onSelect={() => {}} onChanged={() => {}} />
      </QueryClientProvider>,
    );
    expect(screen.getByRole("button", { name: "Detect plumes" })).toBeInTheDocument();
  });
});
