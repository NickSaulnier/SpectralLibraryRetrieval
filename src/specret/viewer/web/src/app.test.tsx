import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { route } from "./api/client";
import type { Identification } from "./api/types";
import { imagePixel } from "./components/ZoomImage";
import { fmt, parseLatLon, toCsv } from "./lib";
import { GasPanel } from "./panels/GasPanel";
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

  it("maps clicks to image pixels through pan and zoom", () => {
    const box = { left: 10, top: 20 } as DOMRect;
    expect(imagePixel(10, 20, box, { scale: 1, tx: 0, ty: 0 })).toEqual({ x: 0, y: 0 });
    // zoomed 4x and panned by (-40, -80): client (50, 100) is image (20, 40)
    expect(imagePixel(50, 100, box, { scale: 4, tx: -40, ty: -80 })).toEqual({ x: 20, y: 40 });
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

describe("GasPanel", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("lists plumes and shows evidence for the selected one", async () => {
    const responses: Record<string, unknown> = {
      "/api/gas/scenes": [{ scene_id: "S1", analyzed: true, job: null }],
      "/api/gas/S1/plumes": {
        scene_id: "S1",
        plumes: [
          { plume_id: 1, gas: "CH4", confidence: 0.97, source_lat: 31.5, source_lon: -103.2, n_px: 40,
            max_alpha_ppmm: 900, q_kg_h: 1200, sigma_q_kg_h: 400, u10: 2.1, wind_source: "HRRR",
            confusers_fitted: "confuser:calcite", carbonate_fraction: 0.4, x: 5, y: 6 },
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
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => new Response(JSON.stringify(responses[url]), { status: 200 })),
    );
    render(
      <QueryClientProvider client={new QueryClient()}>
        <GasPanel />
      </QueryClientProvider>,
    );
    expect(await screen.findByText(/1 plumes from 9 candidates/)).toBeInTheDocument();
    expect(screen.getByText("calcite")).toBeInTheDocument();
    fireEvent.click(screen.getByText("CH4", { selector: "td" }));
    await waitFor(() => expect(screen.getByText(/z-score of the gas/)).toBeInTheDocument());
    expect(screen.getByText(/carbonate under plume 40%/)).toBeInTheDocument();
  });
});
