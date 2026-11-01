/** Small formatting and export helpers shared by the panels. */

export const fmt = (v: number | null | undefined, digits = 0): string =>
  v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });

export const pretty = (s: string | null | undefined): string => (s ?? "").replaceAll("_", " ");

const cell = (v: unknown): string => {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
};

/** RFC 4180 CSV text: quotes fields containing commas, quotes or newlines. */
export const toCsv = (rows: unknown[][]): string => rows.map((r) => r.map(cell).join(",")).join("\n") + "\n";

export function download(filename: string, text: string, type = "text/csv"): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = filename.replace(/[^\w.-]+/g, "_");
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Parse "lat, lon" in any common separator; null if it is not two numbers. */
export function parseLatLon(text: string): { lat: number; lon: number } | null {
  const nums = text.match(/-?\d+(\.\d+)?/g);
  if (!nums || nums.length !== 2) return null;
  const [lat, lon] = nums.map(Number);
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}
