const M_PER_DEG = 111_320;

export interface Wind {
  lat: number;
  lon: number;
  u: number; // m/s towards east
  v: number; // m/s towards north
}

/** The wind at a plume's source as a GeoJSON arrow: a shaft along the air's motion over
 * `seconds` of travel (5 minutes by default, about the age of an EMIT plume's visible extent),
 * and a two-stroke head. */
export function windArrow(w: Wind, seconds = 300): GeoJSON.FeatureCollection {
  const k = Math.cos((w.lat * Math.PI) / 180);
  const at = (east: number, north: number): [number, number] => [w.lon + east / (M_PER_DEG * k), w.lat + north / M_PER_DEG];
  const [de, dn] = [w.u * seconds, w.v * seconds];
  const len = Math.hypot(de, dn);
  if (!(len > 0)) return { type: "FeatureCollection", features: [] };
  const tip = at(de, dn);
  const head = (sign: number) => {
    const a = Math.atan2(dn, de) + Math.PI + sign * (Math.PI / 7);
    return at(de + 0.25 * len * Math.cos(a), dn + 0.25 * len * Math.sin(a));
  };
  const line = (coords: [number, number][]): GeoJSON.Feature => ({
    type: "Feature",
    properties: {},
    geometry: { type: "LineString", coordinates: coords },
  });
  return { type: "FeatureCollection", features: [line([at(0, 0), tip]), line([head(1), tip, head(-1)])] };
}
