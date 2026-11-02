declare module "plotly.js-basic-dist-min" {
  const Plotly: {
    react: (el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
    purge: (el: HTMLElement) => void;
    toImage: (el: HTMLElement, opts: { format: "svg" | "png"; width?: number; height?: number }) => Promise<string>;
  };
  export default Plotly;
}
