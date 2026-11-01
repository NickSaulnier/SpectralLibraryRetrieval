import { useState } from "react";

const ACCEPT = ".asd,.csv,.txt,.tsv";

/** Drop zone (or file picker) for a field spectrometer file. */
export function FieldDrop({ onFile }: { onFile: (file: File) => void }) {
  const [over, setOver] = useState(false);
  return (
    <div
      className={`drop${over ? " over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
      }}
    >
      Drop an .asd or .csv field spectrum here, or{" "}
      <label>
        choose a file
        <input
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = "";
          }}
        />
      </label>
    </div>
  );
}
