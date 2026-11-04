// CSV parsing/serialisation for one partial cycle (columns time_s, voltage_v, current_a,
// temperature_c). Pure module (no DOM). Errors are thrown as CsvError with a user-facing message.

export const CSV_COLUMNS = Object.freeze(["time_s", "voltage_v", "current_a", "temperature_c"]);
export const MAX_ROWS = 20000;
export const MIN_ROWS = 10;
export const MAX_FILE_BYTES = 5 * 1024 * 1024;

export class CsvError extends Error {
  constructor(message) {
    super(message);
    this.name = "CsvError";
  }
}

function splitLine(line, delimiter) {
  return line.split(delimiter).map((cell) => cell.trim().replace(/^"(.*)"$/, "$1").trim());
}

/**
 * Parse CSV text into column arrays.
 * @param {string} text
 * @returns {{time_s: number[], voltage_v: number[], current_a: number[], temperature_c: number[], rows: number}}
 */
export function parseCycleCsv(text) {
  if (typeof text !== "string" || !text.trim()) throw new CsvError("The file is empty.");
  const lines = text.replace(/^\ufeff/, "").split(/\r\n|\n|\r/);
  let headerIndex = lines.findIndex((l) => l.trim() && !l.trim().startsWith("#"));
  if (headerIndex === -1) throw new CsvError("The file has no header row.");
  const headerLine = lines[headerIndex];
  const delimiter = headerLine.includes(",") ? "," : headerLine.includes(";") ? ";" : headerLine.includes("\t") ? "\t" : ",";
  const header = splitLine(headerLine, delimiter).map((c) => c.toLowerCase());
  const index = {};
  for (const col of CSV_COLUMNS) {
    const i = header.indexOf(col);
    if (i === -1) {
      throw new CsvError(`Missing column "${col}". Expected a header row with: ${CSV_COLUMNS.join(", ")}.`);
    }
    index[col] = i;
  }
  const out = { time_s: [], voltage_v: [], current_a: [], temperature_c: [] };
  for (let n = headerIndex + 1; n < lines.length; n += 1) {
    const line = lines[n];
    if (!line.trim() || line.trim().startsWith("#")) continue;
    const cells = splitLine(line, delimiter);
    for (const col of CSV_COLUMNS) {
      const raw = cells[index[col]];
      const value = raw === undefined || raw === "" ? Number.NaN : Number(raw);
      if (!Number.isFinite(value)) {
        throw new CsvError(`Line ${n + 1}: ${col} is ${raw === undefined || raw === "" ? "empty" : `not a number ("${raw.slice(0, 20)}")`}.`);
      }
      out[col].push(value);
    }
    if (out.time_s.length > MAX_ROWS) throw new CsvError(`Too many rows: the limit is ${MAX_ROWS.toLocaleString("en-US")} samples. Downsample the record first.`);
  }
  const rows = out.time_s.length;
  if (rows < MIN_ROWS) throw new CsvError(`Only ${rows} data row(s); at least ${MIN_ROWS} samples are needed.`);
  for (let i = 1; i < rows; i += 1) {
    if (out.time_s[i] < out.time_s[i - 1]) throw new CsvError(`time_s must not decrease (line ${headerIndex + i + 2}).`);
  }
  return { ...out, rows };
}

/** Quick facts about a parsed segment for the UI. */
export function describeSegment(seg) {
  const t = seg.time_s;
  const v = seg.voltage_v;
  return {
    rows: t.length,
    duration: t[t.length - 1] - t[0],
    vStart: v[0],
    vEnd: v[v.length - 1],
    vMin: Math.min(...v),
    vMax: Math.max(...v),
  };
}

/** Serialise a segment ({time_s[], voltage_v[], current_a[], temperature_c[]}) to CSV text. */
export function segmentToCsv(seg) {
  const lines = [CSV_COLUMNS.join(",")];
  const n = seg.time_s.length;
  for (let i = 0; i < n; i += 1) lines.push(CSV_COLUMNS.map((c) => seg[c][i]).join(","));
  return `${lines.join("\n")}\n`;
}
