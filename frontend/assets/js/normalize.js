// Tolerant readers for GET /v1/benchmark (reports/benchmark/summary.json). The summary is written
// by the benchmark stage, so the UI accepts the documented shape and common variants (nested
// "metrics", objects keyed by battery or level, column-oriented predictions) and treats every
// field as optional. Pure module (no DOM).

import { isNum, isObj } from "./guards.js";

/** First finite number found under any of `keys` (also looks inside a nested `metrics`). */
export function num(obj, ...keys) {
  if (!isObj(obj)) return null;
  for (const key of keys) {
    if (isNum(obj[key])) return obj[key];
    if (isObj(obj.metrics) && isNum(obj.metrics[key])) return obj.metrics[key];
  }
  return null;
}

function str(obj, ...keys) {
  if (!isObj(obj)) return null;
  for (const key of keys) if (typeof obj[key] === "string" && obj[key]) return obj[key];
  return null;
}

/** Coverage at 90 %, whatever the key spelling (coverage_90, coverage90, picp_90, coverage{"0.9"}). */
function coverage90(obj) {
  const direct = num(obj, "coverage_90", "coverage90", "picp_90", "coverage@0.9");
  if (direct !== null) return direct;
  const nested = isObj(obj?.coverage) ? obj.coverage : isObj(obj?.metrics?.coverage) ? obj.metrics.coverage : null;
  if (nested) return num(nested, "0.9", "0.90", "90");
  return isNum(obj?.coverage) ? obj.coverage : null;
}

function width90(obj) {
  return num(obj, "mean_width_90", "width_90", "interval_width_90", "mean_width");
}

/** Rows of an array, or of an object keyed by id (the key is stored under `idKey`). */
function rowsOf(value, idKey) {
  if (Array.isArray(value)) return value.filter(isObj);
  if (isObj(value)) return Object.entries(value).filter(([, v]) => isObj(v)).map(([k, v]) => ({ [idKey]: k, ...v }));
  return [];
}

function leaderRow(r) {
  const model = str(r, "model", "model_name", "estimator") || "";
  const featureSet = str(r, "feature_set", "features", "set") || "";
  const isBaseline = Boolean(r.is_baseline) || featureSet === "baseline" || ["mean", "cycle_only", "capacity_oracle"].includes(model);
  return {
    experiment: str(r, "experiment") || "",
    featureSet,
    mode: str(r, "mode") || "",
    model,
    mae: num(r, "mae"),
    rmse: num(r, "rmse"),
    r2: num(r, "r2", "r_squared"),
    coverage90: coverage90(r),
    width90: width90(r),
    n: num(r, "n", "n_cycles"),
    nBatteries: num(r, "n_batteries"),
    isBaseline,
    leaky: Boolean(r.leaky) || model === "capacity_oracle" || featureSet === "legacy_v1" || featureSet === "capacity_oracle",
    postHoc: Boolean(r.post_hoc),
  };
}

/** Headline-like summary row ({feature_set, mode, model, mae, ...}) or null. */
function specRow(hd) {
  if (!isObj(hd)) return null;
  return {
    experiment: str(hd, "experiment") || "",
    featureSet: str(hd, "feature_set") || "",
    mode: str(hd, "mode") || "",
    model: str(hd, "model") || "",
    mae: num(hd, "mae"),
    rmse: num(hd, "rmse"),
    r2: num(hd, "r2"),
    coverage90: coverage90(hd),
    width90: width90(hd),
    n: num(hd, "n", "n_cycles"),
    nBatteries: num(hd, "n_batteries"),
  };
}

/** Normalise predictions into Map(battery -> sorted [{cycle, y, yhat, lo, hi}]) using 90 % bounds. */
function predictionsOf(raw) {
  const out = new Map();
  const push = (battery, p) => {
    if (!out.has(battery)) out.set(battery, []);
    out.get(battery).push(p);
  };
  const point = (r) => ({
    cycle: num(r, "cycle_index", "cycle"),
    y: num(r, "y", "y_true", "soh_pct", "true", "soh_true"),
    yhat: num(r, "yhat", "y_pred", "pred", "prediction", "soh_pred"),
    lo: num(r, "lo_90", "lower_90", "lo90", "lower", "lo"),
    hi: num(r, "hi_90", "upper_90", "hi90", "upper", "hi"),
  });
  if (Array.isArray(raw)) {
    for (const r of raw.filter(isObj)) {
      const battery = str(r, "battery_id", "battery", "group");
      if (battery) push(battery, point(r));
    }
  } else if (isObj(raw)) {
    for (const [battery, value] of Object.entries(raw)) {
      if (Array.isArray(value)) value.filter(isObj).forEach((r) => push(battery, point(r)));
      else if (isObj(value)) {
        // column-oriented: {cycle_index: [...], y_true: [...], ...}
        const col = (...keys) => keys.map((k) => value[k]).find(Array.isArray) || [];
        const cycles = col("cycle_index", "cycle");
        const ys = col("y", "y_true", "soh_pct", "true");
        const yh = col("yhat", "y_pred", "pred", "prediction");
        const lo = col("lo_90", "lower_90", "lower", "lo");
        const hi = col("hi_90", "upper_90", "upper", "hi");
        cycles.forEach((c, i) => push(battery, {
          cycle: isNum(c) ? c : null,
          y: isNum(ys[i]) ? ys[i] : null,
          yhat: isNum(yh[i]) ? yh[i] : null,
          lo: isNum(lo[i]) ? lo[i] : null,
          hi: isNum(hi[i]) ? hi[i] : null,
        }));
      }
    }
  }
  for (const [battery, pts] of out) {
    const clean = pts.filter((p) => isNum(p.cycle)).sort((a, b) => a.cycle - b.cycle);
    if (clean.length) out.set(battery, clean);
    else out.delete(battery);
  }
  return out;
}

/**
 * @param {unknown} raw GET /v1/benchmark body.
 */
export function normalizeBenchmark(raw) {
  const s = isObj(raw) ? raw : {};
  const leaderboard = rowsOf(s.leaderboard ?? s.results ?? s.models, "model").map(leaderRow).filter((r) => r.model || r.featureSet);
  // Secondary experiments (extended cells, leave-one-condition-out) keep their result rows in
  // their own sections; add them to the leaderboard unless it already lists that experiment.
  const listed = new Set(leaderboard.map((r) => r.experiment).filter(Boolean));
  for (const section of [s.extended, s.loco]) {
    if (!isObj(section) || !Array.isArray(section.results)) continue;
    const rows = section.results.filter(isObj).map(leaderRow).filter((r) => (r.model || r.featureSet) && r.experiment && !listed.has(r.experiment));
    leaderboard.push(...rows);
    rows.forEach((r) => listed.add(r.experiment));
  }

  let headline = null;
  let headlineDerived = false;
  if (isObj(s.headline)) {
    headline = specRow(s.headline);
  } else if (leaderboard.length) {
    const candidates = leaderboard.filter((r) => !r.isBaseline && !r.leaky && isNum(r.mae));
    const preferred = candidates.find((r) => r.featureSet === "combined" && r.model === "ridge" && (r.mode || "absolute") === "absolute");
    const best = preferred || candidates.reduce((a, b) => (a && a.mae <= b.mae ? a : b), null);
    if (best) {
      headline = { ...best, nBatteries: null };
      headlineDerived = true;
    }
  }

  const baselines = [];
  if (isObj(s.baselines)) {
    for (const [name, v] of Object.entries(s.baselines)) {
      const mae = isNum(v) ? v : num(v, "mae");
      if (mae !== null) baselines.push({ name, mae, leaky: Boolean(v?.leaky) || name === "capacity_oracle" });
    }
  } else {
    const experiment = headline?.experiment || leaderboard.find((x) => x.experiment)?.experiment || "";
    for (const r of leaderboard.filter((x) => x.isBaseline && isNum(x.mae) && (!x.experiment || x.experiment === experiment))) {
      baselines.push({ name: r.model, mae: r.mae, leaky: r.leaky });
    }
  }

  const perBattery = rowsOf(s.per_battery ?? s.per_battery_table ?? s.batteries, "battery_id").map((r) => ({
    battery: str(r, "battery_id", "battery", "group") || "",
    n: num(r, "n", "n_cycles"),
    mae: num(r, "mae"),
    rmse: num(r, "rmse"),
    r2: num(r, "r2"),
    coverage90: coverage90(r),
  })).filter((r) => r.battery);

  const coverage = rowsOf(s.coverage ?? s.coverage_table ?? s.calibration, "nominal").map((r) => {
    const nominalRaw = isNum(r.nominal) ? r.nominal : Number(r.nominal ?? r.level ?? r.confidence);
    return {
      nominal: Number.isFinite(nominalRaw) ? (nominalRaw > 1 ? nominalRaw / 100 : nominalRaw) : null,
      coverage: num(r, "coverage", "empirical", "picp"),
      width: num(r, "mean_width", "width", "interval_width"),
    };
  }).filter((r) => isNum(r.nominal)).sort((a, b) => a.nominal - b.nominal);

  return {
    headline,
    headlineDerived,
    relativeHeadline: specRow(s.relative_headline),
    baselines,
    leaderboard,
    perBattery,
    coverage,
    predictions: predictionsOf(s.predictions ?? s.per_battery_predictions),
    meta: {
      protocol: str(s, "protocol"),
      packageVersion: str(s, "package_version", "version"),
      generated: str(s, "generated_utc", "created_utc", "timestamp"),
      configSha: str(s, "config_sha256", "config_hash"),
      dataSha: str(s, "data_sha256", "features_sha256"),
      runtime: num(s, "runtime_s", "runtime_seconds"),
    },
  };
}

function categoryOf(id, label, explicit) {
  const c = String(explicit || "").toLowerCase();
  if (c.includes("report") || c.includes("contamin") || c === "v1_reported") return "v1_reported";
  if (c === "v2" || c.includes("headline")) return "v2";
  if (c) return "honest";
  const text = `${id} ${label}`.toLowerCase();
  if (/report|contamin|overlap/.test(text)) return "v1_reported";
  if (/\bv2\b|v2_|headline|partial/.test(text)) return "v2";
  return "honest";
}

/**
 * Normalise the optional `audit` field of the benchmark summary.
 * @returns {{rows: Array<{id: string, label: string, category: string, mae: number|null, r2: number|null, leaky: boolean}>,
 *   facts: {sohVsCapacityR2: number|null, capacityReconstructionR2: number|null, overlap: {rows: number, total: number}|null}}|null}
 */
export function normalizeAudit(raw) {
  if (raw === null || raw === undefined) return null;
  let list = [];
  if (Array.isArray(raw)) list = raw;
  else if (isObj(raw)) {
    const arrayField = ["rows", "scenarios", "results", "table", "experiments"].find((k) => Array.isArray(raw[k]));
    list = arrayField ? raw[arrayField] : rowsOf(Object.fromEntries(Object.entries(raw).filter(([, v]) => isObj(v) && (num(v, "mae") !== null || num(v, "r2") !== null))), "id");
  }
  const rows = list.filter(isObj).map((r, i) => {
    const id = str(r, "id", "name", "key", "scenario") || `scenario_${i + 1}`;
    const label = str(r, "label", "description", "title", "scenario") || id.replace(/_/g, " ");
    return {
      id,
      label,
      category: categoryOf(id, label, str(r, "category", "kind", "group")),
      mae: num(r, "mae"),
      r2: num(r, "r2"),
      leaky: Boolean(r.leaky),
      note: str(r, "note") || "",
    };
  }).filter((r) => r.mae !== null || r.r2 !== null);
  const src = isObj(raw) ? raw : {};
  const overlapObj = isObj(src.contaminated_test_overlap) ? src.contaminated_test_overlap : null;
  const facts = {
    sohVsCapacityR2: num(src, "soh_vs_capacity_within_battery_r2", "within_battery_r2"),
    capacityReconstructionR2: num(src, "capacity_reconstruction_r2"),
    overlap: overlapObj && isNum(overlapObj.overlapping_rows) && isNum(overlapObj.test_rows)
      ? { rows: overlapObj.overlapping_rows, total: overlapObj.test_rows } : null,
  };
  if (!rows.length && facts.sohVsCapacityR2 === null && facts.capacityReconstructionR2 === null) return null;
  return { rows, facts };
}
