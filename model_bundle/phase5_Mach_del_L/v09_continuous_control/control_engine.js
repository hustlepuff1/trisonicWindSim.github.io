/* Continuous coordinated nozzle control with quasi-steady ROM field feedback.
 * P0 follows the design schedule at the CURRENT actual actuator position.
 * This ideal synchronous pressure coupling is an assumption, not identified
 * pressure dynamics or transient CFD. The Mach gain is along the L/P0 path,
 * not a partial derivative at fixed pressure. Inputs outside the planned path
 * are rejected; hull membership is reported separately from path admissibility.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.NozzleControl = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DEFAULT_CONFIG = Object.freeze({
    sample_time_s: 0.05,
    actuator_velocity_time_s: 0.3,
    max_speed_mm_s: 0.5,
    measurement_filter_time_s: 0.2,
    position_kp_per_s: 2.0,
    position_ki_per_s2: 0.5,
    mach_kp_per_s: 1.2,
    mach_ki_per_s2: 0.2,
    anti_windup_time_s: 0.5,
    core_half_width_m: 0.05,
  });
  const CONFIG_ALIASES = Object.freeze({
    dt_s: "sample_time_s", actuator_tau_s: "actuator_velocity_time_s",
    measurement_tau_s: "measurement_filter_time_s",
    position_kp: "position_kp_per_s", position_ki: "position_ki_per_s2",
    mach_kp: "mach_kp_per_s", mach_ki: "mach_ki_per_s2",
    antiwindup_tau_s: "anti_windup_time_s",
  });

  function finite(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new TypeError(`${name} must be a finite number`);
    }
    return value;
  }
  function isVector(value) { return Array.isArray(value) || ArrayBuffer.isView(value); }
  function vector(value, name, length) {
    if (!isVector(value) || (length !== undefined && value.length !== length) || (value.length === 0 && length !== 0)) {
      throw new TypeError(`${name} has an invalid vector shape`);
    }
    for (let i = 0; i < value.length; i++) finite(value[i], `${name}[${i}]`);
    return value;
  }
  function clip(value, low, high) { return Math.max(low, Math.min(high, value)); }
  function minimum(values) {
    let result = Infinity;
    for (const value of values) result = Math.min(result, value);
    return result;
  }
  function increasing(values, name) {
    for (let i = 1; i < values.length; i++) {
      if (!(values[i] > values[i - 1])) throw new RangeError(`${name} must strictly increase`);
    }
  }
  function monotonicSign(values, name) {
    const sign = Math.sign(values[values.length - 1] - values[0]);
    if (!sign) throw new RangeError(`${name} has no unique inverse`);
    for (let i = 1; i < values.length; i++) {
      if (sign * (values[i] - values[i - 1]) <= 0) {
        throw new RangeError(`${name} must be strictly monotonic for a supported inverse`);
      }
    }
    return sign;
  }
  function bracket(values, value) {
    if (value <= values[0]) return 0;
    if (value >= values[values.length - 1]) return values.length - 2;
    let low = 0, high = values.length - 1;
    while (high - low > 1) {
      const mid = Math.floor((low + high) / 2);
      if (values[mid] <= value) low = mid;
      else high = mid;
    }
    return low;
  }
  function mix(a, b, t) { return a + t * (b - a); }
  function mixVector(a, b, t) { return Array.from(a, (value, i) => mix(value, b[i], t)); }

  // Sutherland-Hodgman clipping against the small physical core square.
  function clipPolygon(points, axis, bound, keepGreater) {
    const result = [];
    if (!points.length) return result;
    let previous = points[points.length - 1];
    let previousInside = keepGreater ? previous[axis] >= bound : previous[axis] <= bound;
    for (const current of points) {
      const inside = keepGreater ? current[axis] >= bound : current[axis] <= bound;
      if (inside !== previousInside) {
        const fraction = (bound - previous[axis]) / (current[axis] - previous[axis]);
        result.push([mix(previous[0], current[0], fraction), mix(previous[1], current[1], fraction)]);
      }
      if (inside) result.push(current);
      previous = current;
      previousInside = inside;
    }
    return result;
  }
  function polygonArea(points) {
    let twice = 0;
    for (let i = 0; i < points.length; i++) {
      const next = points[(i + 1) % points.length];
      twice += points[i][0] * next[1] - next[0] * points[i][1];
    }
    return Math.abs(twice) * 0.5;
  }
  function coreMetrics(rom, coefficients, width, wall, halfSize) {
    let area = 0, integral = 0;
    const ny = rom.ny, nz = rom.nz;
    for (let y = 0; y < ny; y++) {
      const y0 = width * rom.y_eta_edges[y], y1 = width * rom.y_eta_edges[y + 1];
      if (y0 >= halfSize || y1 <= 0) continue;
      for (let z = 0; z < nz; z++) {
        const e0 = rom.z_eta_edges[z], e1 = rom.z_eta_edges[z + 1];
        if (e0 * Math.min(wall[y], wall[y + 1]) >= halfSize) break;
        let polygon = [[y0, e0 * wall[y]], [y1, e0 * wall[y + 1]],
                       [y1, e1 * wall[y + 1]], [y0, e1 * wall[y]]];
        polygon = clipPolygon(polygon, 0, 0, true);
        polygon = clipPolygon(polygon, 0, halfSize, false);
        polygon = clipPolygon(polygon, 1, 0, true);
        polygon = clipPolygon(polygon, 1, halfSize, false);
        const weight = polygonArea(polygon);
        if (!(weight > 0)) continue;
        const index = z * ny + y;
        let mach = rom.mean[index];
        for (let mode = 0; mode < coefficients.length; mode++) mach += coefficients[mode] * rom.modes[mode][index];
        area += weight;
        integral += weight * mach;
      }
    }
    const expected = halfSize * halfSize;
    if (!Number.isFinite(integral) || !(area > 0) || Math.abs(area - expected) > expected * 1e-8) {
      throw new RangeError("Cross-section geometry does not cover the specified physical core square");
    }
    const mach = integral / area;
    if (!Number.isFinite(mach) || mach <= 0) throw new RangeError("Cross-field ROM produced a nonpositive core Mach");
    return {core_mach: mach, core_area_m2: area};
  }
  function validateCrossROM(rom) {
    if (!rom || !Number.isInteger(rom.ny) || !Number.isInteger(rom.nz) || rom.ny < 1 || rom.nz < 1) {
      throw new TypeError("field_roms.crossplane requires positive ny and nz");
    }
    vector(rom.y_eta_edges, "y_eta_edges", rom.ny + 1);
    vector(rom.z_eta_edges, "z_eta_edges", rom.nz + 1);
    increasing(rom.y_eta_edges, "y_eta_edges");
    increasing(rom.z_eta_edges, "z_eta_edges");
    for (const edges of [rom.y_eta_edges, rom.z_eta_edges]) {
      if (edges[0] !== 0 || edges[edges.length - 1] !== 1) throw new RangeError("Reference grid edges must span [0,1]");
    }
    vector(rom.mean, "crossplane.mean", rom.ny * rom.nz);
    if (!Array.isArray(rom.modes)) throw new TypeError("crossplane.modes must be an array");
    for (const mode of rom.modes) vector(mode, "crossplane.mode", rom.mean.length);
  }
  function fieldCoreMean(rom, coefficients, width, wall, halfSize = 0.05) {
    validateCrossROM(rom);
    vector(coefficients, "coefficients", rom.modes.length);
    vector(wall, "wall", rom.ny + 1);
    if (!(finite(width, "width") > 0) || !(finite(halfSize, "halfSize") > 0) || minimum(wall) <= 0) {
      throw new RangeError("Physical core and wall dimensions must be positive");
    }
    return coreMetrics(rom, coefficients, width, wall, halfSize).core_mach;
  }

  class Simulator {
    constructor(payload, caseIndex = 0, mode = "mach", config = {}) {
      if (!payload || !Array.isArray(payload.cases) || payload.cases.length !== 1) throw new TypeError("Payload requires one continuous coordinated operating path");
      this.payload = payload;
      this.crossROM = payload.field_roms && payload.field_roms.crossplane;
      validateCrossROM(this.crossROM);
      this.supportHull = payload.support_hull || null;
      if (this.supportHull) {
        vector(this.supportHull.center, "support_hull.center", 2);
        vector(this.supportHull.scale, "support_hull.scale", 2);
        if (minimum(this.supportHull.scale) <= 0 || !Array.isArray(this.supportHull.equations) || !this.supportHull.equations.length) {
          throw new RangeError("Support hull requires positive scales and half-space equations");
        }
        for (const equation of this.supportHull.equations) vector(equation, "support_hull.equation", 3);
      }
      this.config = {...DEFAULT_CONFIG};
      for (const settings of [payload.config || {}, config]) {
        for (const [key, value] of Object.entries(settings)) {
          const canonical = CONFIG_ALIASES[key] || key;
          if (!(canonical in DEFAULT_CONFIG)) throw new TypeError(`Unknown controller setting: ${key}`);
          this.config[canonical] = value;
        }
      }
      for (const [key, value] of Object.entries(this.config)) {
        finite(value, key);
        const allowZero = key === "measurement_filter_time_s" || key.endsWith("ki_per_s2");
        if (value < 0 || (!allowZero && value === 0)) throw new RangeError(`${key} must be ${allowZero ? "nonnegative" : "positive"}`);
      }
      // Keep restore defaults independent of subsequent live tuning.
      this.initialTuning = Object.freeze({
        max_speed_mm_s: this.config.max_speed_mm_s,
        position_kp_per_s: this.config.position_kp_per_s,
        position_ki_per_s2: this.config.position_ki_per_s2,
        mach_kp_per_s: this.config.mach_kp_per_s,
        mach_ki_per_s2: this.config.mach_ki_per_s2,
      });
      this.cases = payload.cases.map((entry, index) => this._validateCase(entry, index));
      this.mode = this._validateMode(mode);
      this.reset(caseIndex, mode);
    }
    _validateMode(mode) {
      if (mode !== "position" && mode !== "mach") throw new RangeError("Mode must be 'position' or 'mach'");
      return mode;
    }
    _validateCase(entry, index) {
      const name = `cases[${index}]`;
      if (!entry || !entry.samples) throw new TypeError(`${name} requires sample tables`);
      for (const key of ["target_mach", "P0_Pa", "L_min_mm", "L_max_mm", "L_initial_mm"]) finite(entry[key], `${name}.${key}`);
      if (entry.target_mach <= 0 || entry.P0_Pa <= 0 || !(entry.L_min_mm < entry.L_max_mm) ||
          entry.L_initial_mm < entry.L_min_mm || entry.L_initial_mm > entry.L_max_mm) throw new RangeError(`${name} has invalid coordinated-path bounds`);
      const samples = entry.samples;
      vector(samples.L_mm, `${name}.L_mm`);
      if (samples.L_mm.length < 2) throw new RangeError("Each operating case requires at least two sampled lengths");
      increasing(samples.L_mm, `${name}.L_mm`);
      if (Math.abs(samples.L_mm[0] - entry.L_min_mm) > 1e-9 ||
          Math.abs(samples.L_mm[samples.L_mm.length - 1] - entry.L_max_mm) > 1e-9) throw new RangeError("Sampled lengths must cover exactly the declared support");
      const count = samples.L_mm.length;
      for (const key of ["throat_height_mm", "core_mach", "cross_width_m", "P0_Pa", "schedule_mach"]) vector(samples[key], `${name}.${key}`, count);
      if (!Array.isArray(samples.training_hull_supported) || samples.training_hull_supported.length !== count ||
          samples.training_hull_supported.some(value => typeof value !== "boolean")) {
        throw new TypeError("Path samples require boolean training_hull_supported flags");
      }
      if (minimum(samples.P0_Pa) <= 0 || minimum(samples.schedule_mach) <= 0) throw new RangeError("Path pressure and schedule Mach must be positive");
      if (monotonicSign(samples.schedule_mach, `${name}.schedule_mach`) !== -1) throw new RangeError("Schedule Mach must decrease as path L increases");
      monotonicSign(samples.throat_height_mm, `${name}.throat_height_mm`);
      if (monotonicSign(samples.core_mach, `${name}.core_mach`) !== -1) throw new RangeError("Mach/L gain must be negative along the coordinated pressure path");
      for (const key of ["long_coefficients", "cross_coefficients", "wall_heights_m", "cross_wall_heights_m"]) {
        if (!Array.isArray(samples[key]) || samples[key].length !== count) throw new TypeError(`${name}.${key} has an invalid sample count`);
        const columns = key === "cross_coefficients" ? this.crossROM.modes.length :
                        key === "cross_wall_heights_m" ? this.crossROM.ny + 1 : samples[key][0].length;
        for (const row of samples[key]) vector(row, `${name}.${key}`, columns);
      }
      for (let i = 0; i < count; i++) {
        if (samples.cross_width_m[i] <= 0 || minimum(samples.wall_heights_m[i]) <= 0 || minimum(samples.cross_wall_heights_m[i]) <= 0) {
          throw new RangeError("Sampled geometry dimensions must be positive");
        }
        if (Math.abs(2000 * minimum(samples.wall_heights_m[i]) - samples.throat_height_mm[i]) > 1e-5) {
          throw new RangeError("Throat table differs from twice the sampled wall minimum");
        }
      }
      return entry;
    }
    _fieldAt(length) {
      const entry = this.currentCase;
      finite(length, "L_mm");
      if (length < entry.L_min_mm || length > entry.L_max_mm) throw new RangeError("Requested field lies outside the planned coordinated L/P0 path");
      const table = entry.samples;
      const i = bracket(table.L_mm, length);
      const fraction = (length - table.L_mm[i]) / (table.L_mm[i + 1] - table.L_mm[i]);
      const result = {actual_L_mm: length};
      for (const key of ["long_coefficients", "cross_coefficients", "wall_heights_m", "cross_wall_heights_m"]) {
        result[key] = mixVector(table[key][i], table[key][i + 1], fraction);
      }
      result.cross_width_m = mix(table.cross_width_m[i], table.cross_width_m[i + 1], fraction);
      result.P0_Pa = mix(table.P0_Pa[i], table.P0_Pa[i + 1], fraction);
      result.schedule_mach = mix(table.schedule_mach[i], table.schedule_mach[i + 1], fraction);
      if (this.supportHull) {
        const hull = this.supportHull;
        const x = (length - hull.center[0]) / hull.scale[0];
        const pressure = (Math.log(result.P0_Pa) - hull.center[1]) / hull.scale[1];
        const tolerance = hull.tolerance === undefined ? 1e-10 : finite(hull.tolerance, "support_hull.tolerance");
        if (tolerance < 0) throw new RangeError("Hull tolerance must be nonnegative");
        result.in_training_hull = hull.equations.every(eq => eq[0] * x + eq[1] * pressure + eq[2] <= tolerance);
        result.support_check_method = "exact_scaled_L_logP0_hull_at_current_interpolated_pressure";
      } else {
        result.in_training_hull = fraction === 0 ? table.training_hull_supported[i] :
                                  fraction === 1 ? table.training_hull_supported[i + 1] :
                                  table.training_hull_supported[i] && table.training_hull_supported[i + 1];
        result.support_check_method = "neighbor_sample_flags_only_no_continuous_hull_guarantee";
      }
      result.training_hull_supported = result.in_training_hull;
      result.throat_height_mm = 2000 * minimum(result.wall_heights_m);
      Object.assign(result, coreMetrics(this.crossROM, result.cross_coefficients, result.cross_width_m,
                                        result.cross_wall_heights_m, this.config.core_half_width_m));
      return result;
    }
    _bounds(mode = this.mode) {
      if (!this.supportBounds) {
        const first = this._fieldAt(this.currentCase.L_min_mm);
        const last = this._fieldAt(this.currentCase.L_max_mm);
        this.supportBounds = {
          position: [Math.min(first.throat_height_mm, last.throat_height_mm), Math.max(first.throat_height_mm, last.throat_height_mm)],
          mach: [Math.min(first.core_mach, last.core_mach), Math.max(first.core_mach, last.core_mach)],
        };
      }
      return this.supportBounds[mode];
    }
    _inverse(value, mode) {
      const bounds = this._bounds(mode);
      if (value < bounds[0] - 1e-10 || value > bounds[1] + 1e-10) {
        throw new RangeError(`Command outside planned ${mode} path range [${bounds[0]}, ${bounds[1]}]`);
      }
      // Refine against the actual interpolated geometry/field functional.
      // A dense scalar table alone would differ slightly when geometry varies.
      let low = this.currentCase.L_min_mm, high = this.currentCase.L_max_mm;
      const key = mode === "position" ? "throat_height_mm" : "core_mach";
      const sign = mode === "mach" ? -1 : Math.sign(this.currentCase.samples.throat_height_mm.at(-1) - this.currentCase.samples.throat_height_mm[0]);
      for (let iteration = 0; iteration < 48; iteration++) {
        const mid = 0.5 * (low + high);
        const observed = this._fieldAt(mid)[key];
        if (sign * (observed - value) < 0) low = mid;
        else high = mid;
      }
      return 0.5 * (low + high);
    }
    _machGain(length) {
      const table = this.currentCase.samples, i = bracket(table.L_mm, length);
      const gain = (table.core_mach[i + 1] - table.core_mach[i]) / (table.L_mm[i + 1] - table.L_mm[i]);
      if (!Number.isFinite(gain) || gain >= 0) throw new RangeError("Invalid nonnegative local Mach gain");
      return gain;
    }
    _error() {
      return this.mode === "position" ? this.commandL - this.actualL :
             (this.command - this.measuredMach) / this.gain;
    }
    _gains() {
      return this.mode === "position" ? [this.config.position_kp_per_s, this.config.position_ki_per_s2] :
             [this.config.mach_kp_per_s, this.config.mach_ki_per_s2];
    }
    getTuning() {
      const [kp, ki] = this._gains();
      return {
        mode: this.mode,
        max_speed_mm_s: this.config.max_speed_mm_s,
        kp_per_s: kp,
        ki_per_s2: ki,
        defaults: {
          max_speed_mm_s: this.initialTuning.max_speed_mm_s,
          kp_per_s: this.initialTuning[`${this.mode}_kp_per_s`],
          ki_per_s2: this.initialTuning[`${this.mode}_ki_per_s2`],
        },
      };
    }
    setTuning(settings) {
      if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
        throw new TypeError("Tuning requires an object of speed and gain values");
      }
      const allowed = ["max_speed_mm_s", "kp_per_s", "ki_per_s2"];
      const values = Object.create(null);
      for (const key of Object.keys(settings)) {
        if (!allowed.includes(key)) throw new TypeError(`Unknown live tuning setting: ${key}`);
        const value = finite(settings[key], key);
        if (value < 0 || (key !== "ki_per_s2" && value === 0)) {
          throw new RangeError(`${key} must be ${key === "ki_per_s2" ? "nonnegative" : "positive"}`);
        }
        values[key] = value;
      }
      const previous = this.getTuning();
      const speed = values.max_speed_mm_s === undefined ? previous.max_speed_mm_s : values.max_speed_mm_s;
      const kp = values.kp_per_s === undefined ? previous.kp_per_s : values.kp_per_s;
      const ki = values.ki_per_s2 === undefined ? previous.ki_per_s2 : values.ki_per_s2;
      const gainsChanged = kp !== previous.kp_per_s || ki !== previous.ki_per_s2;
      if (!gainsChanged && speed === previous.max_speed_mm_s) return this.getState();

      let command = clip(this.velocityCommand, -speed, speed);
      let integral = this.integral;
      let raw = command;
      if (ki === 0 || gainsChanged) {
        const proportional = finite(kp * this._error(), "proportional velocity");
        // PI changes preserve the preceding bounded command. I=0 instead means
        // pure P: retaining a cancelling integral bias could prevent all motion.
        integral = ki === 0 ? 0 : finite(command - proportional, "integral velocity");
        raw = ki === 0 ? proportional : command;
        command = clip(raw, -speed, speed);
      }
      const positionLimited = (this.actualL <= this.currentCase.L_min_mm && command < 0) ||
                              (this.actualL >= this.currentCase.L_max_mm && command > 0);
      if (positionLimited) command = 0;
      const velocity = clip(this.velocity, -speed, speed);
      const speedLimited = raw !== clip(raw, -speed, speed) ||
                           this.velocityCommand !== clip(this.velocityCommand, -speed, speed) ||
                           velocity !== this.velocity;
      // Commit only after validating every supplied and derived value.
      this.config.max_speed_mm_s = speed;
      this.config[`${this.mode}_kp_per_s`] = kp;
      this.config[`${this.mode}_ki_per_s2`] = ki;
      this.integral = integral;
      this.velocity = velocity;
      this.velocityCommand = command;
      this.positionLimited = positionLimited;
      this.speedLimited = speedLimited;
      this.status = positionLimited ? "position_limited" : speedLimited ? "speed_limited" : "tuning_changed";
      return this.getState();
    }
    reset(caseIndex = this.caseIndex, mode = this.mode) {
      if (!Number.isInteger(caseIndex) || caseIndex < 0 || caseIndex >= this.cases.length) throw new RangeError("Unknown operating-case index");
      this._validateMode(mode);
      this.caseIndex = caseIndex;
      this.currentCase = this.cases[caseIndex];
      this.supportBounds = null;
      this.mode = mode;
      this.time = 0;
      this.actualL = this.currentCase.L_initial_mm;
      this.velocity = 0;
      this.velocityCommand = 0;
      this.integral = 0;
      this.field = this._fieldAt(this.actualL);
      this.measuredMach = this.field.core_mach;
      this.command = mode === "position" ? this.field.throat_height_mm : this.field.core_mach;
      this.commandL = this.actualL;
      this.commandField = this.field;
      this.gain = this._machGain(this.commandL);
      this.positionLimited = this.speedLimited = false;
      this.status = "ready";
      return this.getState();
    }
    setCommand(value) {
      finite(value, "command");
      const length = this._inverse(value, this.mode);
      const gain = this._machGain(length);
      this.command = value;
      this.commandL = length;
      this.commandField = this._fieldAt(length);
      this.gain = gain;
      this.status = "tracking";
      return this.getState();
    }
    setMode(mode) {
      this._validateMode(mode);
      if (mode === this.mode) return this.getState();
      this.mode = mode;
      this.command = mode === "position" ? this.field.throat_height_mm : this.field.core_mach;
      this.commandL = this.actualL;
      this.commandField = this.field;
      this.gain = this._machGain(this.commandL);
      const [kp, ki] = this._gains();
      if (ki === 0) {
        this.integral = 0;
        const raw = kp * this._error();
        this.velocityCommand = clip(raw, -this.config.max_speed_mm_s, this.config.max_speed_mm_s);
        this.speedLimited = this.velocityCommand !== raw;
        this.positionLimited = (this.actualL <= this.currentCase.L_min_mm && this.velocityCommand < 0) ||
                               (this.actualL >= this.currentCase.L_max_mm && this.velocityCommand > 0);
        if (this.positionLimited) this.velocityCommand = 0;
      } else {
        // Preserve the preceding velocity command while changing feedback units.
        this.integral = this.velocityCommand - kp * this._error();
      }
      this.status = "mode_changed";
      return this.getState();
    }
    step(ticks = 1) {
      if (!Number.isInteger(ticks) || ticks < 0 || ticks > 100000) throw new RangeError("step requires an integer tick count from 0 to 100000");
      const dt = this.config.sample_time_s;
      const actuatorAlpha = -Math.expm1(-dt / this.config.actuator_velocity_time_s);
      const sensorAlpha = this.config.measurement_filter_time_s === 0 ? 1 :
                          -Math.expm1(-dt / this.config.measurement_filter_time_s);
      const antiWindupAlpha = -Math.expm1(-dt / this.config.anti_windup_time_s);
      for (let tick = 0; tick < ticks; tick++) {
        const [kp, ki] = this._gains();
        const error = this._error();
        const raw = kp * error + (ki === 0 ? 0 : this.integral);
        let command = clip(raw, -this.config.max_speed_mm_s, this.config.max_speed_mm_s);
        this.speedLimited = command !== raw;
        this.positionLimited = (this.actualL <= this.currentCase.L_min_mm && command < 0) ||
                               (this.actualL >= this.currentCase.L_max_mm && command > 0);
        if (this.positionLimited) command = 0;
        let velocity = this.velocity + actuatorAlpha * (command - this.velocity);
        const proposed = this.actualL + dt * velocity;
        const actual = clip(proposed, this.currentCase.L_min_mm, this.currentCase.L_max_mm);
        if (actual !== proposed) {
          this.positionLimited = true;
          velocity = 0;
          command = 0;
        }
        if (ki === 0) this.integral = 0;
        else this.integral += dt * ki * error + antiWindupAlpha * (command - raw);
        this.actualL = actual;
        this.velocity = velocity;
        this.velocityCommand = command;
        this.field = this._fieldAt(actual);
        this.measuredMach += sensorAlpha * (this.field.core_mach - this.measuredMach);
        this.time += dt;
        this.status = this.positionLimited ? "position_limited" : this.speedLimited ? "speed_limited" : "tracking";
      }
      return this.getState();
    }
    getState() {
      const bounds = this._bounds();
      const throatBounds = this._bounds("position"), machBounds = this._bounds("mach");
      const commandField = this.commandField;
      const targetMach = this.mode === "mach" ? this.command : commandField.core_mach;
      return {
        time_s: this.time, sample_time_s: this.config.sample_time_s,
        case_index: this.caseIndex, case_id: this.currentCase.id, mode: this.mode,
        operating_target_mach: this.currentCase.target_mach,
        P0_Pa: this.field.P0_Pa, actual_P0_Pa: this.field.P0_Pa,
        P0_command_Pa: commandField.P0_Pa, command_P0_Pa: commandField.P0_Pa,
        path_mach: this.field.schedule_mach, command_path_mach: commandField.schedule_mach,
        in_training_hull: this.field.in_training_hull,
        command_in_training_hull: commandField.in_training_hull,
        support_check_method: this.field.support_check_method,
        command: this.command, command_unit: this.mode === "position" ? "mm_full_throat_height" : "Mach",
        command_min: bounds[0], command_max: bounds[1],
        actual_L_mm: this.actualL, command_L_mm: this.commandL,
        actual_throat_height_mm: this.field.throat_height_mm,
        command_throat_height_mm: commandField.throat_height_mm,
        true_core_mach: this.field.core_mach, measured_core_mach: this.measuredMach,
        target_core_mach: targetMach, gain_mach_per_mm: this.gain,
        velocity_mm_s: this.velocity, velocity_command_mm_s: this.velocityCommand,
        integral_velocity_mm_s: this.integral,
        error_L_mm: this.commandL - this.actualL, error_mach: targetMach - this.measuredMach,
        equivalent_control_error_mm: this._error(),
        position_limited: this.positionLimited, speed_limited: this.speedLimited, status: this.status,
        L_min_mm: this.currentCase.L_min_mm, L_max_mm: this.currentCase.L_max_mm,
        throat_min_mm: throatBounds[0], throat_max_mm: throatBounds[1],
        mach_min: machBounds[0], mach_max: machBounds[1],
        feedback_definition: "area_mean_of_current_cross_ROM_over_central_100x100mm_square_before_sensor_filter",
        gain_definition: "dM_dL_along_coordinated_L_P0_path_not_fixed_pressure_partial_derivative",
        pressure_policy: "ideal_synchronous_design_schedule_at_current_actual_L; no_independent_pressure_lag",
        dynamics_scope: "assumed_actuator_velocity_lag_and_sensor_filter_with_ideal_pressure_schedule; quasi_steady_fields",
      };
    }
    getFieldState(length = this.actualL) {
      return this._fieldAt(length);
    }
  }

  return {Simulator, DEFAULT_CONFIG, fieldCoreMean};
});
