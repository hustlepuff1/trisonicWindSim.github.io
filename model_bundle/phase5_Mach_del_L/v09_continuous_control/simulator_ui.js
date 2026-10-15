/* Embedded 2D live controls and canvas rendering, with a lazy-loaded 3D display. */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "—";
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const payload = JSON.parse($("simulatorPayload").textContent);
  const longROM = payload.field_roms.longitudinal, crossROM = payload.field_roms.crossplane;
  const simulator = new NozzleControl.Simulator(payload, 0, "mach");
  const volumeViewer = new NozzleVolume.Viewer();
  let state = simulator.getState(), running = true, displayMode = "absolute", heightScale = 1;
  let history = [], baseline = null, fieldCache = null, lastFrame = null, accumulator = 0, lastRender = 0;
  let dirty = true, commandDirty = true, presentation = false;
  const colorStops = [[0, [45, 53, 125]], [.25, [34, 142, 177]], [.5, [94, 197, 131]], [.75, [242, 216, 78]], [1, [194, 52, 46]]];
  const deltaStops = [[0, [38, 87, 164]], [.5, [248, 250, 251]], [1, [190, 48, 54]]];
  function makePalette(stops) {
    return Array.from({length: 512}, (_, i) => {
      const value = i / 511;
      let j = 0;
      while (j < stops.length - 2 && value > stops[j + 1][0]) j++;
      const t = (value - stops[j][0]) / (stops[j + 1][0] - stops[j][0]);
      return "rgb(" + stops[j][1].map((c, k) => Math.round(c + t * (stops[j + 1][1][k] - c))).join(",") + ")";
    });
  }
  const palettes = {absolute: makePalette(colorStops), delta: makePalette(deltaStops)};
  const gradients = {
    absolute: "linear-gradient(90deg,rgb(45,53,125),rgb(34,142,177),rgb(94,197,131),rgb(242,216,78),rgb(194,52,46))",
    delta: "linear-gradient(90deg,rgb(38,87,164),rgb(248,250,251),rgb(190,48,54))"
  };
  function errorMessage(error) {
    $("errorBox").textContent = "입력 또는 계산 오류: " + error.message;
    $("errorBox").style.display = "block";
  }
  function safe(action) {
    return (...args) => { try { $("errorBox").style.display = "none"; const result = action(...args); if (result && typeof result.catch === "function") result.catch(errorMessage); return result; }
      catch (error) { errorMessage(error); return null; } };
  }
  function reconstruct(rom, coefficients) {
    const result = Float64Array.from(rom.mean);
    for (let r = 0; r < coefficients.length; r++) {
      const mode = rom.modes[r], coefficient = coefficients[r];
      for (let i = 0; i < result.length; i++) result[i] += coefficient * mode[i];
    }
    return result;
  }
  function obtainFields() {
    if (!fieldCache || fieldCache.L !== state.actual_L_mm || fieldCache.commandL !== state.command_L_mm) {
      const actual = simulator.getFieldState(), command = simulator.getFieldState(state.command_L_mm);
      fieldCache = {L: state.actual_L_mm, commandL: state.command_L_mm, actual, command,
        longitudinal: reconstruct(longROM, actual.long_coefficients), crossplane: reconstruct(crossROM, actual.cross_coefficients)};
    }
    return fieldCache;
  }
  function record(event = "sample") { history.push({...state, event}); }
  function reset(index = state.case_index) {
    state = simulator.reset(Number(index), state.mode);
    fieldCache = null;
    const fields = obtainFields();
    baseline = {longitudinal: Float64Array.from(fields.longitudinal), crossplane: Float64Array.from(fields.crossplane)};
    history = []; record("reset"); accumulator = 0;
    dirty = commandDirty = true;
    $("commandMessage").textContent = "연속 운전 경로의 시작점으로 초기화했습니다.";
    render(); return state;
  }
  function setMode(mode) {
    state = simulator.setMode(mode); record("mode_change");
    commandDirty = dirty = true;
    $("commandMessage").textContent = "현재 위치·속도를 유지하며 모드를 바꿨습니다.";
    render(); return state;
  }
  function setCommand(value) {
    state = simulator.setCommand(Number(value)); record("command_change");
    commandDirty = dirty = true;
    $("commandMessage").textContent = running ? "새 명령을 추종합니다." : "명령이 설정되었습니다. 재개하면 위치가 움직입니다.";
    render(); return state;
  }
  function advance(ticks) {
    for (let i = 0; i < ticks; i++) { state = simulator.step(1); record(); }
    dirty = true; return state;
  }
  function setRunning(value) { running = Boolean(value); accumulator = 0; lastFrame = null; dirty = true; render(); return state; }
  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(150, rect.width), height = Math.max(140, rect.height), ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f7fafb"; ctx.fillRect(0, 0, width, height);
    ctx.lineJoin = "round"; ctx.font = '11px "UnDotum",Arial,sans-serif';
    return {ctx, width, height};
  }
  function fieldColor(value, kind) {
    const entry = payload.cases[state.case_index];
    const limit = kind === "long" ? entry.long_delta_limit : entry.cross_delta_limit;
    const low = displayMode === "delta" ? -limit : entry.color_min;
    const high = displayMode === "delta" ? limit : entry.color_max;
    return palettes[displayMode][Math.round(clamp((value - low) / (high - low), 0, 1) * 511)];
  }
  function quad(ctx, points, color) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.closePath(); ctx.fill();
    // Cover subpixel seams between adjacent cells without changing field values.
    ctx.strokeStyle = color; ctx.lineWidth = .45; ctx.stroke();
  }
  function line(ctx, points, color, dash = [], width = 1.5) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
    points.forEach((point, i) => i ? ctx.lineTo(...point) : ctx.moveTo(...point));
    ctx.stroke(); ctx.setLineDash([]);
  }
  function drawLong(fields) {
    const {ctx, width, height} = setupCanvas($("longCanvas"));
    const x = longROM.x_edges_m, eta = longROM.eta_edges, wall = fields.actual.wall_heights_m;
    const referenceWalls = payload.cases[state.case_index].samples.wall_heights_m;
    let maxWall = 0;
    for (const row of referenceWalls) for (const value of row) maxWall = Math.max(maxWall, value);
    const left = 51, right = 18, top = 25, bottom = 42;
    const xMin = x[0], xMax = x[x.length - 1];
    const scale = Math.min((width - left - right) / (xMax - xMin), (height - top - bottom) / (maxWall * heightScale));
    const plotWidth = (xMax - xMin) * scale, plotHeight = maxWall * scale * heightScale;
    const x0 = left + (width - left - right - plotWidth) / 2, y0 = top + (height - top - bottom + plotHeight) / 2;
    const X = value => x0 + (value - xMin) * scale, Z = value => y0 - value * scale * heightScale;
    for (let j = 0; j < longROM.neta; j++) for (let i = 0; i < longROM.nx; i++) {
      const k = j * longROM.nx + i, value = fields.longitudinal[k] - (displayMode === "delta" ? baseline.longitudinal[k] : 0);
      quad(ctx, [[X(x[i]), Z(eta[j] * wall[i])], [X(x[i + 1]), Z(eta[j] * wall[i + 1])],
        [X(x[i + 1]), Z(eta[j + 1] * wall[i + 1])], [X(x[i]), Z(eta[j + 1] * wall[i])]], fieldColor(value, "long"));
    }
    line(ctx, x.map((v, i) => [X(v), Z(wall[i])]), "#263d49", [], 1.7);
    line(ctx, x.map((v, i) => [X(v), Z(fields.command.wall_heights_m[i])]), "#dc771c", [6, 4], 1.8);
    line(ctx, [[X(xMin), Z(0)], [X(xMax), Z(0)]], "#526975", [], 1);
    ctx.fillStyle = "#526975"; ctx.textAlign = "center";
    const ticks = [xMin, ...Array.from({length: Math.floor(xMax) - Math.ceil(xMin) + 1}, (_, i) => Math.ceil(xMin) + i)];
    ticks.filter((v, i) => i === 0 || v - xMin > .2).forEach(value => {
      line(ctx, [[X(value), y0], [X(value), y0 + 4]], "#637b86", [], 1);
      ctx.fillText(fmt(value, value === xMin ? 1 : 0), X(value), y0 + 18);
    });
    ctx.fillText("x [m]", x0 + plotWidth / 2, height - 5);
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const value = maxWall * i / 4;
      line(ctx, [[x0 - 4, Z(value)], [x0, Z(value)]], "#637b86", [], 1);
      ctx.fillText(fmt(value, 2), x0 - 7, Z(value) + 4);
    }
    ctx.textAlign = "left"; ctx.fillText("z [m]", x0 - 37, top - 6);
    ctx.fillStyle = "#526975"; ctx.fillText("높이 " + heightScale + "× 표시 · 축 눈금은 실제 길이", x0, 14);
    $("longGeometryLabel").textContent = "y = 0 · 상반면 z ≥ 0 · 높이 " + heightScale + "×";
    $("longSummary").textContent = "전체 목높이 " + fmt(state.actual_throat_height_mm, 3) + " mm · L " + fmt(state.actual_L_mm, 4) + " mm";
  }
  function drawCross(fields) {
    const {ctx, width, height} = setupCanvas($("crossCanvas"));
    const wall = fields.actual.cross_wall_heights_m, halfWidth = fields.actual.cross_width_m;
    const allWalls = payload.cases[state.case_index].samples.cross_wall_heights_m;
    let maxHeight = 0; for (const row of allWalls) for (const v of row) maxHeight = Math.max(maxHeight, v);
    const scale = Math.min((width - 86) / (2 * halfWidth), (height - 58) / (2 * maxHeight));
    const cx = width / 2 + 12, cy = (height - 15) / 2;
    const Y = value => cx + value * scale, Z = value => cy - value * scale;
    const ye = crossROM.y_eta_edges, ze = crossROM.z_eta_edges;
    for (let j = 0; j < crossROM.nz; j++) for (let i = 0; i < crossROM.ny; i++) {
      const k = j * crossROM.ny + i, value = fields.crossplane[k] - (displayMode === "delta" ? baseline.crossplane[k] : 0);
      const color = fieldColor(value, "cross");
      for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
        quad(ctx, [[Y(sy * ye[i] * halfWidth), Z(sz * ze[j] * wall[i])],
          [Y(sy * ye[i + 1] * halfWidth), Z(sz * ze[j] * wall[i + 1])],
          [Y(sy * ye[i + 1] * halfWidth), Z(sz * ze[j + 1] * wall[i + 1])],
          [Y(sy * ye[i] * halfWidth), Z(sz * ze[j + 1] * wall[i])]], color);
      }
    }
    for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
      line(ctx, ye.map((v, i) => [Y(sy * v * halfWidth), Z(sz * wall[i])]), "#263d49", [], 1.5);
      line(ctx, ye.map((v, i) => [Y(sy * v * fields.command.cross_width_m), Z(sz * fields.command.cross_wall_heights_m[i])]), "#dc771c", [5, 3], 1.5);
    }
    for (const sy of [-1, 1]) line(ctx, [[Y(sy * halfWidth), Z(-wall.at(-1))], [Y(sy * halfWidth), Z(wall.at(-1))]], "#263d49");
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 3.8; ctx.strokeRect(Y(-.05), Z(.05), .1 * scale, .1 * scale);
    ctx.strokeStyle = "#192b35"; ctx.lineWidth = 1.7; ctx.setLineDash([5, 3]); ctx.strokeRect(Y(-.05), Z(.05), .1 * scale, .1 * scale); ctx.setLineDash([]);
    ctx.fillStyle = "#526975"; ctx.textAlign = "center";
    for (const value of [-halfWidth, 0, halfWidth]) ctx.fillText(fmt(value, 2), Y(value), Z(-maxHeight) + 17);
    ctx.fillText("y [m]", cx, height - 3); ctx.textAlign = "right";
    for (const value of [-maxHeight, 0, maxHeight]) ctx.fillText(fmt(value, 2), Y(-halfWidth) - 7, Z(value) + 4);
    ctx.textAlign = "left"; ctx.fillText("z [m]", Y(-halfWidth) - 28, 14);
    $("crossSize").textContent = "전체 폭 " + fmt(2 * halfWidth * 1000, 1) + " mm · 중앙 높이 " + fmt(2 * wall[0] * 1000, 2) + " mm";
  }
  function drawTrace(canvas, series, unit) {
    const {ctx, width, height} = setupCanvas(canvas), left = 67, right = 16, top = 18, bottom = 35;
    const end = Math.max(10, state.time_s), begin = Math.max(0, end - 120);
    const visible = history.filter(row => row.time_s >= begin);
    let low = Infinity, high = -Infinity;
    for (const row of visible) for (const trace of series) { const v = trace.value(row); if (Number.isFinite(v)) { low = Math.min(low, v); high = Math.max(high, v); } }
    const pad = Math.max((high - low) * .15, unit === "Mach" ? .0003 : .01);
    low -= pad; high += pad;
    const X = t => left + (t - begin) / (end - begin) * (width - left - right);
    const Y = value => height - bottom - (value - low) / (high - low) * (height - top - bottom);
    ctx.font = '10px "UnDotum",Arial,sans-serif'; ctx.fillStyle = "#637b86";
    for (let i = 0; i <= 4; i++) {
      const v = low + (high - low) * i / 4, t = begin + (end - begin) * i / 4;
      line(ctx, [[left, Y(v)], [width - right, Y(v)]], "#e1e9ed", [], .8);
      ctx.textAlign = "right"; ctx.fillText(fmt(v, unit === "Mach" ? 4 : 2), left - 7, Y(v) + 3);
      ctx.textAlign = "center"; ctx.fillText(fmt(t, 0), X(t), height - bottom + 15);
    }
    ctx.textAlign = "left"; ctx.fillText(unit, 5, 12);
    ctx.textAlign = "right"; ctx.fillText("t [s]", width - 4, height - 2);
    if (begin > 0) { ctx.textAlign = "left"; ctx.fillText("최근 120 s 표시 · CSV는 전체 이력", left, 12); }
    const stride = Math.max(1, Math.floor(visible.length / (width * 2)));
    for (const trace of series) {
      ctx.beginPath(); ctx.strokeStyle = trace.color; ctx.lineWidth = 1.65; ctx.setLineDash(trace.dash || []);
      let connected = false;
      for (let i = 0; i < visible.length; i++) {
        if (i % stride !== 0 && i !== visible.length - 1 && visible[i].event === "sample") continue;
        const value = trace.value(visible[i]);
        if (!Number.isFinite(value)) { connected = false; continue; }
        if (connected) ctx.lineTo(X(visible[i].time_s), Y(value)); else ctx.moveTo(X(visible[i].time_s), Y(value));
        connected = true;
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    line(ctx, [[X(state.time_s), top], [X(state.time_s), height - bottom]], "#9eb3bd", [3, 3], 1);
  }
  function updateControls() {
    const position = state.mode === "position", digits = position ? 4 : 6;
    // Include exact design-endpoint text entries when integration differs only by roundoff.
    const inputMin = !position && Math.abs(state.command_min - 1.5) <= 1e-10 ? Math.min(state.command_min, 1.5) : state.command_min;
    const inputMax = !position && Math.abs(state.command_max - 3.5) <= 1e-10 ? Math.max(state.command_max, 3.5) : state.command_max;
    $("modeSelect").value = state.mode;
    $("commandTitle").textContent = position ? "목표 전체 목높이" : "목표 중앙 평균 Mach";
    $("commandUnit").textContent = position ? "상·하 벽 사이 거리 [mm]" : "시험부 중앙 100×100 mm 면적 평균";
    for (const id of ["commandSlider", "commandNumber"]) {
      const input = $(id); input.min = inputMin; input.max = inputMax;
      input.step = "any"; input.value = String(state.command);
    }
    $("rangeMin").textContent = fmt(state.command_min, digits) + (position ? " mm" : "");
    $("rangeMax").textContent = fmt(state.command_max, digits) + (position ? " mm" : "");
    commandDirty = false;
  }
  function updateMetrics() {
    const position = state.mode === "position";
    $("targetMetricLabel").textContent = position ? "목표 전체 목높이 [mm]" : "목표 중앙 평균 Mach";
    $("targetMetric").textContent = fmt(state.command, position ? 3 : 5);
    $("heightMetric").textContent = fmt(state.actual_throat_height_mm, 3) + " mm";
    $("lengthMetric").textContent = fmt(state.actual_L_mm, 4) + " mm";
    $("coreMetric").textContent = fmt(state.true_core_mach, 5);
    $("sensorMetric").textContent = fmt(state.measured_core_mach, 5);
    $("errorMetricLabel").textContent = position ? "목높이 오차 [mm]" : "목표 − 센서 Mach";
    $("errorMetric").textContent = fmt(position ? state.command - state.actual_throat_height_mm : state.command - state.measured_core_mach, position ? 4 : 6);
    $("pressureBadge").textContent = "P₀ 자동 연동";
    $("actualPressureMetric").textContent = fmt(state.P0_Pa / 1000, 2) + " kPa";
    $("targetPressureMetric").textContent = fmt((state.P0_command_Pa ?? state.command_P0_Pa) / 1000, 2) + " kPa";
    const supported = state.in_training_hull === true;
    $("supportBadge").textContent = supported ? "현재 학습범위 내부" : "현재 새 CFD 검증 필요";
    $("supportBadge").className = "badge " + (supported ? "supported" : "review");
    $("supportBadge").title = "현재/명령 입력의 학습 입력 convex hull 포함 여부. 내부도 독립 CFD 검증 완료를 뜻하지 않습니다. 명령 위치: " + (state.command_in_training_hull ? "학습범위 내부" : "새 CFD 검증 필요");
    $("runningBadge").textContent = !running ? "일시정지" : state.speed_limited ? "계산 중 · 속도 제한" : state.position_limited ? "계산 중 · 위치 제한" : "계산 중";
    $("pauseButton").textContent = running ? "일시정지" : "재개";
    $("timeLabel").textContent = "t = " + fmt(state.time_s, 2) + " s";
    $("statusFooter").textContent = "Δt " + fmt(state.sample_time_s, 2) + " s · |dL/dt| ≤ " + fmt(simulator.config.max_speed_mm_s, 2) + " mm/s · 현재 " + fmt(state.velocity_mm_s, 4) + " mm/s";
    document.body.dataset.case = state.case_id; document.body.dataset.mode = state.mode;
    document.body.dataset.time = fmt(state.time_s, 3); document.body.dataset.actualL = fmt(state.actual_L_mm, 9);
  }
  function updateLegends() {
    const entry = payload.cases[state.case_index];
    for (const kind of ["long", "cross"]) {
      const limit = entry[kind + "_delta_limit"], low = displayMode === "delta" ? -limit : entry.color_min;
      const high = displayMode === "delta" ? limit : entry.color_max;
      $(kind + "LegendTitle").textContent = displayMode === "delta" ? "ΔMach" : "Mach";
      $(kind + "Min").textContent = fmt(low, displayMode === "delta" ? 4 : 2);
      $(kind + "Max").textContent = fmt(high, displayMode === "delta" ? 4 : 2);
      $(kind + "Colorbar").style.background = gradients[displayMode];
    }
    $("longRange").textContent = displayMode === "delta" ? "같은 정규화 좌표의 초기 ROM 대비 차이" : "전 구간 고정 색범위";
  }
  function render() {
    if (!baseline) return;
    if (commandDirty) updateControls(); updateMetrics(); updateLegends();
    if (volumeViewer.view !== "3d") { const fields = obtainFields(); drawLong(fields); drawCross(fields); }
    volumeViewer.update(state);
    const coordinate = $("positionTraceSelect").value === "length" ? "L_mm" : "throat_height_mm";
    drawTrace($("positionCanvas"), [
      {value: row => row["command_" + coordinate], color: "#db781c", dash: [5, 3]},
      {value: row => row["actual_" + coordinate], color: "#2169a2"}], "mm");
    drawTrace($("machCanvas"), [
      {value: row => row.true_core_mach, color: "#2169a2"},
      {value: row => row.measured_core_mach, color: "#db781c", dash: [4, 3]},
      {value: row => row.mode === "mach" ? row.command : null, color: "#237b51", dash: [6, 4]}], "Mach");
    dirty = false;
  }
  function csvText() {
    const columns = ["event", "time_s", "case_id", "mode", "P0_Pa", "P0_command_Pa", "actual_P0_Pa", "command_P0_Pa", "path_mach", "command_path_mach", "in_training_hull", "command_in_training_hull", "operating_target_mach", "command", "command_unit",
      "command_L_mm", "actual_L_mm", "command_throat_height_mm", "actual_throat_height_mm", "target_core_mach",
      "true_core_mach", "measured_core_mach", "velocity_command_mm_s", "velocity_mm_s", "integral_velocity_mm_s",
      "error_L_mm", "error_mach", "speed_limited", "position_limited", "status"];
    const quote = value => '"' + String(value === undefined ? "" : value).replace(/"/g, '""') + '"';
    return "\uFEFF" + columns.join(",") + "\n" + history.map(row => columns.map(key => quote(row[key])).join(",")).join("\n") + "\n";
  }
  function download(blob, filename) {
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function exportCSV() { const text = csvText(); download(new Blob([text], {type: "text/csv;charset=utf-8"}), "nozzle_continuous_" + state.mode + "_history.csv"); return text; }
  function exportPNG() {
    render();
    if (volumeViewer.active && volumeViewer.ready) {
      const out = volumeViewer.exportPNG();
      out.toBlob(blob => { if (blob) download(blob, "nozzle_volume_mach_t" + fmt(state.time_s, 2) + ".png"); }, "image/png");
      return out;
    }
    const out = document.createElement("canvas"); out.width = 2000; out.height = 1280;
    const ctx = out.getContext("2d"); ctx.fillStyle = "#eef3f5"; ctx.fillRect(0, 0, out.width, out.height);
    ctx.fillStyle = "#16323e"; ctx.font = 'bold 31px "UnDotum",Arial';
    ctx.fillText("가변노즐 · " + (state.mode === "position" ? "전체 목높이 제어" : "중앙 평균 Mach 제어"), 40, 53);
    ctx.font = '21px "UnDotum",Arial';
    ctx.fillText("준정상 · 이상적 압력추종 | 설계 질량유량 80 kg/s | 현재 P₀ " + fmt(state.P0_Pa / 1000, 2) + " kPa | 명령 P₀ " + fmt((state.P0_command_Pa ?? state.command_P0_Pa) / 1000, 2) + " kPa | t " + fmt(state.time_s, 2) + " s", 40, 91);
    ctx.font = '19px "UnDotum",Arial';
    ctx.fillText("명령 " + fmt(state.command, state.mode === "position" ? 4 : 6) + (state.mode === "position" ? " mm" : " Mach") + " | 실제 목높이 " + fmt(state.actual_throat_height_mm, 4) + " mm | L " + fmt(state.actual_L_mm, 5) + " mm | 중앙 Mach " + fmt(state.true_core_mach, 6) + " | 센서 " + fmt(state.measured_core_mach, 6), 40, 128);
    const paste = (id, x, y, w, h, title) => {
      ctx.fillStyle = "white"; ctx.fillRect(x, y, w, h); ctx.fillStyle = "#16323e"; ctx.font = 'bold 21px "UnDotum",Arial'; ctx.fillText(title, x + 14, y + 28);
      const canvas = $(id), ratio = Math.min((w - 20) / canvas.width, (h - 52) / canvas.height);
      ctx.drawImage(canvas, x + (w - canvas.width * ratio) / 2, y + 43, canvas.width * ratio, canvas.height * ratio);
    };
    paste("longCanvas", 35, 154, 1235, 534, "노즐 y = 0 상반면 · 높이 " + heightScale + "×");
    paste("crossCanvas", 1285, 154, 680, 534, "시험부 x = 0 · ¼ CFD의 대칭 복원");
    const entry = payload.cases[state.case_index]; ctx.font = '18px "UnDotum",Arial'; ctx.fillStyle = "#526975";
    ctx.fillText(displayMode === "absolute" ? "Mach 고정 색범위: " + fmt(entry.color_min, 2) + " ~ " + fmt(entry.color_max, 2) : "초기화 대비 ΔMach | 종단면 ±" + fmt(entry.long_delta_limit, 5) + " · 단면 ±" + fmt(entry.cross_delta_limit, 5), 45, 720);
    const gradient = ctx.createLinearGradient(45, 0, 380, 0);
    (displayMode === "delta" ? deltaStops : colorStops).forEach(stop => gradient.addColorStop(stop[0], "rgb(" + stop[1].join(",") + ")"));
    ctx.fillStyle = gradient; ctx.fillRect(45, 733, 335, 16); ctx.fillStyle = "#526975";
    ctx.fillText("검은 선: 실제 형상 · 주황 점선: 명령 형상 · 단면 중앙 점선: 100×100 mm 피드백 영역", 418, 750);
    paste("positionCanvas", 35, 774, 955, 360, "명령(주황) · 실제 위치(파랑)");
    paste("machCanvas", 1010, 774, 955, 360, "ROM 중앙 Mach(파랑) · 센서(주황) · Mach 목표(초록)");
    ctx.fillStyle = "#526975"; ctx.font = '18px "UnDotum",Arial';
    ctx.fillText("정상 CFD 기반 ROM을 현재 L에서 재구성합니다. 구동기·센서 동특성은 가정값이며 실제 비정상 CFD/풍동 실험 결과가 아닙니다.", 40, 1183);
    ctx.fillText("P₀는 현재 위치의 설계 스케줄을 지연 없이 따릅니다. 80 kg/s는 설계 목표이며 실제 유량 유지 검증 결과가 아닙니다.", 40, 1213);
    ctx.fillText("현재 입력: " + (state.in_training_hull ? "학습범위 내부" : "새 CFD 검증 필요") + " · ΔMach는 정규화 좌표의 초기 ROM 대비 차이 · 명령/시간 이력은 CSV로 저장됩니다.", 40, 1243);
    out.toBlob(blob => { if (blob) download(blob, "nozzle_continuous_" + state.mode + "_t" + fmt(state.time_s, 2) + ".png"); }, "image/png");
    return out;
  }
  function setDisplayMode(mode) { if (!["absolute", "delta"].includes(mode)) throw new Error("Unknown field display mode"); displayMode = mode; $("fieldModeSelect").value = mode; render(); }
  function setPresentation(value) { presentation = Boolean(value); document.body.classList.toggle("presentation", presentation); $("presentationButton").textContent = presentation ? "기본 화면" : "발표 화면"; render(); }
  document.querySelectorAll("[data-mach]").forEach(button => button.addEventListener("click", safe(() => {
    if (state.mode !== "mach") setMode("mach");
    setCommand(Number(button.dataset.mach));
  })));
  $("modeSelect").addEventListener("change", safe(event => setMode(event.target.value)));
  $("commandSlider").addEventListener("input", safe(event => setCommand(event.target.value)));
  $("commandNumber").addEventListener("change", safe(event => setCommand(event.target.value)));
  $("pauseButton").addEventListener("click", () => setRunning(!running));
  $("resetButton").addEventListener("click", safe(() => reset()));
  $("lowerButton").addEventListener("click", safe(() => setCommand(state.command_min)));
  $("upperButton").addEventListener("click", safe(() => setCommand(state.command_max)));
  for (const [id, sign] of [["minusButton", -1], ["plusButton", 1]]) $(id).addEventListener("click", safe(() => setCommand(clamp(state.command + sign * .1 * (state.command_max - state.command_min), state.command_min, state.command_max))));
  $("currentButton").addEventListener("click", safe(() => setCommand(state.mode === "position" ? state.actual_throat_height_mm : state.true_core_mach)));
  $("fieldModeSelect").addEventListener("change", event => setDisplayMode(event.target.value));
  $("heightScaleSelect").addEventListener("change", event => { heightScale = Number(event.target.value); render(); });
  $("positionTraceSelect").addEventListener("change", render);
  $("presentationButton").addEventListener("click", () => setPresentation(!presentation));
  $("fullscreenButton").addEventListener("click", safe(async () => { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }));
  $("csvButton").addEventListener("click", exportCSV); $("pngButton").addEventListener("click", exportPNG);
  window.addEventListener("resize", () => { dirty = true; });
  window.nozzleSimulatorDebug = {
    getState: () => ({...state}), getFields: () => obtainFields(), getHistory: () => history.map(row => ({...row})),
    getVolumeState: () => volumeViewer.getState(), setView: view => volumeViewer.setView(view), setVolumeSlice: index => volumeViewer.setSlice(index),
    getCSV: csvText, exportPNG, exportCSV, setCase: reset, setMode, setCommand, setRunning,
    step: ticks => { advance(ticks); render(); return state; }, setDisplayMode, setPresentation,
    setHeightScale: value => { if (![1, 2, 4].includes(value)) throw new Error("Invalid height scale"); heightScale = value; $("heightScaleSelect").value = value; render(); },
    getPayloadSummary: () => ({case_count: payload.cases.length, long_cells: longROM.mean.length, cross_quarter_cells: crossROM.mean.length}), render
  };
  try {
    reset(0);
    const query = new URLSearchParams(window.location.search);
    if (query.has("case")) reset(Number(query.get("case")));
    if (query.has("mode")) setMode(query.get("mode"));
    if (query.has("fraction")) setCommand(state.command_min + clamp(Number(query.get("fraction")), 0, 1) * (state.command_max - state.command_min));
    if (query.has("command")) setCommand(Number(query.get("command")));
    if (query.has("seconds")) advance(Math.min(12000, Math.max(0, Math.round(Number(query.get("seconds")) / state.sample_time_s))));
    if (query.has("field")) setDisplayMode(query.get("field"));
    if (query.has("height")) window.nozzleSimulatorDebug.setHeightScale(Number(query.get("height")));
    if (query.get("presentation") === "1") setPresentation(true);
    if (query.get("paused") === "1") running = false;
    if (query.has("view")) volumeViewer.setView(query.get("view"));
    render(); document.body.dataset.ready = "true";
  } catch (error) { running = false; errorMessage(error); document.body.dataset.ready = "error"; }
  function animate(timestamp) {
    if (lastFrame !== null && running) {
      accumulator += Math.min(.25, (timestamp - lastFrame) / 1000) * Number($("speedSelect").value);
      const ticks = Math.floor(accumulator / state.sample_time_s);
      if (ticks) { try { advance(ticks); accumulator -= ticks * state.sample_time_s; }
        catch (error) { running = false; errorMessage(error); } }
    }
    lastFrame = timestamp;
    if (dirty && timestamp - lastRender >= 100) { render(); lastRender = timestamp; }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
})();
