/* Physical centerline profiles sampled from the current 3D display nodes.
 * No additional regression, field interpolation, or controller feedback. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.NozzleProfiles = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const axes = ["z", "y", "x"];
  const colors = {z: "#2169a2", y: "#237b65", x: "#b66a24"};
  const captions = {z: "시험부 x = 0 · y = 0", y: "시험부 x = 0 · z = 0", x: "노즐 중심선 · y = z = 0"};
  const zero = x => Math.abs(x) < 1e-6 ? 0 : x;
  const fmt = (x, n = 3) => Number.isFinite(x) ? x.toFixed(n) : "—";

  function sampleProfiles(shape, coordinates, field) {
    if (shape.length !== 3 || !shape.every(n => Number.isInteger(n) && n >= 2)) throw new Error("Invalid profile grid");
    const [nx, ny, nz] = shape, count = nx * ny * nz;
    if (coordinates.length !== count * 3 || field.length !== count) throw new Error("Incomplete profile field");
    const node = (i, j, k) => (i * ny + j) * nz + k;
    const point = (index, axis) => {
      const values = [coordinates[index * 3], coordinates[index * 3 + 1], coordinates[index * 3 + 2], field[index]];
      if (!values.every(Number.isFinite)) throw new Error("Non-finite profile sample");
      const [x, y, z, mach] = values;
      if ((axis !== "x" && Math.abs(x) > 1e-6) || (axis !== "y" && Math.abs(y) > 1e-6) || (axis !== "z" && Math.abs(z) > 1e-6)) throw new Error("Profile is not on the requested physical centerline");
      return {coordinate_m: zero(values["xyz".indexOf(axis)]), x_m: zero(x), y_m: zero(y), z_m: zero(z), mach};
    };
    const mirror = (points, axis) => points.concat(points.filter(p => p.coordinate_m !== 0).map(p => ({...p, coordinate_m: -p.coordinate_m, [axis + "_m"]: -p.coordinate_m}))).sort((a, b) => a.coordinate_m - b.coordinate_m);
    return {
      x: Array.from({length: nx}, (_, i) => point(node(i, 0, 0), "x")),
      y: mirror(Array.from({length: ny}, (_, j) => point(node(nx - 1, j, 0), "y")), "y"),
      z: mirror(Array.from({length: nz}, (_, k) => point(node(nx - 1, 0, k), "z")), "z")
    };
  }

  function toCSV(data) {
    if (!data) return "";
    const rows = ["source,time_s,actual_L_mm,P0_Pa,profile,coordinate_m,x_m,y_m,z_m,Mach"];
    for (const axis of axes) for (const p of data.lines[axis]) rows.push([data.source, data.time_s, data.actual_L_mm, data.P0_Pa, axis, p.coordinate_m, p.x_m, p.y_m, p.z_m, p.mach].join(","));
    return rows.join("\r\n") + "\r\n";
  }

  function limits(points) {
    const low = Math.min(...points.map(p => p.mach)), high = Math.max(...points.map(p => p.mach));
    // A minimum span avoids amplifying roundoff in nearly uniform profiles.
    const margin = Math.max(.025, (high - low) * .08);
    return [low >= 0 ? Math.max(0, low - margin) : low - margin, high + margin];
  }

  function draw(ctx, width, height, axis, points, selected = null) {
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height);
    const left = 51, right = width - 18, top = 22, bottom = height - 43;
    const x0 = points[0].coordinate_m, x1 = points[points.length - 1].coordinate_m;
    const [y0, y1] = limits(points);
    const px = x => left + (x - x0) / (x1 - x0) * (right - left);
    const py = y => bottom - (y - y0) / (y1 - y0) * (bottom - top);
    ctx.font = '11px "Malgun Gothic", Arial, sans-serif'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const value = y0 + (y1 - y0) * i / 4, y = py(value);
      ctx.strokeStyle = "#e6edf1"; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.fillStyle = "#607784"; ctx.textAlign = "right"; ctx.fillText(fmt(value, 2), left - 8, y + 4);
      const x = x0 + (x1 - x0) * i / 4;
      ctx.textAlign = "center"; ctx.fillText(fmt(zero(x), axis === "x" ? 1 : 2), px(x), bottom + 19);
    }
    ctx.strokeStyle = "#b7c9d3"; ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, bottom); ctx.lineTo(right, bottom); ctx.stroke();
    ctx.fillStyle = "#345362"; ctx.textAlign = "left"; ctx.fillText("Mach", left, 12);
    ctx.textAlign = "center"; ctx.fillText(axis + " (m)", (left + right) / 2, height - 6);
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    if (x0 < 0 && x1 > 0) {
      ctx.strokeStyle = "#cedbe2"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(px(0), top); ctx.lineTo(px(0), bottom); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = colors[axis]; ctx.lineWidth = 2; ctx.beginPath();
    points.forEach((p, i) => { if (i) ctx.lineTo(px(p.coordinate_m), py(p.mach)); else ctx.moveTo(px(p.coordinate_m), py(p.mach)); }); ctx.stroke();
    if (selected !== null) {
      const p = points[selected]; ctx.strokeStyle = "#9bafb9"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(px(p.coordinate_m), top); ctx.lineTo(px(p.coordinate_m), bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = colors[axis]; ctx.beginPath(); ctx.arc(px(p.coordinate_m), py(p.mach), 4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
    return {left, right, x0, x1};
  }

  class Plots {
    constructor() {
      this.data = null; this.selected = {}; this.geometry = {}; this.scheduled = false;
      this.canvases = {};
      for (const axis of axes) {
        const canvas = document.getElementById("volumeProfile" + axis.toUpperCase()); this.canvases[axis] = canvas;
        canvas.addEventListener("pointermove", event => {
          if (!this.data || !this.geometry[axis]) return;
          const {left, right, x0, x1} = this.geometry[axis], pixel = event.clientX - canvas.getBoundingClientRect().left;
          const value = x0 + Math.max(0, Math.min(1, (pixel - left) / (right - left))) * (x1 - x0);
          const points = this.data.lines[axis]; let nearest = 0;
          for (let i = 1; i < points.length; i++) if (Math.abs(points[i].coordinate_m - value) < Math.abs(points[nearest].coordinate_m - value)) nearest = i;
          this.selected[axis] = nearest; this.schedule();
        });
        canvas.addEventListener("pointerleave", () => { delete this.selected[axis]; this.schedule(); });
        canvas.addEventListener("keydown", event => {
          if (!this.data || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault(); const n = this.data.lines[axis].length;
          const next = event.key === "Home" ? 0 : event.key === "End" ? n - 1 : (this.selected[axis] ?? Math.floor(n / 2)) + (event.key === "ArrowLeft" ? -1 : 1);
          this.selected[axis] = Math.max(0, Math.min(n - 1, next)); this.schedule();
        });
        if (window.ResizeObserver) new ResizeObserver(() => this.schedule()).observe(canvas);
      }
      window.addEventListener("resize", () => this.schedule());
      document.getElementById("volumeProfilesCSV").addEventListener("click", () => this.download(new Blob(["\ufeff", toCSV(this.data)], {type: "text/csv;charset=utf-8"}), "csv"));
      document.getElementById("volumeProfilesPNG").addEventListener("click", () => {
        const canvas = this.exportPNG(); if (canvas) canvas.toBlob(blob => { if (blob) this.download(blob, "png"); }, "image/png");
      });
    }
    update(viewer, state, changed) {
      if (!viewer.ready || !viewer.field || !state) return;
      const lines = changed || !this.data ? sampleProfiles([viewer.nx, viewer.ny, viewer.nz], viewer.coordinates, viewer.field) : this.data.lines;
      this.data = {source: "3d_display_nodes", time_s: state.time_s, actual_L_mm: viewer.lastL, P0_Pa: state.P0_Pa, lines};
      document.getElementById("volumeProfilesStatus").textContent = "t = " + fmt(state.time_s, 2) + " s · L = " + fmt(viewer.lastL, 4) + " mm · P₀ = " + fmt(state.P0_Pa / 1000, 2) + " kPa";
      document.getElementById("volumeProfilesCSV").disabled = false; document.getElementById("volumeProfilesPNG").disabled = false;
      this.schedule();
    }
    schedule() {
      if (!this.data || this.scheduled) return;
      this.scheduled = true; requestAnimationFrame(() => { this.scheduled = false; this.render(); });
    }
    render() {
      if (!this.data) return;
      for (const axis of axes) {
        const canvas = this.canvases[axis], rect = canvas.getBoundingClientRect(); if (!rect.width || !rect.height) continue;
        const ratio = Math.min(window.devicePixelRatio || 1, 2), width = rect.width, height = rect.height;
        if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); }
        const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        const points = this.data.lines[axis], selected = this.selected[axis] ?? null;
        this.geometry[axis] = draw(ctx, width, height, axis, points, selected);
        const text = selected === null ? "Mach " + fmt(Math.min(...points.map(p => p.mach)), 4) + " ~ " + fmt(Math.max(...points.map(p => p.mach)), 4) : axis + " = " + fmt(points[selected].coordinate_m, 4) + " m · Mach " + fmt(points[selected].mach, 5);
        document.getElementById("volumeProfile" + axis.toUpperCase() + "Value").textContent = text;
        canvas.setAttribute("aria-label", "Mach–" + axis + ", " + captions[axis] + ". " + text + ". 좌우 방향키로 값 확인.");
      }
    }
    getData() { return this.data ? JSON.parse(JSON.stringify(this.data)) : null; }
    download(blob, extension) {
      if (!this.data) return;
      const a = document.createElement("a"), url = URL.createObjectURL(blob); a.href = url; a.download = "nozzle_mach_profiles_t" + fmt(this.data.time_s, 2) + "." + extension;
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    exportPNG() {
      if (!this.data) return null;
      const out = document.createElement("canvas"); out.width = 1800; out.height = 700; const ctx = out.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, out.width, out.height); ctx.fillStyle = "#16323e"; ctx.font = 'bold 28px "Malgun Gothic", Arial'; ctx.fillText("실시간 마하수 분포 · 현재 3D 예측장", 35, 48);
      ctx.font = '18px "Malgun Gothic", Arial'; ctx.fillText(document.getElementById("volumeProfilesStatus").textContent, 35, 83);
      axes.forEach((axis, i) => {
        const left = 30 + i * 590; ctx.fillStyle = "#16323e"; ctx.font = 'bold 22px Arial'; ctx.fillText("Mach–" + axis, left + 20, 131);
        ctx.font = '17px "Malgun Gothic", Arial'; ctx.fillText(captions[axis], left + 20, 161);
        ctx.save(); ctx.translate(left, 180); ctx.scale(1.5, 1.5); draw(ctx, 380, 285, axis, this.data.lines[axis]); ctx.restore();
      });
      ctx.fillStyle = "#607784"; ctx.font = '18px "Malgun Gothic", Arial'; ctx.fillText("Mach–x: 노즐 입구 → x = 0 전체 중심선 · Mach–y / Mach–z: 시험부 중심선의 대칭 복원", 35, 658); return out;
    }
  }
  return {sampleProfiles, toCSV, Plots};
});
