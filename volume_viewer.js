/* Nozzle-volume Mach ROM display, inlet through the physical x=0 plane. Scalars are sampled from the three-dimensional
 * trained model; symmetry reflects the quarter geometry, never the training set.
 * Binary contract: nozzle-volume-v1, float32 little endian, xyz/z-fast nodes. */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const fmt = (x, n = 2) => Number.isFinite(x) ? x.toFixed(n) : "—";
  const stops = [[45,53,125], [34,142,177], [94,197,131], [242,216,78], [194,52,46]];
  const gradient = "linear-gradient(90deg," + stops.map(c => "rgb(" + c.join(",") + ")").join(",") + ")";
  function bracket(values, x) {
    let a = 0, b = values.length - 1;
    if (x <= values[a]) return [a, a, 0];
    if (x >= values[b]) return [b, b, 0];
    while (b - a > 1) { const m = (a + b) >>> 1; if (values[m] <= x) a = m; else b = m; }
    return [a, b, (x - values[a]) / (values[b] - values[a])];
  }
  function compile(gl, kind, source) {
    const shader = gl.createShader(kind); gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }
  class VolumeViewer {
    constructor(options = {}) {
      this.manifestURL = options.manifestURL || "volume/manifest.json";
      this.canvas = $("volumeCanvas"); this.overlay = $("volumeOverlay"); this.axes = $("volumeAxes");
      this.ready = false; this.loading = false; this.active = false; this.state = null; this.lastL = null;
      this.view = "2d"; this.symmetry = "full"; this.surface = "cuts"; this.slice = 0;
      this.showY = true; this.showZ = false; this.yaw = -.30; this.pitch = .60; this.zoom = 1;
      this.panX = 0; this.panY = 0; this.renderScheduled = false; this.lastUpdateMs = 0;
      this.pointers = new Map(); this.requestId = 0;
      this.profiles = new NozzleProfiles.Plots();
      this.bind();
      $("volumeColorbar").style.background = gradient;
      this.bindContextEvents();
      this.load();
    }
    bindContextEvents() {
      this.canvas.addEventListener("webglcontextlost", event => {
        event.preventDefault(); this.contextLost = true; this.showStatus("3D 표시가 일시 중단되었습니다. 잠시 후 자동으로 복구합니다.", "loading");
      });
      this.canvas.addEventListener("webglcontextrestored", () => {
        try { this.initRenderer(); this.contextLost = false; this.upload(); this.schedule(); this.hideStatus(); }
        catch (error) { this.fail(error); }
      });
    }
    bind() {
      document.querySelectorAll("[data-volume-view]").forEach(button => button.addEventListener("click", () => this.setView(button.dataset.volumeView)));
      $("volumeSymmetry").addEventListener("change", event => { this.symmetry = event.target.value; this.schedule(); });
      $("volumeSurface").addEventListener("change", event => {
        this.surface = event.target.value;
        for (const id of ["volumeSlice", "volumeTestSection", "volumeYPlane", "volumeZPlane"]) $(id).disabled = this.surface !== "cuts";
        this.schedule();
      });
      $("volumeSlice").addEventListener("input", event => this.setSlice(Number(event.target.value)));
      $("volumeYPlane").addEventListener("change", event => { this.showY = event.target.checked; this.schedule(); });
      $("volumeZPlane").addEventListener("change", event => { this.showZ = event.target.checked; this.schedule(); });
      $("volumeResetView").addEventListener("click", () => this.resetCamera());
      $("volumeTestSection").addEventListener("click", () => {
        if (!this.ready) return;
        this.setSlice(this.nx - 1);
      });
      $("volumeRetry").addEventListener("click", () => this.load());
      $("volumeProfilesRetry").addEventListener("click", () => this.load());
      this.bindCanvasControls();
      window.addEventListener("resize", () => this.schedule());
      if (window.ResizeObserver) new ResizeObserver(() => this.schedule()).observe(this.canvas);
    }
    bindCanvasControls() {
      this.canvas.addEventListener("wheel", event => { event.preventDefault(); this.zoom = clamp(this.zoom * Math.exp(-event.deltaY * .001), .35, 8); this.schedule(); }, {passive: false});
      this.canvas.addEventListener("contextmenu", event => event.preventDefault());
      this.canvas.addEventListener("pointerdown", event => { this.canvas.setPointerCapture(event.pointerId); this.pointers.set(event.pointerId, [event.clientX, event.clientY]); });
      this.canvas.addEventListener("pointermove", event => {
        const old = this.pointers.get(event.pointerId); if (!old) return;
        const dx = event.clientX - old[0], dy = event.clientY - old[1];
        if (this.pointers.size === 2) {
          const other = [...this.pointers.entries()].find(([id]) => id !== event.pointerId)[1];
          const before = Math.hypot(old[0] - other[0], old[1] - other[1]);
          const after = Math.hypot(event.clientX - other[0], event.clientY - other[1]);
          if (before > 4) this.zoom = clamp(this.zoom * after / before, .35, 8);
        } else if (event.shiftKey || event.buttons === 2) { this.panX += dx; this.panY -= dy; }
        else { this.yaw += dx * .006; this.pitch = clamp(this.pitch + dy * .006, -1.45, 1.45); }
        this.pointers.set(event.pointerId, [event.clientX, event.clientY]); this.schedule();
      });
      for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) this.canvas.addEventListener(name, event => this.pointers.delete(event.pointerId));
      this.canvas.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "Home"].includes(event.key)) return;
        event.preventDefault();
        if (event.key === "Home") this.resetCamera();
        else if (["+", "="].includes(event.key)) this.zoom = clamp(this.zoom * 1.15, .35, 8);
        else if (event.key === "-") this.zoom = clamp(this.zoom / 1.15, .35, 8);
        else if (event.key === "ArrowLeft") this.yaw -= .12;
        else if (event.key === "ArrowRight") this.yaw += .12;
        else this.pitch = clamp(this.pitch + (event.key === "ArrowUp" ? -.12 : .12), -1.45, 1.45);
        this.schedule();
      });
    }
    setView(view) {
      if (!["2d", "3d", "both"].includes(view)) throw new Error("Unknown field view: " + view);
      this.view = view; this.active = view !== "2d";
      document.body.dataset.fieldView = view;
      $("volumePanel").hidden = !this.active;
      $("twoDFields").hidden = view === "3d";
      $("twoDFieldTools").hidden = view === "3d";
      $("pngButton").textContent = this.active ? "3D 그림 PNG" : "현재 화면 PNG";
      $("pngButton").disabled = this.active && !this.ready;
      document.querySelectorAll("[data-volume-view]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.volumeView === view)));
      if (this.active && !this.ready) this.load();
      if (this.active && this.ready) { this.update(this.state, true); this.schedule(); }
      window.dispatchEvent(new Event("resize"));
      return this.getState();
    }
    setSlice(index) {
      if (!this.ready) return;
      this.slice = Math.round(clamp(index, 0, this.nx - 1)); $("volumeSlice").value = this.slice;
      this.xIndices = this.makePlane(0, this.slice); this.uploadIndices("x", this.xIndices); this.updateLabels(); this.schedule();
    }
    resetCamera() { this.yaw = -.30; this.pitch = .60; this.zoom = 1; this.panX = this.panY = 0; this.schedule(); }
    showStatus(message, kind = "loading") {
      this.overlay.hidden = false; this.overlay.dataset.status = kind;
      $("volumeLoadMessage").textContent = message; $("volumeRetry").hidden = kind !== "error";
      $("volumeBusy").hidden = kind !== "loading"; $("volumeLoadStatus").textContent = kind === "error" ? "3D 데이터 연결 확인 필요" : "3D 모델 불러오는 중";
      if (!this.ready) {
        $("volumeProfilesStatus").textContent = kind === "error" ? "마하수 분포를 불러오지 못했습니다. 연결을 확인한 뒤 다시 불러오기를 눌러 주세요." : "현재 운전점의 마하수 분포를 불러오고 있습니다.";
        $("volumeProfilesRetry").hidden = kind !== "error";
      }
    }
    hideStatus() { this.overlay.hidden = true; }
    fail(error) {
      this.loading = false; this.ready = false; if (this.loader) this.loader.abort();
      if (this.active) $("pngButton").disabled = true;
      const reason = location.protocol === "file:" ? "3D 데이터는 웹사이트 주소 또는 로컬 웹 서버에서 열어 주세요." : "3D 데이터를 불러오지 못했습니다. 연결을 확인하고 다시 시도해 주세요.";
      this.showStatus(reason, "error"); this.error = error.message;
      console.error("Volume ROM display:", error);
    }
    async load() {
      if (this.loading || this.ready) return;
      this.loading = true; this.error = null; this.loader = new AbortController(); this.showStatus("입구부터 시험부 x = 0까지의 노즐 3D 모델을 불러오고 있습니다.");
      try {
        const manifestURL = new URL(this.manifestURL, document.baseURI);
        const response = await fetch(manifestURL, {signal: this.loader.signal, cache: "no-cache"}); if (!response.ok) throw new Error("manifest HTTP " + response.status);
        const m = await response.json();
        if (m.schema !== "nozzle-volume-v1") throw new Error("Unsupported volume manifest schema");
        if (m.domain?.id !== "nozzle_x_le_0" || m.domain.x_max_m !== 0 || m.domain.endpoint !== "physical_x_plane") throw new Error("The viewer requires the nozzle-only model ending at the physical x=0 plane");
        if (!Array.isArray(m.grid.shape) || m.grid.shape.length !== 3 || m.grid.order !== "xyz-zfast") throw new Error("Unsupported volume grid");
        [this.nx, this.ny, this.nz] = m.grid.shape; this.n = this.nx * this.ny * this.nz; this.rank = m.rank;
        if (!m.grid.shape.every(x => Number.isInteger(x) && x >= 2) || !Number.isInteger(this.rank) || this.rank < 1) throw new Error("Invalid volume dimensions");
        for (const values of [m.geometry_L_mm, m.path.L_mm]) if (!values?.length || values.some((x, i) => !Number.isFinite(x) || (i && x < values[i - 1]))) throw new Error("Unordered volume L samples");
        if (m.path.coefficients.length !== m.path.L_mm.length || m.path.coefficients.some(row => row.length !== this.rank || row.some(x => !Number.isFinite(x)))) throw new Error("Invalid volume path coefficients");
        const lengths = {mean: this.n, modes: this.rank * this.n, coordinates: m.geometry_L_mm.length * this.n * 3};
        const total = Object.values(lengths).reduce((sum, count) => sum + count * 4, 0); this.loadedBytes = 0; this.totalBytes = total;
        const read = async key => {
          const info = m.arrays[key]; if (!info || info.count !== lengths[key]) throw new Error("Invalid " + key + " count");
          const assetURL = new URL(info.file, manifestURL); if (info.sha256) assetURL.searchParams.set("v", info.sha256);
          const res = await fetch(assetURL, {signal: this.loader.signal}); if (!res.ok) throw new Error(key + " HTTP " + res.status);
          let buffer;
          if (res.body?.getReader) {
            const reader = res.body.getReader(), chunks = []; let bytes = 0;
            for (;;) { const part = await reader.read(); if (part.done) break; chunks.push(part.value); bytes += part.value.length; this.loadedBytes += part.value.length; this.progress(); }
            const all = new Uint8Array(bytes); let offset = 0; for (const chunk of chunks) { all.set(chunk, offset); offset += chunk.length; } buffer = all.buffer;
          } else { buffer = await res.arrayBuffer(); this.loadedBytes += buffer.byteLength; this.progress(); }
          if (buffer.byteLength !== lengths[key] * 4) throw new Error("Incomplete " + key + " data");
          const values = new Float32Array(buffer); for (const x of values) if (!Number.isFinite(x)) throw new Error("Non-finite " + key + " data");
          return values;
        };
        const [mean, modes, geometry] = await Promise.all([read("mean"), read("modes"), read("coordinates")]);
        const endpointTolerance = 1e-6, planeSize = this.ny * this.nz;
        for (let g = 0; g < m.geometry_L_mm.length; g++) {
          const start = g * this.n * 3, lastPlane = start + (this.nx - 1) * planeSize * 3;
          for (let i = 0; i < this.n; i++) if (geometry[start + i * 3] > endpointTolerance) throw new Error("Nozzle display geometry contains downstream points");
          for (let i = 0; i < planeSize; i++) if (Math.abs(geometry[lastPlane + i * 3]) > endpointTolerance) throw new Error("The nozzle outlet must be the exact physical x=0 plane");
        }
        this.manifest = m; this.mean = mean; this.modes = modes; this.geometry = geometry;
        this.field = new Float32Array(this.n); this.coordinates = new Float32Array(this.n * 3); this.vertices = new Float32Array(this.n * 4);
        this.colorMin = m.color_range[0]; this.colorMax = m.color_range[1];
        this.surfaceIndices = this.makeSurface(); this.yIndices = this.makePlane(1, 0); this.zIndices = this.makePlane(2, 0); this.edgeIndices = this.makeEdges();
        this.initRenderer(); this.loading = false; this.ready = true; $("pngButton").disabled = false;
        $("volumeSlice").max = this.nx - 1;
        this.update(this.state, true);
        this.setSlice(this.nx - 1); this.hideStatus(); this.updateLabels(); this.schedule();
      } catch (error) { this.fail(error); }
    }
    progress() {
      if (!this.loading) return;
      const now = performance.now();
      if (this.loadedBytes < this.totalBytes && now - (this.lastProgressTime || 0) < 150) return;
      this.lastProgressTime = now;
      const progress = fmt(this.loadedBytes / 1048576, 1) + " / " + fmt(this.totalBytes / 1048576, 1) + " MB";
      $("volumeLoadMessage").textContent = "3D 데이터 " + progress;
      $("volumeProfilesStatus").textContent = "마하수 분포 불러오는 중 · " + progress;
    }
    node(i, j, k) { return (i * this.ny + j) * this.nz + k; }
    makePlane(axis, fixed) {
      const shape = [this.nx, this.ny, this.nz], uv = [0,1,2].filter(a => a !== axis), ids = [];
      const at = (a, b) => { const p = [0,0,0]; p[axis] = fixed; p[uv[0]] = a; p[uv[1]] = b; return this.node(...p); };
      for (let a = 0; a < shape[uv[0]] - 1; a++) for (let b = 0; b < shape[uv[1]] - 1; b++) {
        const i = at(a, b), j = at(a + 1, b), k = at(a + 1, b + 1), l = at(a, b + 1); ids.push(i,j,k,i,k,l);
      }
      return new Uint32Array(ids);
    }
    makeSurface() {
      const parts = [this.makePlane(0, 0), this.makePlane(0, this.nx - 1), this.makePlane(1, this.ny - 1), this.makePlane(2, this.nz - 1)];
      const all = new Uint32Array(parts.reduce((s, x) => s + x.length, 0)); let off = 0;
      for (const part of parts) { all.set(part, off); off += part.length; } return all;
    }
    makeEdges() {
      const ids = [], sizes = [this.nx, this.ny, this.nz];
      for (let axis = 0; axis < 3; axis++) {
        const other = [0,1,2].filter(x => x !== axis);
        for (const a of [0, sizes[other[0]] - 1]) for (const b of [0, sizes[other[1]] - 1]) {
          for (let t = 0; t < sizes[axis] - 1; t++) { const p = [0,0,0]; p[other[0]] = a; p[other[1]] = b; p[axis] = t; ids.push(this.node(...p)); p[axis]++; ids.push(this.node(...p)); }
        }
      }
      return new Uint32Array(ids);
    }
    initRenderer() {
      this.gl = this.canvas.getContext("webgl", {antialias: true, alpha: false, preserveDrawingBuffer: true});
      if (!this.gl || !this.gl.getExtension("OES_element_index_uint")) {
        // A replacement is necessary: a canvas cannot change its context type.
        if (this.gl) { const replacement = this.canvas.cloneNode(true); this.canvas.replaceWith(replacement); this.canvas = replacement; this.canvas.setAttribute("tabindex", "0"); this.bindCanvasControls(); this.bindContextEvents(); if (window.ResizeObserver) new ResizeObserver(() => this.schedule()).observe(this.canvas); }
        this.gl = null; this.ctx = this.canvas.getContext("2d"); this.backend = "canvas2d"; return;
      }
      const gl = this.gl;
      const vs = compile(gl, gl.VERTEX_SHADER, `attribute vec4 aData; uniform vec3 uCenter; uniform vec2 uMirror; uniform mat3 uRotation; uniform vec2 uScale; uniform vec2 uPan; varying float vMach;
      void main(){vec3 p=vec3(aData.x-uCenter.x,aData.y*uMirror.x-uCenter.y,aData.z*uMirror.y-uCenter.z); vec3 q=uRotation*p; gl_Position=vec4(q.x*uScale.x+uPan.x,q.y*uScale.y+uPan.y,q.z/40.0,1.0);vMach=aData.w;}`);
      const fs = compile(gl, gl.FRAGMENT_SHADER, `precision mediump float; varying float vMach; uniform vec2 uRange; uniform bool uLine; void main(){if(uLine){gl_FragColor=vec4(.19,.28,.33,1.0);return;}float t=clamp((vMach-uRange.x)/(uRange.y-uRange.x),0.0,1.0)*4.0;vec3 a;vec3 b; if(t<1.0){a=vec3(45.,53.,125.);b=vec3(34.,142.,177.);}else if(t<2.0){a=vec3(34.,142.,177.);b=vec3(94.,197.,131.);}else if(t<3.0){a=vec3(94.,197.,131.);b=vec3(242.,216.,78.);}else{a=vec3(242.,216.,78.);b=vec3(194.,52.,46.);}gl_FragColor=vec4(mix(a,b,min(fract(t),1.0))/255.0,1.0);if(t>=4.0)gl_FragColor=vec4(vec3(194.,52.,46.)/255.0,1.0);}`);
      this.program = gl.createProgram(); gl.attachShader(this.program, vs); gl.attachShader(this.program, fs); gl.linkProgram(this.program);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
      gl.useProgram(this.program); this.uniforms = {};
      for (const key of ["Center", "Mirror", "Rotation", "Scale", "Pan", "Range", "Line"]) this.uniforms[key] = gl.getUniformLocation(this.program, "u" + key);
      this.vertexBuffer = gl.createBuffer(); this.indexBuffers = {};
      this.attribute = gl.getAttribLocation(this.program, "aData"); gl.enableVertexAttribArray(this.attribute);
      this.backend = "webgl";
      this.uploadIndices("surface", this.surfaceIndices); this.uploadIndices("y", this.yIndices); this.uploadIndices("z", this.zIndices); this.uploadIndices("edges", this.edgeIndices);
      if (this.xIndices) this.uploadIndices("x", this.xIndices);
    }
    uploadIndices(name, indices) {
      if (!this.gl) return; const gl = this.gl;
      const buffer = this.indexBuffers[name] || gl.createBuffer(); this.indexBuffers[name] = buffer;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    }
    update(state, force = false) {
      if (state) this.state = state;
      if (!this.ready || !this.state) return;
      const L = this.state.actual_L_mm;
      if (!force && Math.abs(L - this.lastL) < 1e-10) { this.profiles.update(this, this.state, false); return; }
      const begin = performance.now(), m = this.manifest;
      const [pa, pb, t] = bracket(m.path.L_mm, L), [ga, gb, g] = bracket(m.geometry_L_mm, L);
      this.coefficients = m.path.coefficients[pa].map((v, i) => v + t * (m.path.coefficients[pb][i] - v));
      this.field.set(this.mean);
      for (let r = 0; r < this.rank; r++) {
        const coefficient = this.coefficients[r], offset = r * this.n;
        for (let i = 0; i < this.n; i++) this.field[i] += coefficient * this.modes[offset + i];
      }
      const a = ga * this.n * 3, b = gb * this.n * 3;
      let min = Infinity, max = -Infinity, sum = 0;
      this.bounds = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
      for (let i = 0; i < this.n; i++) {
        for (let axis = 0; axis < 3; axis++) {
          const k = i * 3 + axis, value = this.geometry[a + k] + g * (this.geometry[b + k] - this.geometry[a + k]);
          this.coordinates[k] = value; this.vertices[i * 4 + axis] = value;
          this.bounds[axis][0] = Math.min(this.bounds[axis][0], value); this.bounds[axis][1] = Math.max(this.bounds[axis][1], value);
        }
        const v = this.field[i]; this.vertices[i * 4 + 3] = v; min = Math.min(min, v); max = Math.max(max, v); sum += v;
      }
      this.fieldStats = {min, max, sample_mean: sum / this.n}; this.lastL = L; this.upload();
      this.profiles.update(this, this.state, true);
      this.lastUpdateMs = performance.now() - begin; this.updateLabels(); this.schedule();
    }
    upload() { if (this.gl) { this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer); this.gl.bufferData(this.gl.ARRAY_BUFFER, this.vertices, this.gl.DYNAMIC_DRAW); } }
    updateLabels() {
      if (!this.ready || !this.bounds) return;
      const m = this.manifest;
      $("volumeMin").textContent = fmt(this.colorMin); $("volumeMax").textContent = fmt(this.colorMax);
      let sliceMin = Infinity, sliceMax = -Infinity;
      const sliceStart = this.slice * this.ny * this.nz;
      for (let i = sliceStart; i < sliceStart + this.ny * this.nz; i++) { sliceMin = Math.min(sliceMin, this.coordinates[i * 3]); sliceMax = Math.max(sliceMax, this.coordinates[i * 3]); }
      this.sliceRange = [sliceMin, sliceMax];
      const testSection = this.slice === this.nx - 1 && Math.max(Math.abs(sliceMin), Math.abs(sliceMax)) <= 1e-6;
      $("volumeSliceValue").textContent = testSection ? "x = 0.000 m" : "x ≈ " + fmt((sliceMin + sliceMax) / 2, 3) + " m";
      $("volumeSliceValue").title = testSection ? "시험부 단면: 정확한 x = 0 평면" : "격자 횡단면의 실제 x 범위: " + fmt(sliceMin, 5) + " ~ " + fmt(sliceMax, 5) + " m";
      $("volumeSliceValue").setAttribute("aria-label", $("volumeSliceValue").title);
      $("volumeActualL").textContent = "L " + fmt(this.lastL, 4) + " mm · P₀ " + fmt(this.state.P0_Pa / 1000, 2) + " kPa";
      $("volumeLoadStatus").textContent = (m.training?.case_count ? m.training.case_count + "개 CFD · " : "") + "노즐 체적 Mach POD–GPR";
      $("volumeDomain").textContent = "입구 → 시험부 x = 0 · " + fmt(this.bounds[0][0], 2) + " ~ 0 m";
      $("volumeSummary").textContent = "입구부터 x = 0까지의 ¼ 노즐 체적을 학습하고 y = 0, z = 0에서 대칭 복원합니다. 화면은 " + this.n.toLocaleString("ko-KR") + "개 표시점으로 재구성합니다.";
      if (m.training?.cell_count) $("volumeTrainingSummary").textContent = "학습: " + m.training.cell_count.toLocaleString("ko-KR") + "개 노즐 체적 셀 · 표시: " + this.nx + " × " + this.ny + " × " + this.nz + " 격자";
      this.canvas.setAttribute("aria-label", "입구부터 시험부 x = 0까지 노즐 3D 마하수 예측. L " + fmt(this.lastL, 4) + " mm. 회전은 방향키, 확대는 더하기, 축소는 빼기, 초기 시점은 Home 키.");
    }
    schedule() {
      if (!this.active || !this.ready || this.renderScheduled || this.contextLost) return;
      this.renderScheduled = true; requestAnimationFrame(() => { this.renderScheduled = false; this.render(); });
    }
    camera() {
      const rect = this.canvas.getBoundingClientRect(), width = Math.max(200, rect.width), height = Math.max(200, rect.height);
      const pixelRatio = Math.min(devicePixelRatio || 1, 2);
      if (this.canvas.width !== Math.round(width * pixelRatio) || this.canvas.height !== Math.round(height * pixelRatio)) { this.canvas.width = Math.round(width * pixelRatio); this.canvas.height = Math.round(height * pixelRatio); }
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const rotation = new Float32Array([cy, -sp * sy, cp * sy, -sy, -sp * cy, cp * cy, 0, cp, sp]);
      const quarter = this.symmetry === "quarter";
      const center = [(this.bounds[0][0] + this.bounds[0][1]) / 2,
        quarter ? (this.bounds[1][0] + this.bounds[1][1]) / 2 : 0,
        quarter ? (this.bounds[2][0] + this.bounds[2][1]) / 2 : 0];
      const halves = [(this.bounds[0][1] - this.bounds[0][0]) / 2,
        quarter ? (this.bounds[1][1] - this.bounds[1][0]) / 2 : Math.max(Math.abs(this.bounds[1][0]), Math.abs(this.bounds[1][1])),
        quarter ? (this.bounds[2][1] - this.bounds[2][0]) / 2 : Math.max(Math.abs(this.bounds[2][0]), Math.abs(this.bounds[2][1]))];
      const projectedWidth = Math.abs(rotation[0]) * halves[0] + Math.abs(rotation[3]) * halves[1] + Math.abs(rotation[6]) * halves[2];
      const projectedHeight = Math.abs(rotation[1]) * halves[0] + Math.abs(rotation[4]) * halves[1] + Math.abs(rotation[7]) * halves[2];
      const scale = 1.14 * Math.max(projectedWidth / (width / height), projectedHeight, .001) / this.zoom;
      return {width, height, pixelRatio, rotation, center, scale, mirrors: this.symmetry === "quarter" ? [[1,1]] : [[1,1],[-1,1],[1,-1],[-1,-1]]};
    }
    visibleParts() {
      const parts = [];
      if (this.surface === "surface") { parts.push(["surface", this.surfaceIndices]); if (this.symmetry === "quarter") parts.push(["y", this.yIndices], ["z", this.zIndices]); }
      else { if (this.xIndices) parts.push(["x", this.xIndices]); if (this.showY) parts.push(["y", this.yIndices]); if (this.showZ) parts.push(["z", this.zIndices]); }
      return parts;
    }
    render() {
      if (!this.active || !this.ready || !this.bounds || this.contextLost) return;
      const c = this.camera(); if (this.gl) this.renderGL(c); else this.renderCanvas(c); this.drawAxes(c);
    }
    renderGL(c) {
      const gl = this.gl, u = this.uniforms;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height); gl.clearColor(.965,.980,.986,1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE); gl.useProgram(this.program);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer); gl.vertexAttribPointer(this.attribute, 4, gl.FLOAT, false, 0, 0);
      gl.uniform3fv(u.Center, c.center); gl.uniformMatrix3fv(u.Rotation, false, c.rotation);
      gl.uniform2f(u.Scale, 1 / (c.scale * c.width / c.height), 1 / c.scale); gl.uniform2f(u.Pan, 2 * this.panX / c.width, 2 * this.panY / c.height);
      gl.uniform2f(u.Range, this.colorMin, this.colorMax); gl.uniform1i(u.Line, 0);
      gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(1,1);
      for (const mirror of c.mirrors) {
        gl.uniform2fv(u.Mirror, mirror);
        for (const [name, indices] of this.visibleParts()) {
          // Center planes already span both signs of their normal by reflection.
          if ((name === "y" && mirror[0] === -1) || (name === "z" && mirror[1] === -1)) continue;
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffers[name]); gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);
        }
      }
      gl.disable(gl.POLYGON_OFFSET_FILL); gl.uniform1i(u.Line, 1); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffers.edges);
      for (const mirror of c.mirrors) { gl.uniform2fv(u.Mirror, mirror); gl.drawElements(gl.LINES, this.edgeIndices.length, gl.UNSIGNED_INT, 0); }
    }
    transform(i, mirror, c) {
      const r = c.rotation, x = this.coordinates[i*3] - c.center[0], y = this.coordinates[i*3+1] * mirror[0] - c.center[1], z = this.coordinates[i*3+2] * mirror[1] - c.center[2];
      const scale = c.height / (2 * c.scale);
      return [c.width/2 + (r[0]*x+r[3]*y+r[6]*z)*scale + this.panX, c.height/2 - (r[1]*x+r[4]*y+r[7]*z)*scale - this.panY, r[2]*x+r[5]*y+r[8]*z];
    }
    color(value) {
      const t = clamp((value - this.colorMin) / (this.colorMax - this.colorMin), 0, 1) * 4, i = Math.min(3, Math.floor(t)), f = t - i;
      return "rgb(" + stops[i].map((x,j) => Math.round(x + f * (stops[i+1][j]-x))).join(",") + ")";
    }
    renderCanvas(c) {
      const ctx = this.ctx; if (!ctx) return;
      ctx.setTransform(c.pixelRatio,0,0,c.pixelRatio,0,0); ctx.fillStyle = "#f7fafb"; ctx.fillRect(0,0,c.width,c.height);
      const triangles = [];
      for (const mirror of c.mirrors) for (const [name, ids] of this.visibleParts()) {
        if ((name === "y" && mirror[0] === -1) || (name === "z" && mirror[1] === -1)) continue;
        for (let i=0; i<ids.length; i+=3) {
          const a=this.transform(ids[i],mirror,c), b=this.transform(ids[i+1],mirror,c), d=this.transform(ids[i+2],mirror,c);
          triangles.push({p:[a,b,d], z:(a[2]+b[2]+d[2])/3, m:(this.field[ids[i]]+this.field[ids[i+1]]+this.field[ids[i+2]])/3});
        }
      }
      triangles.sort((a,b)=>b.z-a.z); ctx.lineWidth=.35;
      for (const t of triangles) { ctx.beginPath(); ctx.moveTo(t.p[0][0],t.p[0][1]); ctx.lineTo(t.p[1][0],t.p[1][1]); ctx.lineTo(t.p[2][0],t.p[2][1]); ctx.closePath(); ctx.fillStyle=ctx.strokeStyle=this.color(t.m); ctx.fill(); ctx.stroke(); }
      ctx.strokeStyle="#526975"; ctx.lineWidth=.7;
      for (const mirror of c.mirrors) { ctx.beginPath(); for(let i=0;i<this.edgeIndices.length;i+=2){const a=this.transform(this.edgeIndices[i],mirror,c),b=this.transform(this.edgeIndices[i+1],mirror,c);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);} ctx.stroke(); }
    }
    drawAxes(c) {
      const canvas = this.axes, ctx = canvas.getContext("2d"), r = c.rotation;
      canvas.width = 110; canvas.height = 100; ctx.clearRect(0,0,110,100); ctx.font = 'bold 12px Arial'; ctx.lineWidth = 2;
      [[0,"x", "#b65b45"],[1,"y","#427b6c"],[2,"z","#3878a2"]].forEach(([axis,label,color]) => {
        const dx = r[axis*3]*31, dy = -r[axis*3+1]*31; ctx.strokeStyle=ctx.fillStyle=color; ctx.beginPath(); ctx.moveTo(49,57);ctx.lineTo(49+dx,57+dy);ctx.stroke();ctx.fillText(label,49+dx+4,57+dy+3);
      });
      ctx.fillStyle="#69818d";ctx.font='10px Arial';ctx.fillText("실제 길이 비율",17,94);
    }
    exportPNG() {
      this.render(); const out = document.createElement("canvas"); out.width=1800;out.height=1080; const ctx=out.getContext("2d");
      ctx.fillStyle="#eef3f5";ctx.fillRect(0,0,out.width,out.height);ctx.fillStyle="#16323e";ctx.font='bold 29px "Malgun Gothic",Arial';ctx.fillText("노즐 3D 마하수 · 입구 → 시험부 x = 0",38,52);
      ctx.font='19px "Malgun Gothic",Arial';ctx.fillText($("volumeActualL").textContent+" · "+$("volumeSliceValue").textContent+" · "+(this.symmetry === "full" ? "대칭 복원" : "¼ 해석 영역"),38,90);
      const ratio=Math.min(1724/this.canvas.width,800/this.canvas.height);ctx.drawImage(this.canvas,(1800-this.canvas.width*ratio)/2,118,this.canvas.width*ratio,this.canvas.height*ratio);
      const bar=ctx.createLinearGradient(40,0,360,0);stops.forEach((color,i)=>bar.addColorStop(i/4,"rgb("+color.join(",")+")"));ctx.fillStyle=bar;ctx.fillRect(40,952,320,16);ctx.fillStyle="#526975";ctx.font='17px "Malgun Gothic",Arial';ctx.fillText("Mach "+fmt(this.colorMin)+" ~ "+fmt(this.colorMax)+" · 고정 색범위",383,967);
      ctx.fillText($("volumeTrainingSummary").textContent,40,1005);ctx.fillText("현재 운전점의 정상 CFD 기반 예측 · 대칭 가정에 따른 복원 · 구동기·센서 동특성은 가정값",40,1041);return out;
    }
    getState() {
      return {view:this.view,ready:this.ready,loading:this.loading,error:this.error||null,backend:this.backend||null,actual_L_mm:this.lastL,rank:this.rank||null,
        domain:this.manifest?.domain||null,node_count:this.n||0,grid_shape:this.ready?[this.nx,this.ny,this.nz]:null,training:this.manifest?.training||null,field_stats:this.fieldStats||null,geometry_bounds:this.bounds||null,
        slice_index:this.slice,slice_x_range_m:this.sliceRange||null,slice_x_m:this.ready&&this.coordinates?this.coordinates[this.slice*this.ny*this.nz*3]:null,symmetry:this.symmetry,surface:this.surface,update_ms:this.lastUpdateMs,loaded_bytes:this.loadedBytes||0,coefficients:this.coefficients||null};
    }
  }
  window.NozzleVolume = {Viewer:VolumeViewer};
})();
