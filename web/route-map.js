const METRES_PER_DEGREE = 111320;

function niceDistance(distance) {
  const magnitude = 10 ** Math.floor(Math.log10(distance));
  return [5, 2, 1].map(step => step * magnitude).find(step => step <= distance) || magnitude;
}

export class RouteMap {
  constructor(canvas, onPick) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.backdrop = document.createElement('canvas');
    this.layers = {roads: true, route: true, gps: true, classifier: false};
    this.onPick = onPick;
    this.cx = this.cy = 0;
    this.scale = 1;
    this.selected = 0;
    this.drag = null;
    const resize = new ResizeObserver(() => this.resize());
    resize.observe(canvas);
    canvas.addEventListener('wheel', event => {
      if (!this.data) return;
      event.preventDefault();
      this.zoom(Math.exp(-event.deltaY * 0.002), event.offsetX, event.offsetY);
    }, {passive: false});
    canvas.addEventListener('pointerdown', event => this.pointerDown(event));
    canvas.addEventListener('pointermove', event => this.pointerMove(event));
    canvas.addEventListener('pointerup', event => this.pointerEnd(event, true));
    canvas.addEventListener('pointercancel', event => this.pointerEnd(event, false));
    canvas.addEventListener('keydown', event => this.keyboard(event));
  }

  setData(data) {
    this.data = data;
    this.origin = data.bounds[0].map((value, i) => (value + data.bounds[1][i]) / 2);
    this.metresLon = METRES_PER_DEGREE * Math.max(0.05, Math.cos(this.origin[1] * Math.PI / 180));
    const project = point => point ? [(point[0] - this.origin[0]) * this.metresLon,
      (this.origin[1] - point[1]) * METRES_PER_DEGREE] : null;
    this.vertices = data.network.vertices.map(project);
    this.gps = data.observations.map(point => project(point.raw));
    this.snapped = data.observations.map(point => project(point.snapped));
    this.route = data.route.map(line => line.map(project));
    this.classifier = data.classifier.map(line => line.map(project));
    this.bounds = data.bounds.map(project);
    this.fit();
  }

  resize() {
    const bounds = this.canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    this.width = bounds.width;
    this.height = bounds.height;
    this.ratio = Math.min(window.devicePixelRatio || 1, 2);
    for (const surface of [this.canvas, this.backdrop]) {
      surface.width = Math.round(this.width * this.ratio);
      surface.height = Math.round(this.height * this.ratio);
    }
    if (this.data) this.fit();
    else this.refresh();
  }

  fit() {
    if (!this.bounds || !this.width) return;
    this.cx = 0;
    this.cy = 0;
    this.fitScale = Math.min((this.width - 74) / Math.max(80, Math.abs(this.bounds[0][0] - this.bounds[1][0])),
      (this.height - 116) / Math.max(80, Math.abs(this.bounds[0][1] - this.bounds[1][1])));
    this.scale = this.fitScale;
    this.refresh();
  }

  toScreen(point) {
    return [(point[0] - this.cx) * this.scale + this.width / 2, (point[1] - this.cy) * this.scale + this.height / 2];
  }

  zoom(factor, x = this.width / 2, y = this.height / 2) {
    if (!this.data) return;
    const before = this.scale;
    this.scale = Math.max(this.fitScale * 0.24, Math.min(this.fitScale * 55, before * factor));
    this.cx += (x - this.width / 2) * (1 / before - 1 / this.scale);
    this.cy += (y - this.height / 2) * (1 / before - 1 / this.scale);
    this.refresh();
  }

  pointerDown(event) {
    if (!this.data || event.button !== 0 || this.drag) return;
    this.canvas.setPointerCapture(event.pointerId);
    this.drag = {id: event.pointerId, x: event.clientX, y: event.clientY, cx: this.cx, cy: this.cy, moved: false};
    this.canvas.classList.add('dragging');
  }

  pointerMove(event) {
    if (this.drag?.id !== event.pointerId) return;
    const dx = event.clientX - this.drag.x, dy = event.clientY - this.drag.y;
    this.drag.moved ||= Math.hypot(dx, dy) > 3;
    if (!this.drag.moved) return;
    this.cx = this.drag.cx - dx / this.scale;
    this.cy = this.drag.cy - dy / this.scale;
    this.refresh();
  }

  pointerEnd(event, canPick) {
    if (this.drag?.id !== event.pointerId) return;
    const clicked = canPick && !this.drag.moved;
    this.drag = null;
    this.canvas.classList.remove('dragging');
    if (!clicked) return;
    let nearest = -1, best = 16 ** 2;
    this.gps.forEach((position, index) => {
      const shown = this.layers.route ? this.snapped[index] || position : position;
      if (!shown) return;
      const [x, y] = this.toScreen(shown);
      const distance = (x - event.offsetX) ** 2 + (y - event.offsetY) ** 2;
      if (distance < best) { best = distance; nearest = index; }
    });
    if (nearest >= 0) this.onPick(nearest);
  }

  keyboard(event) {
    if (!this.data) return;
    if (event.key === '+' || event.key === '=') this.zoom(1.25);
    else if (event.key === '-') this.zoom(0.8);
    else if (event.key === '0') this.fit();
    else {
      const movement = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.key];
      if (!movement) return;
      this.cx += movement[0] * 32 / this.scale;
      this.cy += movement[1] * 32 / this.scale;
      this.refresh();
    }
    event.preventDefault();
  }

  refresh() {
    if (!this.width) return;
    if (this.refreshPending) return;
    this.refreshPending = true;
    requestAnimationFrame(() => {
      this.refreshPending = false;
      this.drawBackdrop();
      this.drawSelection();
    });
  }

  drawBackdrop() {
    const ctx = this.backdrop.getContext('2d');
    const {width, height, ratio} = this;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const fill = ctx.createRadialGradient(width * .51, height * .48, 12, width * .52, height * .47, width * .58);
    fill.addColorStop(0, '#233d32');
    fill.addColorStop(1, '#142820');
    ctx.fillStyle = fill;
    ctx.fillRect(0, 0, width, height);
    this.grid(ctx);
    if (!this.data) return;
    if (this.layers.roads) this.drawRoads(ctx);
    if (this.layers.classifier) {
      ctx.setLineDash([4, 5]);
      this.drawLines(ctx, this.classifier, '#a9aff3cc', 2.2);
      ctx.setLineDash([]);
    }
    if (this.layers.gps) this.drawGPS(ctx);
    if (this.layers.route) {
      this.drawLines(ctx, this.route, '#9dde632f', 8);
      this.drawLines(ctx, this.route, '#d0f38b', 2.6);
    }
    const series = this.layers.route ? this.snapped : this.layers.gps ? this.gps : [];
    const start = series.find(Boolean), end = series.findLast(Boolean);
    if (start) this.mark(ctx, start, 'A', '#d0f38b');
    if (end && end !== start) this.mark(ctx, end, 'B', '#edf5ec');
    const metres = niceDistance(83 / this.scale);
    document.querySelector('#scale-line').style.width = `${Math.round(metres * this.scale)}px`;
    document.querySelector('#scale-label').textContent = metres < 1000 ? `${metres} m` : `${metres / 1000} km`;
    const lat = this.origin[1] - this.cy / METRES_PER_DEGREE;
    const lon = this.origin[0] + this.cx / this.metresLon;
    const signed = (value, plus, minus) => `${Math.abs(value).toFixed(4)}° ${value >= 0 ? plus : minus}`;
    document.querySelector('#coords').textContent = `${signed(lat, 'N', 'S')}  ${signed(lon, 'E', 'W')}  ·  WGS 84`;
  }

  grid(ctx) {
    const step = this.data ? niceDistance(150 / this.scale) * this.scale : 52;
    const [x, y] = this.toScreen([0, 0]);
    ctx.beginPath();
    for (let line = x % step; line < this.width; line += step) { ctx.moveTo(line, 0); ctx.lineTo(line, this.height); }
    for (let line = y % step; line < this.height; line += step) { ctx.moveTo(0, line); ctx.lineTo(this.width, line); }
    ctx.strokeStyle = '#88ad941a';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  drawRoads(ctx) {
    const vertices = this.vertices.map(point => this.toScreen(point));
    ctx.beginPath();
    for (const [a, b] of this.data.network.edges) {
      const start = vertices[a], end = vertices[b];
      if ((start[0] < 0 && end[0] < 0) || (start[0] > this.width && end[0] > this.width) ||
          (start[1] < 0 && end[1] < 0) || (start[1] > this.height && end[1] > this.height)) continue;
      ctx.moveTo(...start);
      ctx.lineTo(...end);
    }
    ctx.strokeStyle = '#76998a5e';
    ctx.lineWidth = this.scale > this.fitScale * 3.2 ? 1.25 : .85;
    ctx.stroke();
  }

  drawGPS(ctx) {
    this.drawLines(ctx, [this.gps], '#ffaf7b55', 1);
    ctx.beginPath();
    const density = this.scale >= this.fitScale * 2 ? 1 : Math.max(1, Math.floor(this.gps.length / 430));
    for (let i = 0; i < this.gps.length; i += density) {
      if (!this.gps[i]) continue;
      const [x, y] = this.toScreen(this.gps[i]);
      ctx.moveTo(x + 1.5, y);
      ctx.arc(x, y, 1.5, 0, Math.PI * 2);
    }
    ctx.fillStyle = '#ffb179c7';
    ctx.fill();
  }

  drawLines(ctx, lines, color, weight) {
    ctx.beginPath();
    for (const line of lines) {
      let previous = false;
      for (const point of line) {
        if (!point) { previous = false; continue; }
        const screen = this.toScreen(point);
        if (previous) ctx.lineTo(...screen);
        else ctx.moveTo(...screen);
        previous = true;
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = weight;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  mark(ctx, point, label, color) {
    const [x, y] = this.toScreen(point);
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.fillStyle = '#163026';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y + .5);
  }

  drawSelection() {
    if (!this.canvas.width) return;
    const ctx = this.context;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.backdrop, 0, 0);
    if (!this.data) return;
    ctx.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
    const raw = this.layers.gps ? this.gps[this.selected] : null;
    const snap = this.layers.route ? this.snapped[this.selected] : null;
    if (raw && snap) {
      ctx.setLineDash([3, 3]);
      this.drawLines(ctx, [[raw, snap]], '#fff4b8', 1.2);
      ctx.setLineDash([]);
    }
    for (const [point, color, size] of [[raw, '#ffb179', 4], [snap, '#def8a6', 5]]) {
      if (!point) continue;
      const [x, y] = this.toScreen(point);
      ctx.beginPath();
      ctx.arc(x, y, size + 3, 0, Math.PI * 2);
      ctx.fillStyle = `${color}44`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#122c24';
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
  }
}
