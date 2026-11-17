/*
 * 页面交互与可视化
 *  - 训练循环：requestAnimationFrame 驱动小批量梯度下降
 *  - 4 个视图：决策边界 / 损失与准确率 / 参数连接图 / 权重分布
 */
(function () {
  'use strict';

  var MLP = window.MLP.MLP;
  var makeDataset = window.MLP.makeDataset;

  var FONT = '"PingFang SC","Microsoft YaHei",system-ui,sans-serif';

  var $ = function (id) {
    return document.getElementById(id);
  };

  var state = {
    running: false,
    sizes: [2, 8, 8, 1],
    activation: 'relu',
    lr: 0.03,
    momentum: 0.9,
    l2: 0.0004,
    batch: 16,
    speed: 8,
    dataset: 'circle',
    noise: 0.06,
    samples: 400,
    step: 0,
    loss: 0,
    acc: 0,
    history: [],
    model: null,
    data: null,
  };

  var canvasEls = {
    boundary: $('cvBoundary'),
    loss: $('cvLoss'),
    net: $('cvNet'),
    hist: $('cvHist'),
  };

  var VIEW = { lo: -1.3, hi: 1.3 };
  var GRID = 76;

  var off = document.createElement('canvas');
  off.width = GRID;
  off.height = GRID;
  var offCtx = off.getContext('2d');
  var gridImg = offCtx.createImageData(GRID, GRID);

  var lastSample = 0;
  var lastBoundary = 0;
  var dirty = true;

  /* ---------------- 画布适配 ---------------- */

  function fitCanvas(canvas) {
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width));
    var h = Math.max(1, Math.round(rect.height));
    var pw = Math.round(w * dpr);
    var ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: w, h: h, ctx: ctx };
  }

  /* ---------------- 决策边界 ---------------- */

  function drawBoundary() {
    if (!state.model || !state.data) return;
    var view = fitCanvas(canvasEls.boundary);
    var w = view.w;
    var h = view.h;
    var ctx = view.ctx;
    var span = VIEW.hi - VIEW.lo;
    var model = state.model;
    var d = gridImg.data;

    for (var gy = 0; gy < GRID; gy++) {
      var y = VIEW.hi - span * ((gy + 0.5) / GRID);
      for (var gx = 0; gx < GRID; gx++) {
        var x = VIEW.lo + span * ((gx + 0.5) / GRID);
        var p = model.predict2(x, y);
        var conf = Math.abs(p - 0.5) * 2;
        var a = 0.13 + 0.52 * conf;
        var cr = 239 + (56 - 239) * p;
        var cg = 68 + (189 - 68) * p;
        var cb = 68 + (248 - 68) * p;
        var k = (gy * GRID + gx) * 4;
        d[k] = 9 + (cr - 9) * a;
        d[k + 1] = 13 + (cg - 13) * a;
        d[k + 2] = 24 + (cb - 24) * a;
        d[k + 3] = 255;
      }
    }

    offCtx.putImageData(gridImg, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, w, h);

    var X = state.data.X;
    var Y = state.data.Y;
    var n = state.data.n;
    for (var i = 0; i < n; i++) {
      var px = ((X[i * 2] - VIEW.lo) / span) * w;
      var py = (1 - (X[i * 2 + 1] - VIEW.lo) / span) * h;
      ctx.beginPath();
      ctx.arc(px, py, 2.7, 0, Math.PI * 2);
      ctx.fillStyle = Y[i] === 1 ? '#7dd3fc' : '#fda4af';
      ctx.fill();
      ctx.lineWidth = 0.9;
      ctx.strokeStyle = 'rgba(6,11,20,0.9)';
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(120,160,220,0.16)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  }

  /* ---------------- 损失 / 准确率曲线 ---------------- */

  function drawLoss() {
    var view = fitCanvas(canvasEls.loss);
    var w = view.w;
    var h = view.h;
    var ctx = view.ctx;
    ctx.clearRect(0, 0, w, h);

    var padL = 44;
    var padR = 42;
    var padT = 14;
    var padB = 24;
    var iw = w - padL - padR;
    var ih = h - padT - padB;
    if (iw < 30 || ih < 30) return;

    ctx.font = '10.5px ' + FONT;
    ctx.strokeStyle = 'rgba(120,160,220,0.10)';
    ctx.lineWidth = 1;
    for (var g = 0; g <= 4; g++) {
      var gy = Math.round(padT + (ih * g) / 4) + 0.5;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + iw, gy);
      ctx.stroke();
    }

    var hist = state.history;
    if (hist.length < 2) {
      ctx.fillStyle = 'rgba(139,160,187,0.9)';
      ctx.textAlign = 'center';
      ctx.fillText('点击「开始训练」，这里会画出损失与准确率', padL + iw / 2, padT + ih / 2);
      return;
    }

    var xMax = Math.max(1, hist[hist.length - 1].step);
    var lMax = 0.1;
    for (var i = 0; i < hist.length; i++) {
      if (hist[i].loss > lMax) lMax = hist[i].loss;
    }
    lMax *= 1.12;

    var xOf = function (s) {
      return padL + (s / xMax) * iw;
    };
    var yLoss = function (v) {
      return padT + ih - (Math.min(v, lMax) / lMax) * ih;
    };
    var yAcc = function (v) {
      return padT + ih - v * ih;
    };

    // 损失曲线下的渐变填充
    ctx.beginPath();
    ctx.moveTo(xOf(hist[0].step), padT + ih);
    for (var a = 0; a < hist.length; a++) ctx.lineTo(xOf(hist[a].step), yLoss(hist[a].loss));
    ctx.lineTo(xOf(hist[hist.length - 1].step), padT + ih);
    ctx.closePath();
    var grad = ctx.createLinearGradient(0, padT, 0, padT + ih);
    grad.addColorStop(0, 'rgba(96,165,250,0.24)');
    grad.addColorStop(1, 'rgba(96,165,250,0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // 损失
    ctx.beginPath();
    for (var b = 0; b < hist.length; b++) {
      var lx = xOf(hist[b].step);
      var ly = yLoss(hist[b].loss);
      if (b === 0) ctx.moveTo(lx, ly);
      else ctx.lineTo(lx, ly);
    }
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // 准确率
    ctx.beginPath();
    for (var c = 0; c < hist.length; c++) {
      var ax = xOf(hist[c].step);
      var ay = yAcc(hist[c].acc);
      if (c === 0) ctx.moveTo(ax, ay);
      else ctx.lineTo(ax, ay);
    }
    ctx.strokeStyle = 'rgba(52,211,153,0.9)';
    ctx.lineWidth = 1.4;
    ctx.stroke();

    // 坐标轴刻度
    ctx.fillStyle = 'rgba(139,160,187,0.9)';
    ctx.textAlign = 'right';
    for (var t = 0; t <= 4; t++) {
      var lv = lMax * (1 - t / 4);
      ctx.fillText(lv.toFixed(2), padL - 7, padT + (ih * t) / 4 + 3.5);
    }
    ctx.textAlign = 'left';
    for (var u = 0; u <= 4; u++) {
      ctx.fillText(Math.round((1 - u / 4) * 100) + '%', padL + iw + 7, padT + (ih * u) / 4 + 3.5);
    }
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(120,140,166,0.85)';
    ctx.fillText('训练步数 ' + xMax.toLocaleString('en-US'), padL + iw, h - 6);
  }

  /* ---------------- 参数连接图 ---------------- */

  function drawNetwork() {
    if (!state.model) return;
    var view = fitCanvas(canvasEls.net);
    var w = view.w;
    var h = view.h;
    var ctx = view.ctx;
    ctx.clearRect(0, 0, w, h);

    var sizes = state.model.sizes;
    var L = sizes.length;
    var padX = 54;
    var padTop = 22;
    var padBottom = 26;
    var usableH = h - padTop - padBottom;
    var maxN = Math.max.apply(null, sizes);
    var gap = maxN > 1 ? Math.min(30, usableH / (maxN - 1)) : 0;
    var r = Math.max(3.5, Math.min(8.5, gap * 0.36 || 8));

    var cols = [];
    for (var i = 0; i < L; i++) {
      var x = padX + ((w - padX * 2) * i) / (L - 1);
      var count = sizes[i];
      var total = (count - 1) * gap;
      var y0 = padTop + usableH / 2 - total / 2;
      var ys = [];
      for (var j = 0; j < count; j++) ys.push(y0 + j * gap);
      cols.push({ x: x, ys: ys });
    }

    // 权重连线：颜色表示符号，粗细表示绝对值
    for (var l = 0; l < state.model.W.length; l++) {
      var W = state.model.W[l];
      var nIn = sizes[l];
      var nOut = sizes[l + 1];
      var A = cols[l];
      var B = cols[l + 1];
      for (var o = 0; o < nOut; o++) {
        for (var ii = 0; ii < nIn; ii++) {
          var wv = W[o * nIn + ii];
          var mag = Math.min(1, Math.abs(wv) / 1.1);
          if (mag < 0.02) continue;
          var alpha = 0.07 + 0.62 * mag;
          ctx.strokeStyle = wv >= 0
            ? 'rgba(56,189,248,' + alpha.toFixed(3) + ')'
            : 'rgba(248,113,113,' + alpha.toFixed(3) + ')';
          ctx.lineWidth = 0.4 + 2.1 * mag;
          ctx.beginPath();
          ctx.moveTo(A.x + r, A.ys[ii]);
          ctx.lineTo(B.x - r, B.ys[o]);
          ctx.stroke();
        }
      }
    }

    // 神经元节点：填充深浅表示偏置大小
    ctx.textAlign = 'center';
    ctx.font = '10.5px ' + FONT;
    for (var p = 0; p < L; p++) {
      var col = cols[p];
      for (var q = 0; q < col.ys.length; q++) {
        var fill = 'rgba(148,197,255,0.9)';
        if (p > 0) {
          var bv = Math.tanh(state.model.b[p - 1][q]);
          var ba = 0.28 + 0.55 * Math.abs(bv);
          fill = bv >= 0
            ? 'rgba(56,189,248,' + ba.toFixed(3) + ')'
            : 'rgba(248,113,113,' + ba.toFixed(3) + ')';
        }
        ctx.beginPath();
        ctx.arc(col.x, col.ys[q], r, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 1.1;
        ctx.strokeStyle = 'rgba(7,12,21,0.9)';
        ctx.stroke();
      }
      var label = p === 0 ? '输入' : (p === L - 1 ? '输出' : '隐藏 ' + p);
      ctx.fillStyle = 'rgba(139,160,187,0.95)';
      ctx.fillText(label + ' · ' + sizes[p], col.x, h - 8);
    }
  }

  /* ---------------- 权重分布 ---------------- */

  function drawHist() {
    if (!state.model) return;
    var view = fitCanvas(canvasEls.hist);
    var w = view.w;
    var h = view.h;
    var ctx = view.ctx;
    ctx.clearRect(0, 0, w, h);

    var vals = state.model.flatWeights();
    var st = state.model.stats();
    var range = Math.max(0.25, st.maxAbs * 1.05);
    var BINS = 37;
    var bins = new Float32Array(BINS);
    for (var i = 0; i < vals.length; i++) {
      var bi = Math.floor(((vals[i] + range) / (2 * range)) * BINS);
      if (bi < 0) bi = 0;
      if (bi >= BINS) bi = BINS - 1;
      bins[bi]++;
    }
    var maxC = 1;
    for (var j = 0; j < BINS; j++) if (bins[j] > maxC) maxC = bins[j];

    var padL = 12;
    var padR = 12;
    var padT = 16;
    var padB = 42;
    var iw = w - padL - padR;
    var ih = h - padT - padB;
    if (iw < 30 || ih < 24) return;

    var baseY = padT + ih;
    var bw = iw / BINS;
    for (var k = 0; k < BINS; k++) {
      var bh = (bins[k] / maxC) * ih;
      if (bh < 0.8) continue;
      var bx = padL + k * bw;
      ctx.fillStyle = k >= BINS / 2 ? 'rgba(56,189,248,0.72)' : 'rgba(248,113,113,0.72)';
      ctx.fillRect(bx + 0.7, baseY - bh, Math.max(1, bw - 1.4), bh);
    }

    // 零刻度线
    ctx.strokeStyle = 'rgba(160,190,225,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL + iw / 2, padT - 4);
    ctx.lineTo(padL + iw / 2, baseY);
    ctx.stroke();

    ctx.font = '10px ' + FONT;
    ctx.fillStyle = 'rgba(139,160,187,0.85)';
    ctx.textAlign = 'left';
    ctx.fillText('-' + range.toFixed(1), padL, baseY + 12);
    ctx.textAlign = 'right';
    ctx.fillText('+' + range.toFixed(1), padL + iw, baseY + 12);

    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(178,197,220,0.95)';
    ctx.fillText('权重 ' + st.count + ' 个 · 均值 ' + st.mean.toFixed(3) + ' · 标准差 ' + st.std.toFixed(3), padL, h - 16);
    ctx.fillStyle = 'rgba(139,160,187,0.85)';
    ctx.fillText('偏置均值 ' + st.biasMean.toFixed(3) + ' · 最大 |w| ' + st.maxAbs.toFixed(2), padL, h - 4);
  }

  /* ---------------- 指标与训练 ---------------- */

  function updateMetrics() {
    $('mSteps').textContent = state.step.toLocaleString('en-US');
    $('mLoss').textContent = state.history.length ? state.loss.toFixed(3) : '—';
    $('mAcc').textContent = state.history.length ? (state.acc * 100).toFixed(1) + '%' : '—';
    $('mParams').textContent = state.model ? state.model.paramCount : '—';
  }

  function trainSteps() {
    var opts = {
      batch: state.batch,
      lr: state.lr,
      momentum: state.momentum,
      l2: state.l2,
    };
    for (var i = 0; i < state.speed; i++) {
      state.model.trainStep(state.data, opts);
      state.step++;
    }
  }

  function samplePoint() {
    var ev = state.model.evaluate(state.data);
    state.loss = ev.loss;
    state.acc = ev.acc;
    state.history.push({ step: state.step, loss: ev.loss, acc: ev.acc });
    if (state.history.length > 900) {
      state.history = state.history.filter(function (_, i) {
        return i % 2 === 0;
      });
    }
    updateMetrics();
  }

  /* ---------------- 重建 ---------------- */

  function rebuildModel() {
    state.model = new MLP(state.sizes, {
      activation: state.activation,
      seed: (Math.random() * 1e9) | 0,
    });
  }

  function rebuildData() {
    state.data = makeDataset(state.dataset, state.samples, state.noise, (Math.random() * 1e9) | 0);
  }

  function reset(rebuildDataToo) {
    if (rebuildDataToo || !state.data) rebuildData();
    rebuildModel();
    state.step = 0;
    state.loss = 0;
    state.acc = 0;
    state.history = [];
    $('archText').textContent = state.sizes.join(' → ');
    updateMetrics();
    dirty = true;
  }

  /* ---------------- 主循环 ---------------- */

  function frame(ts) {
    if (state.running) {
      trainSteps();
      if (ts - lastSample > 90) {
        samplePoint();
        lastSample = ts;
      }
      dirty = true;
    }
    if (dirty) {
      if (!state.running || ts - lastBoundary > 90) {
        drawBoundary();
        lastBoundary = ts;
      }
      drawLoss();
      drawNetwork();
      drawHist();
      dirty = false;
    }
    requestAnimationFrame(frame);
  }

  /* ---------------- 控件绑定 ---------------- */

  function fmtLr(v) {
    return v < 0.01 ? v.toFixed(4) : v.toFixed(3);
  }

  function syncStructure() {
    var layers = parseInt($('layers').value, 10);
    var neurons = parseInt($('neurons').value, 10);
    var sizes = [2];
    for (var i = 0; i < layers; i++) sizes.push(neurons);
    sizes.push(1);
    state.sizes = sizes;
    reset(false);
  }

  $('datasetSeg').addEventListener('click', function (e) {
    var btn = e.target.closest('button[data-kind]');
    if (!btn) return;
    state.dataset = btn.getAttribute('data-kind');
    var kids = $('datasetSeg').children;
    for (var i = 0; i < kids.length; i++) kids[i].classList.toggle('on', kids[i] === btn);
    reset(true);
  });

  $('noise').addEventListener('input', function (e) {
    state.noise = parseFloat(e.target.value);
    $('vNoise').textContent = state.noise.toFixed(2);
    reset(true);
  });

  $('samples').addEventListener('input', function (e) {
    state.samples = parseInt(e.target.value, 10);
    $('vSamples').textContent = String(state.samples);
    reset(true);
  });

  $('layers').addEventListener('change', syncStructure);
  $('neurons').addEventListener('change', syncStructure);

  $('activation').addEventListener('change', function (e) {
    state.activation = e.target.value;
    reset(false);
  });

  $('lr').addEventListener('input', function (e) {
    state.lr = Math.pow(10, -3 + 2.7 * (parseInt(e.target.value, 10) / 100));
    $('vLr').textContent = fmtLr(state.lr);
    dirty = true;
  });

  $('mom').addEventListener('input', function (e) {
    state.momentum = parseFloat(e.target.value);
    $('vMom').textContent = state.momentum.toFixed(2);
    dirty = true;
  });

  $('l2').addEventListener('input', function (e) {
    state.l2 = parseFloat(e.target.value);
    $('vL2').textContent = state.l2.toFixed(4);
    dirty = true;
  });

  $('batch').addEventListener('change', function (e) {
    state.batch = parseInt(e.target.value, 10);
  });

  $('speed').addEventListener('input', function (e) {
    state.speed = parseInt(e.target.value, 10);
    $('vSpeed').textContent = state.speed + ' 步/帧';
  });

  $('btnToggle').addEventListener('click', function () {
    state.running = !state.running;
    var btn = $('btnToggle');
    btn.textContent = state.running ? '暂停训练' : '继续训练';
    btn.classList.toggle('paused', state.running);
    dirty = true;
  });

  $('btnStep').addEventListener('click', function () {
    state.model.trainStep(state.data, {
      batch: state.batch,
      lr: state.lr,
      momentum: state.momentum,
      l2: state.l2,
    });
    state.step++;
    samplePoint();
    dirty = true;
  });

  $('btnReset').addEventListener('click', function () {
    reset(false);
  });

  window.addEventListener('resize', function () {
    dirty = true;
  });

  /* ---------------- 启动 ---------------- */

  state.lr = Math.pow(10, -3 + 2.7 * (parseInt($('lr').value, 10) / 100));
  $('vLr').textContent = fmtLr(state.lr);
  rebuildData();
  rebuildModel();
  reset(false);
  updateMetrics();
  requestAnimationFrame(frame);
})();
