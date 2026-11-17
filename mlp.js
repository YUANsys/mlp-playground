/*
 * MLP 从零实现
 * ------------------------------------------------------------
 * 仅使用原生 JavaScript，不依赖任何深度学习框架。
 *  - 多层感知机：任意层数 / 神经元数 / ReLU·Tanh·Sigmoid
 *  - 输出层：Sigmoid + 二元交叉熵（BCE），梯度恰为 p - y
 *  - 优化器：小批量动量 SGD，可选 L2 正则
 *  - 提供参数统计与二分类数据生成器，供可视化使用
 */
(function (root) {
  'use strict';

  /* ---------------- 随机数工具 ---------------- */

  // 可复现的伪随机数（mulberry32）
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // 标准正态分布（Box-Muller）
  function gauss(rng) {
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  /* ---------------- 激活函数 ---------------- */

  const ACTIVATIONS = {
    relu: { name: 'ReLU' },
    tanh: { name: 'Tanh' },
    sigmoid: { name: 'Sigmoid' },
  };

  /* ---------------- 多层感知机 ---------------- */

  class MLP {
    /**
     * @param {number[]} sizes 各层神经元数量，例如 [2, 8, 8, 1]
     * @param {{activation?: string, seed?: number}} opts
     */
    constructor(sizes, opts) {
      opts = opts || {};
      this.sizes = sizes.slice();
      this.activationName = ACTIVATIONS[opts.activation] ? opts.activation : 'relu';
      const rng = mulberry32(opts.seed == null ? 20261117 : opts.seed);

      const L = this.sizes.length - 1; // 权重层数
      this.W = [];   // 权重：W[l][o * nIn + i]
      this.b = [];   // 偏置
      this.vW = [];  // 权重动量缓存
      this.vb = [];  // 偏置动量缓存
      this.gW = [];  // 权重梯度累加
      this.gb = [];  // 偏置梯度累加
      this.z = [];   // 各层线性输出（未激活）
      this.a = [new Float32Array(this.sizes[0])]; // 各层激活值，a[0] 为输入
      this.delta = []; // 各层误差信号

      for (let l = 0; l < L; l++) {
        const nIn = this.sizes[l];
        const nOut = this.sizes[l + 1];
        // He 初始化（ReLU）/ Xavier 初始化（Tanh、Sigmoid）
        const scale = this.activationName === 'relu'
          ? Math.sqrt(2 / nIn)
          : Math.sqrt(1 / nIn);

        const W = new Float32Array(nIn * nOut);
        for (let i = 0; i < W.length; i++) W[i] = gauss(rng) * scale;

        this.W.push(W);
        this.b.push(new Float32Array(nOut));
        this.vW.push(new Float32Array(nIn * nOut));
        this.vb.push(new Float32Array(nOut));
        this.gW.push(new Float32Array(nIn * nOut));
        this.gb.push(new Float32Array(nOut));
        this.z.push(new Float32Array(nOut));
        this.a.push(new Float32Array(nOut));
        this.delta.push(new Float32Array(nOut));
      }

      this.input = new Float32Array(this.sizes[0]);
      this.paramCount = 0;
      for (let l = 0; l < L; l++) {
        this.paramCount += this.sizes[l] * this.sizes[l + 1] + this.sizes[l + 1];
      }

      this.epoch = 0;      // 完整遍历数据集的次数
      this.stepCount = 0;  // 参数更新次数
      this.perm = null;    // 样本随机遍历顺序
      this.cursor = 0;
    }

    /** 前向传播，返回输出层概率 p */
    forward(x) {
      const a = this.a;
      const z = this.z;
      const W = this.W;
      const b = this.b;
      const L = W.length;
      const mode = this.activationName;

      a[0][0] = x[0];
      a[0][1] = x[1];

      for (let l = 0; l < L; l++) {
        const nIn = this.sizes[l];
        const nOut = this.sizes[l + 1];
        const Wl = W[l];
        const bl = b[l];
        const zl = z[l];
        const al = a[l + 1];
        const prev = a[l];
        const isOut = l === L - 1;

        for (let o = 0; o < nOut; o++) {
          let s = bl[o];
          const off = o * nIn;
          for (let i = 0; i < nIn; i++) s += Wl[off + i] * prev[i];
          zl[o] = s;
          if (isOut) {
            al[o] = sigmoid(s);
          } else if (mode === 'relu') {
            al[o] = s > 0 ? s : 0;
          } else if (mode === 'tanh') {
            al[o] = Math.tanh(s);
          } else {
            al[o] = sigmoid(s);
          }
        }
      }
      return a[L][0];
    }

    /** 二维输入的便捷前向 */
    predict2(x, y) {
      this.input[0] = x;
      this.input[1] = y;
      return this.forward(this.input);
    }

    /** 反向传播：累加单个样本的梯度 */
    backward(y) {
      const L = this.W.length;
      const a = this.a;
      const depth = this.delta;
      const W = this.W;

      // 输出层：Sigmoid + BCE 的梯度
      depth[L - 1][0] = a[L][0] - y;

      for (let l = L - 1; l >= 0; l--) {
        const nIn = this.sizes[l];
        const nOut = this.sizes[l + 1];
        const dl = depth[l];
        const gWl = this.gW[l];
        const gbl = this.gb[l];
        const prev = a[l];

        // 累加当前层权重的梯度
        for (let o = 0; o < nOut; o++) {
          const d = dl[o];
          if (d === 0) continue;
          const off = o * nIn;
          gbl[o] += d;
          for (let i = 0; i < nIn; i++) gWl[off + i] += d * prev[i];
        }

        // 误差信号回传到上一层
        if (l > 0) {
          const pd = depth[l - 1];
          const act = a[l];        // 本层激活值，用于求导
          const Wl = W[l];
          for (let i = 0; i < nIn; i++) {
            let s = 0;
            for (let o = 0; o < nOut; o++) s += Wl[o * nIn + i] * dl[o];
            if (this.activationName === 'relu') {
              s *= act[i] > 0 ? 1 : 0;
            } else if (this.activationName === 'tanh') {
              s *= 1 - act[i] * act[i];
            } else {
              s *= act[i] * (1 - act[i]);
            }
            pd[i] = s;
          }
        }
      }
    }

    /**
     * 小批量训练一步
     * @param {{X: Float32Array, Y: Uint8Array, n: number}} data
     * @param {{batch:number, lr:number, momentum:number, l2:number}} opts
     * @returns {number} 当前批次的平均损失
     */
    trainStep(data, opts) {
      const X = data.X;
      const Y = data.Y;
      const n = data.n;
      const bs = Math.max(1, Math.min(opts.batch | 0, n));

      if (!this.perm || this.perm.length !== n) {
        this.perm = new Uint32Array(n);
        for (let i = 0; i < n; i++) this.perm[i] = i;
        this.cursor = 0;
      }
      // 遍历完一轮后打乱顺序
      if (this.cursor + bs > n) {
        for (let i = n - 1; i > 0; i--) {
          const j = (Math.random() * (i + 1)) | 0;
          const t = this.perm[i];
          this.perm[i] = this.perm[j];
          this.perm[j] = t;
        }
        this.cursor = 0;
        this.epoch++;
      }

      const start = this.cursor;
      this.cursor += bs;

      // 清零梯度
      for (let l = 0; l < this.W.length; l++) {
        this.gW[l].fill(0);
        this.gb[l].fill(0);
      }

      let loss = 0;
      for (let k = start; k < start + bs; k++) {
        const idx = this.perm[k];
        const p = this.predict2(X[idx * 2], X[idx * 2 + 1]);
        const t = Y[idx];
        const pc = clamp(p, 1e-7, 1 - 1e-7);
        loss += -(t * Math.log(pc) + (1 - t) * Math.log(1 - pc));
        this.backward(t);
      }

      const lr = opts.lr;
      const mu = opts.momentum;
      const l2 = opts.l2 || 0;
      const inv = 1 / bs;

      // 动量 SGD 更新参数
      for (let l = 0; l < this.W.length; l++) {
        const Wl = this.W[l];
        const gWl = this.gW[l];
        const vWl = this.vW[l];
        for (let i = 0; i < Wl.length; i++) {
          let g = gWl[i] * inv;
          if (l2 > 0) g += l2 * Wl[i];
          const v = mu * vWl[i] - lr * g;
          vWl[i] = v;
          Wl[i] += v;
        }
        const bl = this.b[l];
        const gbl = this.gb[l];
        const vbl = this.vb[l];
        for (let o = 0; o < bl.length; o++) {
          const v = mu * vbl[o] - lr * gbl[o] * inv;
          vbl[o] = v;
          bl[o] += v;
        }
      }

      this.stepCount++;
      return loss * inv;
    }

    /** 在完整数据集上评估损失与准确率 */
    evaluate(data) {
      const X = data.X;
      const Y = data.Y;
      let correct = 0;
      let loss = 0;
      for (let i = 0; i < data.n; i++) {
        const p = this.predict2(X[i * 2], X[i * 2 + 1]);
        const pc = clamp(p, 1e-7, 1 - 1e-7);
        const t = Y[i];
        loss += -(t * Math.log(pc) + (1 - t) * Math.log(1 - pc));
        if ((p >= 0.5 ? 1 : 0) === t) correct++;
      }
      return { loss: loss / data.n, acc: correct / data.n };
    }

    /** 全部权重展平（用于直方图） */
    flatWeights() {
      let total = 0;
      for (let l = 0; l < this.W.length; l++) total += this.W[l].length;
      const out = new Float32Array(total);
      let k = 0;
      for (let l = 0; l < this.W.length; l++) {
        out.set(this.W[l], k);
        k += this.W[l].length;
      }
      return out;
    }

    /** 参数统计 */
    stats() {
      let sum = 0;
      let sum2 = 0;
      let count = 0;
      let maxAbs = 0;
      for (let l = 0; l < this.W.length; l++) {
        const W = this.W[l];
        for (let i = 0; i < W.length; i++) {
          const v = W[i];
          sum += v;
          sum2 += v * v;
          count++;
          if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
        }
      }
      const mean = sum / count;
      const std = Math.sqrt(Math.max(0, sum2 / count - mean * mean));
      let bSum = 0;
      let bCount = 0;
      for (let l = 0; l < this.b.length; l++) {
        const b = this.b[l];
        for (let i = 0; i < b.length; i++) {
          bSum += b[i];
          bCount++;
        }
      }
      return { mean, std, maxAbs, count, biasMean: bSum / bCount };
    }
  }

  /* ---------------- 二分类数据集 ---------------- */

  const DATASETS = [
    { id: 'circle', name: '同心圆' },
    { id: 'xor', name: '异或' },
    { id: 'spiral', name: '双螺旋' },
    { id: 'moons', name: '双月' },
    { id: 'blobs', name: '双高斯团' },
  ];

  /**
   * 生成坐标为 [-1, 1] 附近的二分类数据
   */
  function makeDataset(kind, n, noise, seed) {
    const rng = mulberry32((seed >>> 0) || 1);
    const X = new Float32Array(n * 2);
    const Y = new Uint8Array(n);

    for (let k = 0; k < n; k++) {
      let x = 0;
      let y = 0;
      let label = 0;

      if (kind === 'circle') {
        const t = rng() * Math.PI * 2;
        const inner = rng() < 0.5;
        const r = inner ? Math.sqrt(rng()) * 0.38 : 0.62 + Math.sqrt(rng()) * 0.28;
        x = Math.cos(t) * r;
        y = Math.sin(t) * r;
        label = inner ? 0 : 1;
      } else if (kind === 'xor') {
        const sx = rng() < 0.5 ? -1 : 1;
        const sy = rng() < 0.5 ? -1 : 1;
        x = sx * (0.42 + Math.abs(gauss(rng)) * 0.2);
        y = sy * (0.42 + Math.abs(gauss(rng)) * 0.2);
        label = sx * sy > 0 ? 1 : 0;
      } else if (kind === 'spiral') {
        const cls = k % 2;
        const half = Math.max(1, Math.floor(n / 2));
        const frac = Math.floor(k / 2) / (half - 1 || 1);
        const r = 0.06 + frac * 0.94;
        const t = frac * 1.85 * Math.PI * 2 + cls * Math.PI;
        x = r * Math.cos(t);
        y = r * Math.sin(t);
        label = cls;
      } else if (kind === 'moons') {
        const cls = k % 2;
        const t = rng() * Math.PI;
        if (cls === 0) {
          x = Math.cos(t) * 0.6 - 0.26;
          y = Math.sin(t) * 0.6 - 0.22;
        } else {
          x = -Math.cos(t) * 0.6 + 0.26;
          y = -Math.sin(t) * 0.6 + 0.2 + 0.34;
        }
        label = cls;
      } else {
        const cls = rng() < 0.5 ? 0 : 1;
        x = (cls === 0 ? -0.42 : 0.42) + gauss(rng) * 0.19;
        y = (cls === 0 ? -0.38 : 0.38) + gauss(rng) * 0.19;
        label = cls;
      }

      X[k * 2] = clamp(x + gauss(rng) * noise, -1.35, 1.35);
      X[k * 2 + 1] = clamp(y + gauss(rng) * noise, -1.35, 1.35);
      Y[k] = label;
    }

    return { X, Y, n, kind };
  }

  const api = { MLP, makeDataset, DATASETS, ACTIVATIONS };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.MLP = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
