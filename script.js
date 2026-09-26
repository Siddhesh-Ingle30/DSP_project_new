const $ = (id) => document.getElementById(id);

let currentObjectUrls = [];
let lastResult = null;

function setStatus(message, progress = null) {
  $("status").textContent = message;
  if (progress !== null) {
    $("progressWrap").classList.remove("hidden");
    $("progressBar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
  }
}

function setFileName(inputId, labelId) {
  $(inputId).addEventListener("change", () => {
    const file = $(inputId).files[0];
    $(labelId).textContent = file ? file.name : "Choose a WAV file";
  });
}
setFileName("mixFile", "mixName");
setFileName("speechFile", "speechName");
setFileName("musicFile", "musicName");

function nextOdd(n) {
  n = Math.max(5, Math.floor(n));
  return n % 2 === 0 ? n + 1 : n;
}

function normalize(x, peak = 0.95) {
  let max = 0;
  for (let i = 0; i < x.length; i++) max = Math.max(max, Math.abs(x[i]));
  if (!max) return x.slice();
  const scale = peak / max;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale;
  return out;
}

function downmix(buffer) {
  const channels = buffer.numberOfChannels;
  const n = buffer.length;
  const out = new Float32Array(n);
  for (let c = 0; c < channels; c++) {
    const ch = buffer.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += ch[i] / channels;
  }
  return out;
}

async function decodeFile(file) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const data = await file.arrayBuffer();
  const decoded = await ctx.decodeAudioData(data.slice(0));
  const samples = downmix(decoded);
  const fs = decoded.sampleRate;
  await ctx.close();
  return { samples: normalize(samples), fs };
}

/* -------------------- WAV writer -------------------- */

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 32768 : s * 32767, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function attachAudio(audioId, downloadId, samples, fs, filename) {
  const blob = encodeWav(samples, fs);
  const url = URL.createObjectURL(blob);
  currentObjectUrls.push(url);
  $(audioId).src = url;
  $(downloadId).href = url;
  $(downloadId).download = filename;
}

/* -------------------- FIR design -------------------- */

function designFIRLowpass(taps, cutoff, fs) {
  taps = nextOdd(taps);
  const h = new Float32Array(taps);
  const M = (taps - 1) / 2;
  const fc = cutoff / fs;

  for (let n = 0; n < taps; n++) {
    const k = n - M;
    let sinc;
    if (k === 0) sinc = 2 * fc;
    else sinc = Math.sin(2 * Math.PI * fc * k) / (Math.PI * k);

    const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (taps - 1));
    h[n] = sinc * w;
  }

  // Normalize DC gain.
  let sum = 0;
  for (let i = 0; i < taps; i++) sum += h[i];
  for (let i = 0; i < taps; i++) h[i] /= sum;

  return h;
}

function designFIRHighpass(taps, cutoff, fs) {
  const low = designFIRLowpass(taps, cutoff, fs);
  const high = new Float32Array(taps);
  const mid = (taps - 1) / 2;
  for (let i = 0; i < taps; i++) high[i] = -low[i];
  high[mid] += 1;
  return high;
}

/*
  FIR filtering uses the browser's native ConvolverNode. This performs
  convolution efficiently in the browser instead of a very slow
  JavaScript sample-by-sample convolution.
*/
async function convolveOffline(input, kernel, fs) {
  const ctx = new OfflineAudioContext(1, input.length + kernel.length - 1, fs);
  const source = ctx.createBufferSource();
  const inputBuffer = ctx.createBuffer(1, input.length, fs);
  inputBuffer.copyToChannel(input, 0);

  const impulse = ctx.createBuffer(1, kernel.length, fs);
  impulse.copyToChannel(kernel, 0);

  const convolver = ctx.createConvolver();
  convolver.normalize = false;
  convolver.buffer = impulse;

  source.buffer = inputBuffer;
  source.connect(convolver);
  convolver.connect(ctx.destination);
  source.start();

  const rendered = await ctx.startRendering();
  const data = rendered.getChannelData(0);
  return data.slice(0, input.length);
}

/* -------------------- IIR Butterworth -------------------- */

function butterworthQValues(order) {
  const q = [];
  const sections = order / 2;
  for (let k = 0; k < sections; k++) {
    const theta = Math.PI * (2 * k + 1) / (2 * order);
    q.push(1 / (2 * Math.cos(theta)));
  }
  return q;
}

function processBiquad(input, fs, cutoff, type, Q) {
  // RBJ cookbook coefficients.
  const w0 = 2 * Math.PI * cutoff / fs;
  const cos = Math.cos(w0);
  const sin = Math.sin(w0);
  const alpha = sin / (2 * Q);

  let b0, b1, b2, a0, a1, a2;

  if (type === "lowpass") {
    b0 = (1 - cos) / 2;
    b1 = 1 - cos;
    b2 = (1 - cos) / 2;
  } else {
    b0 = (1 + cos) / 2;
    b1 = -(1 + cos);
    b2 = (1 + cos) / 2;
  }

  a0 = 1 + alpha;
  a1 = -2 * cos;
  a2 = 1 - alpha;

  b0 /= a0; b1 /= a0; b2 /= a0;
  a1 /= a0; a2 /= a0;

  const out = new Float32Array(input.length);

  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < input.length; i++) {
    const x0 = input[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    out[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }
  return out;
}

function iirFilter(input, fs, cutoff, order, type) {
  order = Math.max(2, Math.min(8, Math.floor(order)));
  if (order % 2 !== 0) order++;
  const qValues = butterworthQValues(order);
  let out = input.slice();

  for (const q of qValues) {
    out = processBiquad(out, fs, cutoff, type, q);
  }
  return out;
}

/* -------------------- FFT -------------------- */

function reverseBits(x, bits) {
  let y = 0;
  for (let i = 0; i < bits; i++) {
    y = (y << 1) | (x & 1);
    x >>>= 1;
  }
  return y;
}

function fft(re, im, inverse = false) {
  const n = re.length;
  let bits = Math.round(Math.log2(n));

  for (let i = 0; i < n; i++) {
    const j = reverseBits(i, bits);
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const sign = inverse ? 1 : -1;
    const angle = sign * 2 * Math.PI / size;
    const wpr = Math.cos(angle);
    const wpi = Math.sin(angle);

    for (let start = 0; start < n; start += size) {
      let wr = 1, wi = 0;
      for (let j = 0; j < half; j++) {
        const even = start + j;
        const odd = even + half;

        const tr = wr * re[odd] - wi * im[odd];
        const ti = wr * im[odd] + wi * re[odd];

        const er = re[even], ei = im[even];
        re[even] = er + tr;
        im[even] = ei + ti;
        re[odd] = er - tr;
        im[odd] = ei - ti;

        const nextWr = wr * wpr - wi * wpi;
        wi = wr * wpi + wi * wpr;
        wr = nextWr;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/*
  Memory-safe FFT separation:
  The long audio is processed in overlapping FFT frames instead of
  allocating one enormous FFT for the entire recording.
*/
async function fftSeparation(input, fs, cutoff, report) {
  const frameSize = 16384;
  const hop = frameSize / 2;
  const nFrames = Math.ceil((input.length - frameSize) / hop) + 1;

  const speech = new Float32Array(input.length);
  const music = new Float32Array(input.length);
  const weight = new Float32Array(input.length);

  const window = new Float32Array(frameSize);
  for (let n = 0; n < frameSize; n++) {
    window[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / (frameSize - 1));
  }

  for (let frame = 0; frame < nFrames; frame++) {
    const start = frame * hop;
    const re = new Float64Array(frameSize);
    const im = new Float64Array(frameSize);

    for (let n = 0; n < frameSize; n++) {
      const idx = start + n;
      re[n] = idx < input.length ? input[idx] * window[n] : 0;
    }

    fft(re, im, false);

    for (let k = 0; k < frameSize; k++) {
      const freq = (k <= frameSize / 2)
        ? k * fs / frameSize
        : (k - frameSize) * fs / frameSize;

      if (Math.abs(freq) <= cutoff) {
        im[k] = im[k];
      } else {
        re[k] = re[k];
        im[k] = im[k];
      }
    }

    // Preserve the original spectrum separately for the two binary masks.
    const sr = new Float64Array(frameSize);
    const si = new Float64Array(frameSize);
    const mr = new Float64Array(frameSize);
    const mi = new Float64Array(frameSize);

    for (let k = 0; k < frameSize; k++) {
      const freq = (k <= frameSize / 2)
        ? k * fs / frameSize
        : (k - frameSize) * fs / frameSize;

      if (Math.abs(freq) <= cutoff) {
        sr[k] = re[k]; si[k] = im[k];
      } else {
        mr[k] = re[k]; mi[k] = im[k];
      }
    }

    fft(sr, si, true);
    fft(mr, mi, true);

    for (let n = 0; n < frameSize; n++) {
      const idx = start + n;
      if (idx < input.length) {
        speech[idx] += sr[n] * window[n];
        music[idx] += mr[n] * window[n];
        weight[idx] += window[n] * window[n];
      }
    }

    if (report && (frame % 4 === 0 || frame === nFrames - 1)) {
      report(25 + 35 * ((frame + 1) / nFrames));
      await new Promise(r => setTimeout(r, 0));
    }
  }

  for (let i = 0; i < input.length; i++) {
    const w = weight[i];
    if (w > 1e-8) {
      speech[i] /= w;
      music[i] /= w;
    }
  }

  return {
    speech: normalize(speech),
    music: normalize(music)
  };
}

/* -------------------- Metrics -------------------- */

function correlation(reference, estimated) {
  const n = Math.min(reference.length, estimated.length);
  let mr = 0, me = 0;
  for (let i = 0; i < n; i++) {
    mr += reference[i];
    me += estimated[i];
  }
  mr /= n; me /= n;

  let num = 0, dr = 0, de = 0;
  for (let i = 0; i < n; i++) {
    const a = reference[i] - mr;
    const b = estimated[i] - me;
    num += a * b;
    dr += a * a;
    de += b * b;
  }
  const den = Math.sqrt(dr * de);
  return den ? num / den : 0;
}

function snr(reference, estimated) {
  const n = Math.min(reference.length, estimated.length);
  let dot = 0, ee = 0;
  for (let i = 0; i < n; i++) {
    dot += reference[i] * estimated[i];
    ee += estimated[i] * estimated[i];
  }
  const scale = dot / (ee + 1e-12);

  let signalPower = 0, noisePower = 0;
  for (let i = 0; i < n; i++) {
    const err = reference[i] - scale * estimated[i];
    signalPower += reference[i] * reference[i];
    noisePower += err * err;
  }
  signalPower /= n;
  noisePower /= n;
  return noisePower > 0 ? 10 * Math.log10(signalPower / noisePower) : Infinity;
}

function rmse(reference, estimated) {
  const n = Math.min(reference.length, estimated.length);
  let dot = 0, ee = 0;
  for (let i = 0; i < n; i++) {
    dot += reference[i] * estimated[i];
    ee += estimated[i] * estimated[i];
  }
  const scale = dot / (ee + 1e-12);

  let mse = 0;
  for (let i = 0; i < n; i++) {
    const e = reference[i] - scale * estimated[i];
    mse += e * e;
  }
  return Math.sqrt(mse / n);
}

function fmt(x) {
  return Number.isFinite(x) ? x.toFixed(4) : "∞";
}

function metricsRow(method, speech, music, speechRef, musicRef) {
  if (!speechRef || !musicRef) {
    return `<tr><td>${method}</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>`;
  }
  return `<tr>
    <td>${method}</td>
    <td>${fmt(correlation(speechRef, speech))}</td>
    <td>${fmt(correlation(musicRef, music))}</td>
    <td>${fmt(snr(speechRef, speech))}</td>
    <td>${fmt(snr(musicRef, music))}</td>
    <td>${fmt(rmse(speechRef, speech))}</td>
    <td>${fmt(rmse(musicRef, music))}</td>
  </tr>`;
}

/* -------------------- Canvas graphs -------------------- */

function clearCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(500, Math.floor(rect.width));
  const h = Math.max(230, Math.floor(rect.height));
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawAxes(ctx, w, h, xLabel, yLabel) {
  const left = 52, right = 15, top = 18, bottom = 38;
  ctx.strokeStyle = "#d9dce7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, h - bottom);
  ctx.lineTo(w - right, h - bottom);
  ctx.stroke();

  ctx.fillStyle = "#68718b";
  ctx.font = "11px Arial";
  ctx.fillText(xLabel, w - 78, h - 10);
  ctx.save();
  ctx.translate(14, h / 2 + 25);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  return { left, right, top, bottom };
}

function drawWave(canvasId, data, fs, title) {
  const { ctx, w, h } = clearCanvas($(canvasId));
  const a = drawAxes(ctx, w, h, "Time (s)", "Amplitude");

  const maxSeconds = Math.min(30, data.length / fs);
  const end = Math.max(1, Math.floor(maxSeconds * fs));
  const points = Math.min(5000, end);
  const step = Math.max(1, Math.floor(end / points));

  let min = Infinity, max = -Infinity;
  for (let i = 0; i < end; i += step) {
    const v = data[i];
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  if (Math.abs(max - min) < 1e-8) { min = -1; max = 1; }

  ctx.strokeStyle = "#6540c7";
  ctx.lineWidth = 1.1;
  ctx.beginPath();

  for (let p = 0, i = 0; i < end; p++, i += step) {
    const x = a.left + (p / Math.max(1, points - 1)) * (w - a.left - a.right);
    const y = a.top + (1 - (data[i] - min) / (max - min)) * (h - a.top - a.bottom);
    if (p === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "#17203a";
  ctx.font = "bold 12px Arial";
  ctx.fillText(title, a.left, 13);
}

function magnitudeSpectrum(data, fs) {
  const maxSeconds = Math.min(60, data.length / fs);
  const count = Math.min(data.length, Math.floor(maxSeconds * fs), 262144);
  let n = 1;
  while ((n << 1) <= count) n <<= 1;

  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const win = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
    re[i] = data[i] * win;
  }
  fft(re, im, false);

  const half = n / 2;
  const freq = new Float32Array(half);
  const mag = new Float32Array(half);
  for (let k = 0; k < half; k++) {
    freq[k] = k * fs / n;
    mag[k] = 20 * Math.log10(Math.max(1e-10, Math.hypot(re[k], im[k]) / n));
  }
  return { freq, mag };
}

function drawSpectrum(canvasId, data, fs, title, cutoff) {
  const { ctx, w, h } = clearCanvas($(canvasId));
  const a = drawAxes(ctx, w, h, "Frequency (Hz)", "Magnitude (dB)");
  const s = magnitudeSpectrum(data, fs);

  const maxFreq = Math.min(8000, fs / 2);
  let minDb = Infinity, maxDb = -Infinity;
  for (let i = 0; i < s.freq.length; i++) {
    if (s.freq[i] > maxFreq) break;
    minDb = Math.min(minDb, s.mag[i]);
    maxDb = Math.max(maxDb, s.mag[i]);
  }
  minDb = Math.min(minDb, -100);
  maxDb = Math.max(maxDb, 0);
  if (maxDb - minDb < 10) maxDb = minDb + 10;

  ctx.strokeStyle = "#0b91a6";
  ctx.lineWidth = 1.25;
  ctx.beginPath();

  let started = false;
  for (let i = 0; i < s.freq.length; i++) {
    const f = s.freq[i];
    if (f > maxFreq) break;
    const x = a.left + (f / maxFreq) * (w - a.left - a.right);
    const y = a.top + (1 - (s.mag[i] - minDb) / (maxDb - minDb)) * (h - a.top - a.bottom);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (cutoff && cutoff < maxFreq) {
    const x = a.left + (cutoff / maxFreq) * (w - a.left - a.right);
    ctx.strokeStyle = "#e17b20";
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(x, a.top);
    ctx.lineTo(x, h - a.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#c46a16";
    ctx.font = "11px Arial";
    ctx.fillText(`Cutoff ${cutoff} Hz`, Math.min(x + 5, w - 100), a.top + 14);
  }

  ctx.fillStyle = "#17203a";
  ctx.font = "bold 12px Arial";
  ctx.fillText(title, a.left, 13);
}

function drawOverallWave(canvasId, series, fs) {
  const { ctx, w, h } = clearCanvas($(canvasId));
  const a = drawAxes(ctx, w, h, "Time (s)", "Amplitude");
  const maxSeconds = Math.min(15, series[0].data.length / fs);
  const end = Math.max(1, Math.floor(maxSeconds * fs));
  const points = 2500;
  const step = Math.max(1, Math.floor(end / points));

  const styles = [
    ["#101936", "Mixed"],
    ["#6540c7", "FFT Speech"],
    ["#0b91a6", "FIR Speech"],
    ["#d06f19", "IIR Speech"]
  ];

  for (let sidx = 0; sidx < Math.min(styles.length, series.length); sidx++) {
    const [stroke, label] = styles[sidx];
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.05;
    ctx.beginPath();
    const data = series[sidx].data;
    for (let p = 0, i = 0; i < end; p++, i += step) {
      const x = a.left + (p / (points - 1)) * (w - a.left - a.right);
      const y = a.top + (1 - (data[i] + 1) / 2) * (h - a.top - a.bottom);
      if (p === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = stroke;
    ctx.fillText(label, a.left + sidx * 92, h - 13);
  }
}

function drawOverallSpec(canvasId, series, fs) {
  const { ctx, w, h } = clearCanvas($(canvasId));
  const a = drawAxes(ctx, w, h, "Frequency (Hz)", "Magnitude (dB)");
  const maxFreq = Math.min(8000, fs / 2);
  const styles = [
    ["#101936", "Mixed"],
    ["#6540c7", "FFT Speech"],
    ["#0b91a6", "FIR Speech"],
    ["#d06f19", "IIR Speech"]
  ];

  for (let sidx = 0; sidx < Math.min(styles.length, series.length); sidx++) {
    const spec = magnitudeSpectrum(series[sidx].data, fs);
    ctx.strokeStyle = styles[sidx][0];
    ctx.lineWidth = 1;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < spec.freq.length; i++) {
      const f = spec.freq[i];
      if (f > maxFreq) break;
      const x = a.left + (f / maxFreq) * (w - a.left - a.right);
      const y = a.top + (1 - (spec.mag[i] + 100) / 100) * (h - a.top - a.bottom);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.font = "11px Arial";
  for (let i = 0; i < styles.length; i++) {
    ctx.fillStyle = styles[i][0];
    ctx.fillText(styles[i][1], a.left + i * 92, h - 13);
  }
}

/* -------------------- Main processing -------------------- */

async function processProject() {
  const mixFile = $("mixFile").files[0];
  if (!mixFile) {
    setStatus("Please select the mixed WAV input first.");
    return;
  }

  $("startBtn").disabled = true;
  $("results").classList.add("hidden");
  currentObjectUrls.forEach(URL.revokeObjectURL);
  currentObjectUrls = [];

  try {
    setStatus("Decoding mixed audio...", 2);
    const mixDecoded = await decodeFile(mixFile);
    const mix = mixDecoded.samples;
    const fs = mixDecoded.fs;

    const cutoff = Number($("cutoff").value);
    const firTaps = nextOdd(Number($("firTaps").value));
    const iirOrder = Math.max(2, Math.min(8, Number($("iirOrder").value)));

    if (cutoff <= 0 || cutoff >= fs / 2) {
      throw new Error(`Cutoff must be between 1 and ${Math.floor(fs / 2 - 1)} Hz for this audio.`);
    }

    setStatus(`Loaded ${mix.length.toLocaleString()} samples at ${fs} Hz. Running FFT separation...`, 5);

    const fftOut = await fftSeparation(mix, fs, cutoff, setProgress);
    setStatus("Running FIR low-pass / high-pass filters...", 63);

    const firLow = designFIRLowpass(firTaps, cutoff, fs);
    const firHigh = designFIRHighpass(firTaps, cutoff, fs);
    const firSpeech = normalize(await convolveOffline(mix, firLow, fs));
    const firMusic = normalize(await convolveOffline(mix, firHigh, fs));

    setStatus("Running IIR Butterworth filters...", 77);
    await new Promise(r => setTimeout(r, 20));
    const iirSpeech = normalize(iirFilter(mix, fs, cutoff, iirOrder, "lowpass"));
    const iirMusic = normalize(iirFilter(mix, fs, cutoff, iirOrder, "highpass"));

    setStatus("Loading optional reference files...", 87);
    let speechRef = null, musicRef = null;

    if ($("speechFile").files[0]) {
      const r = await decodeFile($("speechFile").files[0]);
      if (r.fs !== fs) throw new Error("Speech reference sampling rate does not match the mixed input.");
      speechRef = normalize(r.samples.slice(0, mix.length));
    }

    if ($("musicFile").files[0]) {
      const r = await decodeFile($("musicFile").files[0]);
      if (r.fs !== fs) throw new Error("Music reference sampling rate does not match the mixed input.");
      musicRef = normalize(r.samples.slice(0, mix.length));
    }

    setStatus("Creating audio outputs and graphs...", 91);

    attachAudio("fftSpeechAudio", "fftSpeechDownload", fftOut.speech, fs, "FFT_Speech.wav");
    attachAudio("fftMusicAudio", "fftMusicDownload", fftOut.music, fs, "FFT_Music.wav");
    attachAudio("firSpeechAudio", "firSpeechDownload", firSpeech, fs, "FIR_Speech.wav");
    attachAudio("firMusicAudio", "firMusicDownload", firMusic, fs, "FIR_Music.wav");
    attachAudio("iirSpeechAudio", "iirSpeechDownload", iirSpeech, fs, "IIR_Speech.wav");
    attachAudio("iirMusicAudio", "iirMusicDownload", iirMusic, fs, "IIR_Music.wav");

    drawWave("waveMix", mix, fs, "Mixed Input");
    drawWave("waveFftSpeech", fftOut.speech, fs, "FFT Speech");
    drawWave("waveFftMusic", fftOut.music, fs, "FFT Music");
    drawWave("waveFirSpeech", firSpeech, fs, "FIR Speech");
    drawWave("waveFirMusic", firMusic, fs, "FIR Music");
    drawWave("waveIirSpeech", iirSpeech, fs, "IIR Speech");
    drawWave("waveIirMusic", iirMusic, fs, "IIR Music");

    drawSpectrum("specMix", mix, fs, "Mixed Input Spectrum", cutoff);
    drawSpectrum("specFftSpeech", fftOut.speech, fs, "FFT Speech Spectrum", cutoff);
    drawSpectrum("specFftMusic", fftOut.music, fs, "FFT Music Spectrum", cutoff);
    drawSpectrum("specFirSpeech", firSpeech, fs, "FIR Speech Spectrum", cutoff);
    drawSpectrum("specFirMusic", firMusic, fs, "FIR Music Spectrum", cutoff);
    drawSpectrum("specIirSpeech", iirSpeech, fs, "IIR Speech Spectrum", cutoff);
    drawSpectrum("specIirMusic", iirMusic, fs, "IIR Music Spectrum", cutoff);

    drawOverallWave("overallWave", [
      {data: mix}, {data: fftOut.speech}, {data: firSpeech}, {data: iirSpeech}
    ], fs);

    drawOverallSpec("overallSpec", [
      {data: mix}, {data: fftOut.speech}, {data: firSpeech}, {data: iirSpeech}
    ], fs);

    const tbody = $("metricsTable").querySelector("tbody");
    tbody.innerHTML =
      metricsRow("FFT", fftOut.speech, fftOut.music, speechRef, musicRef) +
      metricsRow("FIR", firSpeech, firMusic, speechRef, musicRef) +
      metricsRow("IIR", iirSpeech, iirMusic, speechRef, musicRef);

    $("results").classList.remove("hidden");
    setStatus(`Completed successfully • ${mix.length.toLocaleString()} samples • ${ (mix.length / fs).toFixed(2) } seconds • Cutoff ${cutoff} Hz`, 100);
    window.scrollTo({ top: $("results").offsetTop - 20, behavior: "smooth" });

    lastResult = { fs, mix, fftOut, firSpeech, firMusic, iirSpeech, iirMusic };
  } catch (err) {
    console.error(err);
    setStatus("Error: " + (err.message || err));
  } finally {
    $("startBtn").disabled = false;
  }
}

function setProgress(value) {
  setStatus(`FFT separation in progress... ${value.toFixed(0)}%`, value);
}

$("startBtn").addEventListener("click", processProject);
