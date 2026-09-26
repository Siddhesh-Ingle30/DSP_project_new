/* ============================================================
   DSP AUDIO SEPARATION STUDIO
   Client-Side JavaScript Version
   FFT + FIR + IIR
   GitHub Pages Compatible
   ============================================================ */

const $ = (id) => document.getElementById(id);

let currentObjectUrls = [];
let lastResult = null;


/* ============================================================
   BASIC UI FUNCTIONS
   ============================================================ */

function setStatus(message, progress = null) {
    $("status").textContent = message;

    if (progress !== null) {
        $("progressWrap").classList.remove("hidden");
        $("progressBar").style.width =
            `${Math.max(0, Math.min(100, progress))}%`;
    }
}


function setFileName(inputId, labelId) {

    $(inputId).addEventListener("change", () => {

        const file = $(inputId).files[0];

        $(labelId).textContent =
            file ? file.name : "Choose a WAV file";
    });
}


setFileName("mixFile", "mixName");
setFileName("speechFile", "speechName");
setFileName("musicFile", "musicName");


function nextOdd(n) {

    n = Math.max(5, Math.floor(n));

    return n % 2 === 0 ? n + 1 : n;
}


/* ============================================================
   AUDIO NORMALIZATION
   ============================================================ */

function normalize(x, peak = 0.95) {

    let max = 0;

    for (let i = 0; i < x.length; i++) {
        max = Math.max(max, Math.abs(x[i]));
    }

    if (max === 0) {
        return x.slice();
    }

    const scale = peak / max;

    const out = new Float32Array(x.length);

    for (let i = 0; i < x.length; i++) {
        out[i] = x[i] * scale;
    }

    return out;
}


/* ============================================================
   AUDIO DECODING
   ============================================================ */

function downmix(buffer) {

    const channels = buffer.numberOfChannels;
    const n = buffer.length;

    const out = new Float32Array(n);

    for (let c = 0; c < channels; c++) {

        const channel = buffer.getChannelData(c);

        for (let i = 0; i < n; i++) {
            out[i] += channel[i] / channels;
        }
    }

    return out;
}


async function decodeFile(file) {

    const AudioContextClass =
        window.AudioContext ||
        window.webkitAudioContext;

    const ctx = new AudioContextClass();

    const data = await file.arrayBuffer();

    const decoded =
        await ctx.decodeAudioData(data.slice(0));

    const samples = downmix(decoded);

    const fs = decoded.sampleRate;

    await ctx.close();

    return {
        samples: normalize(samples),
        fs: fs
    };
}


/* ============================================================
   WAV ENCODER
   ============================================================ */

function encodeWav(samples, sampleRate) {

    const buffer =
        new ArrayBuffer(44 + samples.length * 2);

    const view = new DataView(buffer);


    function writeString(offset, text) {

        for (let i = 0; i < text.length; i++) {

            view.setUint8(
                offset + i,
                text.charCodeAt(i)
            );
        }
    }


    writeString(0, "RIFF");

    view.setUint32(
        4,
        36 + samples.length * 2,
        true
    );

    writeString(8, "WAVE");

    writeString(12, "fmt ");

    view.setUint32(16, 16, true);

    view.setUint16(20, 1, true);

    view.setUint16(22, 1, true);

    view.setUint32(
        24,
        sampleRate,
        true
    );

    view.setUint32(
        28,
        sampleRate * 2,
        true
    );

    view.setUint16(32, 2, true);

    view.setUint16(34, 16, true);

    writeString(36, "data");

    view.setUint32(
        40,
        samples.length * 2,
        true
    );


    let offset = 44;


    for (let i = 0; i < samples.length; i++) {

        const s =
            Math.max(
                -1,
                Math.min(1, samples[i])
            );

        view.setInt16(
            offset,
            s < 0
                ? s * 32768
                : s * 32767,
            true
        );

        offset += 2;
    }


    return new Blob(
        [buffer],
        { type: "audio/wav" }
    );
}


/* ============================================================
   AUDIO PLAYER + DOWNLOAD
   ============================================================ */

function attachAudio(
    audioId,
    downloadId,
    samples,
    fs,
    filename
) {

    const blob =
        encodeWav(samples, fs);

    const url =
        URL.createObjectURL(blob);

    currentObjectUrls.push(url);

    $(audioId).src = url;

    $(downloadId).href = url;

    $(downloadId).download = filename;
}


/* ============================================================
   FIR FILTER DESIGN
   ============================================================ */

function designFIRLowpass(
    taps,
    cutoff,
    fs
) {

    taps = nextOdd(taps);

    const h =
        new Float32Array(taps);

    const M =
        (taps - 1) / 2;

    const fc =
        cutoff / fs;


    for (let n = 0; n < taps; n++) {

        const k = n - M;

        let sinc;


        if (k === 0) {

            sinc = 2 * fc;

        } else {

            sinc =
                Math.sin(
                    2 * Math.PI * fc * k
                )
                /
                (Math.PI * k);
        }


        /* Hamming window */

        const window =
            0.54 -
            0.46 *
            Math.cos(
                (2 * Math.PI * n)
                /
                (taps - 1)
            );


        h[n] =
            sinc * window;
    }


    /* Normalize DC gain */

    let sum = 0;

    for (let i = 0; i < taps; i++) {
        sum += h[i];
    }


    for (let i = 0; i < taps; i++) {
        h[i] /= sum;
    }


    return h;
}


/* ============================================================
   FIR HIGH-PASS
   ============================================================ */

function designFIRHighpass(
    taps,
    cutoff,
    fs
) {

    const low =
        designFIRLowpass(
            taps,
            cutoff,
            fs
        );

    const high =
        new Float32Array(taps);

    const center =
        (taps - 1) / 2;


    for (let i = 0; i < taps; i++) {

        high[i] =
            -low[i];
    }


    high[center] += 1;


    return high;
}


/* ============================================================
   FIR CONVOLUTION
   Uses browser OfflineAudioContext
   ============================================================ */

async function convolveOffline(
    input,
    kernel,
    fs
) {

    const outputLength =
        input.length +
        kernel.length -
        1;


    const ctx =
        new OfflineAudioContext(
            1,
            outputLength,
            fs
        );


    const source =
        ctx.createBufferSource();


    const inputBuffer =
        ctx.createBuffer(
            1,
            input.length,
            fs
        );


    inputBuffer.copyToChannel(
        input,
        0
    );


    const impulse =
        ctx.createBuffer(
            1,
            kernel.length,
            fs
        );


    impulse.copyToChannel(
        kernel,
        0
    );


    const convolver =
        ctx.createConvolver();


    convolver.normalize = false;

    convolver.buffer = impulse;


    source.buffer =
        inputBuffer;


    source.connect(
        convolver
    );


    convolver.connect(
        ctx.destination
    );


    source.start();


    const rendered =
        await ctx.startRendering();


    const data =
        rendered.getChannelData(0);


    return data.slice(
        0,
        input.length
    );
}


/* ============================================================
   IIR BUTTERWORTH
   ============================================================ */

function butterworthQValues(order) {

    const q = [];

    const sections =
        order / 2;


    for (let k = 0; k < sections; k++) {

        const theta =
            Math.PI *
            (2 * k + 1)
            /
            (2 * order);


        q.push(
            1 /
            (
                2 *
                Math.cos(theta)
            )
        );
    }


    return q;
}


/* ============================================================
   IIR BIQUAD
   ============================================================ */

function processBiquad(
    input,
    fs,
    cutoff,
    type,
    Q
) {

    const w0 =
        2 *
        Math.PI *
        cutoff /
        fs;


    const cos =
        Math.cos(w0);

    const sin =
        Math.sin(w0);


    const alpha =
        sin /
        (2 * Q);


    let b0;
    let b1;
    let b2;

    let a0;
    let a1;
    let a2;


    if (type === "lowpass") {

        b0 =
            (1 - cos) / 2;

        b1 =
            1 - cos;

        b2 =
            (1 - cos) / 2;

    } else {

        b0 =
            (1 + cos) / 2;

        b1 =
            -(1 + cos);

        b2 =
            (1 + cos) / 2;
    }


    a0 =
        1 + alpha;

    a1 =
        -2 * cos;

    a2 =
        1 - alpha;


    b0 /= a0;
    b1 /= a0;
    b2 /= a0;

    a1 /= a0;
    a2 /= a0;


    const out =
        new Float32Array(
            input.length
        );


    let x1 = 0;
    let x2 = 0;

    let y1 = 0;
    let y2 = 0;


    for (let i = 0; i < input.length; i++) {

        const x0 =
            input[i];


        const y0 =
            b0 * x0 +
            b1 * x1 +
            b2 * x2 -
            a1 * y1 -
            a2 * y2;


        out[i] = y0;


        x2 = x1;
        x1 = x0;

        y2 = y1;
        y1 = y0;
    }


    return out;
}


/* ============================================================
   IIR FILTER
   ============================================================ */

function iirFilter(
    input,
    fs,
    cutoff,
    order,
    type
) {

    order =
        Math.max(
            2,
            Math.min(
                8,
                Math.floor(order)
            )
        );


    if (order % 2 !== 0) {
        order++;
    }


    const qValues =
        butterworthQValues(order);


    let out =
        input.slice();


    for (const q of qValues) {

        out =
            processBiquad(
                out,
                fs,
                cutoff,
                type,
                q
            );
    }


    return out;
}


/* ============================================================
   FFT
   ============================================================ */

function reverseBits(
    x,
    bits
) {

    let y = 0;


    for (let i = 0; i < bits; i++) {

        y =
            (y << 1) |
            (x & 1);

        x >>>= 1;
    }


    return y;
}


/* ============================================================
   RADIX-2 FFT
   ============================================================ */

function fft(
    re,
    im,
    inverse = false
) {

    const n =
        re.length;


    const bits =
        Math.round(
            Math.log2(n)
        );


    /* Bit reversal */

    for (let i = 0; i < n; i++) {

        const j =
            reverseBits(
                i,
                bits
            );


        if (j > i) {

            let temp =
                re[i];

            re[i] =
                re[j];

            re[j] =
                temp;


            temp =
                im[i];

            im[i] =
                im[j];

            im[j] =
                temp;
        }
    }


    /* FFT stages */

    for (
        let size = 2;
        size <= n;
        size <<= 1
    ) {

        const half =
            size >> 1;


        const sign =
            inverse
                ? 1
                : -1;


        const angle =
            sign *
            2 *
            Math.PI /
            size;


        const wpr =
            Math.cos(angle);

        const wpi =
            Math.sin(angle);


        for (
            let start = 0;
            start < n;
            start += size
        ) {

            let wr = 1;
            let wi = 0;


            for (
                let j = 0;
                j < half;
                j++
            ) {

                const even =
                    start + j;

                const odd =
                    even + half;


                const tr =
                    wr * re[odd] -
                    wi * im[odd];


                const ti =
                    wr * im[odd] +
                    wi * re[odd];


                const er =
                    re[even];

                const ei =
                    im[even];


                re[even] =
                    er + tr;

                im[even] =
                    ei + ti;


                re[odd] =
                    er - tr;

                im[odd] =
                    ei - ti;


                const nextWr =
                    wr * wpr -
                    wi * wpi;


                wi =
                    wr * wpi +
                    wi * wpr;


                wr =
                    nextWr;
            }
        }
    }


    /* Inverse FFT scaling */

    if (inverse) {

        for (let i = 0; i < n; i++) {

            re[i] /= n;
            im[i] /= n;
        }
    }
}


/* ============================================================
   FFT-BASED SPEECH / MUSIC SEPARATION
   ============================================================ */

async function fftSeparation(
    input,
    fs,
    cutoff,
    report
) {

    /*
       Frame-based FFT processing.

       This avoids creating one enormous FFT
       for the complete 4+ minute recording.
    */


    const frameSize =
        16384;


    const hop =
        frameSize / 2;


    const nFrames =
        Math.max(
            1,
            Math.ceil(
                (input.length - frameSize)
                /
                hop
            ) + 1
        );


    const speech =
        new Float32Array(
            input.length
        );


    const music =
        new Float32Array(
            input.length
        );


    const weight =
        new Float32Array(
            input.length
        );


    /* Hann window */

    const window =
        new Float32Array(
            frameSize
        );


    for (
        let n = 0;
        n < frameSize;
        n++
    ) {

        window[n] =
            0.5 -
            0.5 *
            Math.cos(
                2 *
                Math.PI *
                n /
                (frameSize - 1)
            );
    }


    for (
        let frame = 0;
        frame < nFrames;
        frame++
    ) {

        const start =
            frame * hop;


        const re =
            new Float64Array(
                frameSize
            );


        const im =
            new Float64Array(
                frameSize
            );


        /* Windowed input */

        for (
            let n = 0;
            n < frameSize;
            n++
        ) {

            const index =
                start + n;


            re[n] =
                index < input.length
                    ? input[index] *
                      window[n]
                    : 0;
        }


        /* Forward FFT */

        fft(
            re,
            im,
            false
        );


        /*
           Create separate spectra.
        */

        const speechRe =
            new Float64Array(
                frameSize
            );

        const speechIm =
            new Float64Array(
                frameSize
            );


        const musicRe =
            new Float64Array(
                frameSize
            );

        const musicIm =
            new Float64Array(
                frameSize
            );


        for (
            let k = 0;
            k < frameSize;
            k++
        ) {

            let frequency;


            if (
                k <=
                frameSize / 2
            ) {

                frequency =
                    k * fs /
                    frameSize;

            } else {

                frequency =
                    (k - frameSize) *
                    fs /
                    frameSize;
            }


            /*
               Low frequencies → speech
               High frequencies → music
            */

            if (
                Math.abs(frequency)
                <= cutoff
            ) {

                speechRe[k] =
                    re[k];

                speechIm[k] =
                    im[k];

            } else {

                musicRe[k] =
                    re[k];

                musicIm[k] =
                    im[k];
            }
        }


        /* Inverse FFT */

        fft(
            speechRe,
            speechIm,
            true
        );


        fft(
            musicRe,
            musicIm,
            true
        );


        /* Overlap-add */

        for (
            let n = 0;
            n < frameSize;
            n++
        ) {

            const index =
                start + n;


            if (
                index <
                input.length
            ) {

                speech[index] +=
                    speechRe[n] *
                    window[n];


                music[index] +=
                    musicRe[n] *
                    window[n];


                weight[index] +=
                    window[n] *
                    window[n];
            }
        }


        if (
            report &&
            (
                frame % 4 === 0 ||
                frame === nFrames - 1
            )
        ) {

            report(
                25 +
                35 *
                (
                    (frame + 1)
                    /
                    nFrames
                )
            );


            await new Promise(
                resolve =>
                    setTimeout(
                        resolve,
                        0
                    )
            );
        }
    }


    /* Normalize overlap */

    for (
        let i = 0;
        i < input.length;
        i++
    ) {

        if (
            weight[i] >
            1e-8
        ) {

            speech[i] /=
                weight[i];

            music[i] /=
                weight[i];
        }
    }


    return {

        speech:
            normalize(speech),

        music:
            normalize(music)
    };
}


/* ============================================================
   METRIC FUNCTIONS
   ============================================================ */


/*
   Find the best time alignment between
   reference and estimated signals.

   This is important because FIR/IIR filters
   can introduce phase/time delay.
*/

function findBestLag(
    reference,
    estimated,
    maxLagSamples = 500
) {

    const downsample =
        8;


    const maxLag =
        Math.floor(
            maxLagSamples /
            downsample
        );


    const usable =
        Math.min(
            reference.length,
            estimated.length
        );


    const length =
        Math.min(
            usable,
            120000
        );


    let bestLag = 0;

    let bestCorrelation =
        -Infinity;


    for (
        let lag = -maxLag;
        lag <= maxLag;
        lag++
    ) {

        let sumXY = 0;
        let sumX2 = 0;
        let sumY2 = 0;


        for (
            let i = 0;
            i < length;
            i += downsample
        ) {

            const j =
                i +
                lag *
                downsample;


            if (
                j < 0 ||
                j >= length
            ) {

                continue;
            }


            const x =
                reference[i];


            const y =
                estimated[j];


            sumXY +=
                x * y;


            sumX2 +=
                x * x;


            sumY2 +=
                y * y;
        }


        const denominator =
            Math.sqrt(
                sumX2 *
                sumY2
            );


        if (
            denominator >
            1e-12
        ) {

            const corr =
                sumXY /
                denominator;


            if (
                corr >
                bestCorrelation
            ) {

                bestCorrelation =
                    corr;

                bestLag =
                    lag *
                    downsample;
            }
        }
    }


    return bestLag;
}


/*
   Create aligned portions.
*/

function alignedSignals(
    reference,
    estimated
) {

    const n =
        Math.min(
            reference.length,
            estimated.length
        );


    const lag =
        findBestLag(
            reference,
            estimated
        );


    let refStart = 0;
    let estStart = 0;


    if (lag > 0) {

        estStart =
            lag;

    } else {

        refStart =
            -lag;
    }


    const length =
        Math.min(
            n - refStart,
            n - estStart
        );


    return {

        reference:
            reference.slice(
                refStart,
                refStart + length
            ),

        estimated:
            estimated.slice(
                estStart,
                estStart + length
            ),

        lag:
            lag
    };
}


/* ============================================================
   CORRELATION
   ============================================================ */

function correlation(
    reference,
    estimated
) {

    const aligned =
        alignedSignals(
            reference,
            estimated
        );


    const x =
        aligned.reference;

    const y =
        aligned.estimated;


    const n =
        Math.min(
            x.length,
            y.length
        );


    if (n === 0) {
        return 0;
    }


    let meanX = 0;
    let meanY = 0;


    for (let i = 0; i < n; i++) {

        meanX += x[i];
        meanY += y[i];
    }


    meanX /= n;
    meanY /= n;


    let numerator = 0;

    let denominatorX = 0;

    let denominatorY = 0;


    for (let i = 0; i < n; i++) {

        const dx =
            x[i] - meanX;


        const dy =
            y[i] - meanY;


        numerator +=
            dx * dy;


        denominatorX +=
            dx * dx;


        denominatorY +=
            dy * dy;
    }


    const denominator =
        Math.sqrt(
            denominatorX *
            denominatorY
        );


    return denominator
        ? numerator / denominator
        : 0;
}


/* ============================================================
   SNR
   ============================================================ */

function snr(
    reference,
    estimated
) {

    const aligned =
        alignedSignals(
            reference,
            estimated
        );


    const x =
        aligned.reference;

    const y =
        aligned.estimated;


    const n =
        Math.min(
            x.length,
            y.length
        );


    if (n === 0) {
        return 0;
    }


    let dot = 0;

    let estimatedEnergy = 0;


    for (let i = 0; i < n; i++) {

        dot +=
            x[i] *
            y[i];


        estimatedEnergy +=
            y[i] *
            y[i];
    }


    const scale =
        dot /
        (
            estimatedEnergy +
            1e-12
        );


    let signalPower = 0;

    let errorPower = 0;


    for (let i = 0; i < n; i++) {

        const estimate =
            scale *
            y[i];


        const error =
            x[i] -
            estimate;


        signalPower +=
            x[i] *
            x[i];


        errorPower +=
            error *
            error;
    }


    signalPower /= n;

    errorPower /= n;


    if (
        errorPower <=
        1e-15
    ) {

        return Infinity;
    }


    return (
        10 *
        Math.log10(
            signalPower /
            errorPower
        )
    );
}


/* ============================================================
   RMSE
   ============================================================ */

function rmse(
    reference,
    estimated
) {

    const aligned =
        alignedSignals(
            reference,
            estimated
        );


    const x =
        aligned.reference;

    const y =
        aligned.estimated;


    const n =
        Math.min(
            x.length,
            y.length
        );


    if (n === 0) {
        return 0;
    }


    let dot = 0;

    let energy = 0;


    for (let i = 0; i < n; i++) {

        dot +=
            x[i] *
            y[i];


        energy +=
            y[i] *
            y[i];
    }


    const scale =
        dot /
        (
            energy +
            1e-12
        );


    let error = 0;


    for (let i = 0; i < n; i++) {

        const difference =
            x[i] -
            scale *
            y[i];


        error +=
            difference *
            difference;
    }


    return Math.sqrt(
        error / n
    );
}


/* ============================================================
   METRIC TABLE ROW
   ============================================================ */

function fmt(value) {

    if (
        !Number.isFinite(value)
    ) {

        return "∞";
    }


    return value.toFixed(4);
}


function metricsRow(
    method,
    speech,
    music,
    speechRef,
    musicRef
) {

    if (
        !speechRef ||
        !musicRef
    ) {

        return `
        <tr>
            <td>${method}</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>`;
    }


    return `
    <tr>
        <td>${method}</td>

        <td>
            ${fmt(
                correlation(
                    speechRef,
                    speech
                )
            )}
        </td>

        <td>
            ${fmt(
                correlation(
                    musicRef,
                    music
                )
            )}
        </td>

        <td>
            ${fmt(
                snr(
                    speechRef,
                    speech
                )
            )}
        </td>

        <td>
            ${fmt(
                snr(
                    musicRef,
                    music
                )
            )}
        </td>

        <td>
            ${fmt(
                rmse(
                    speechRef,
                    speech
                )
            )}
        </td>

        <td>
            ${fmt(
                rmse(
                    musicRef,
                    music
                )
            )}
        </td>
    </tr>`;
}


/* ============================================================
   CANVAS SETUP
   ============================================================ */

function clearCanvas(
    canvas
) {

    const dpr =
        window.devicePixelRatio ||
        1;


    const rect =
        canvas.getBoundingClientRect();


    const width =
        Math.max(
            500,
            Math.floor(
                rect.width
            )
        );


    const height =
        Math.max(
            230,
            Math.floor(
                rect.height
            )
        );


    canvas.width =
        width *
        dpr;


    canvas.height =
        height *
        dpr;


    const ctx =
        canvas.getContext(
            "2d"
        );


    ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );


    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    return {

        ctx: ctx,

        w: width,

        h: height
    };
}


/* ============================================================
   GRAPH AXES
   ============================================================ */

function drawAxes(
    ctx,
    w,
    h,
    xLabel,
    yLabel
) {

    const left = 60;
    const right = 20;
    const top = 30;
    const bottom = 42;


    ctx.strokeStyle =
        "#d9dce7";


    ctx.lineWidth = 1;


    ctx.beginPath();


    ctx.moveTo(
        left,
        top
    );


    ctx.lineTo(
        left,
        h - bottom
    );


    ctx.lineTo(
        w - right,
        h - bottom
    );


    ctx.stroke();


    /* X-axis label */

    ctx.fillStyle =
        "#68718b";


    ctx.font =
        "12px Arial";


    ctx.textAlign =
        "right";


    ctx.fillText(
        xLabel,
        w - right,
        h - 12
    );


    /* Y-axis label */

    ctx.save();


    ctx.translate(
        15,
        h / 2
    );


    ctx.rotate(
        -Math.PI / 2
    );


    ctx.textAlign =
        "center";


    ctx.fillText(
        yLabel,
        0,
        0
    );


    ctx.restore();


    return {

        left,
        right,
        top,
        bottom
    };
}


/* ============================================================
   TIME-DOMAIN WAVEFORM
   ============================================================ */

function drawWave(
    canvasId,
    data,
    fs,
    title
) {

    const {

        ctx,
        w,
        h

    } =
        clearCanvas(
            $(canvasId)
        );


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Time (s)",
            "Amplitude"
        );


    const maxSeconds =
        Math.min(
            30,
            data.length / fs
        );


    const end =
        Math.max(
            1,
            Math.floor(
                maxSeconds *
                fs
            )
        );


    const points =
        Math.min(
            5000,
            end
        );


    const step =
        Math.max(
            1,
            Math.floor(
                end / points
            )
        );


    let min =
        Infinity;


    let max =
        -Infinity;


    for (
        let i = 0;
        i < end;
        i += step
    ) {

        const value =
            data[i];


        min =
            Math.min(
                min,
                value
            );


        max =
            Math.max(
                max,
                value
            );
    }


    if (
        Math.abs(
            max - min
        ) < 1e-8
    ) {

        min = -1;
        max = 1;
    }


    ctx.strokeStyle =
        "#6540c7";


    ctx.lineWidth =
        1.1;


    ctx.beginPath();


    let point = 0;


    for (
        let i = 0;
        i < end;
        i += step
    ) {

        const x =
            axis.left +
            (
                point /
                Math.max(
                    1,
                    points - 1
                )
            ) *
            (
                w -
                axis.left -
                axis.right
            );


        const y =
            axis.top +
            (
                1 -
                (
                    data[i] -
                    min
                ) /
                (
                    max -
                    min
                )
            ) *
            (
                h -
                axis.top -
                axis.bottom
            );


        if (point === 0) {

            ctx.moveTo(
                x,
                y
            );

        } else {

            ctx.lineTo(
                x,
                y
            );
        }


        point++;
    }


    ctx.stroke();


    /* Graph title */

    ctx.fillStyle =
        "#17203a";


    ctx.font =
        "bold 12px Arial";


    ctx.textAlign =
        "left";


    ctx.fillText(
        title,
        axis.left,
        17
    );
}


/* ============================================================
   FFT MAGNITUDE SPECTRUM
   ============================================================ */

function magnitudeSpectrum(
    data,
    fs
) {

    const maxSeconds =
        Math.min(
            60,
            data.length / fs
        );


    const count =
        Math.min(
            data.length,
            Math.floor(
                maxSeconds *
                fs
            ),
            262144
        );


    let n = 1;


    while (
        (n << 1) <= count
    ) {

        n <<= 1;
    }


    const re =
        new Float64Array(n);


    const im =
        new Float64Array(n);


    /* Hann window */

    for (
        let i = 0;
        i < n;
        i++
    ) {

        const window =
            0.5 -
            0.5 *
            Math.cos(
                2 *
                Math.PI *
                i /
                (n - 1)
            );


        re[i] =
            data[i] *
            window;
    }


    fft(
        re,
        im,
        false
    );


    const half =
        n / 2;


    const frequency =
        new Float32Array(
            half
        );


    const magnitude =
        new Float32Array(
            half
        );


    for (
        let k = 0;
        k < half;
        k++
    ) {

        frequency[k] =
            k *
            fs /
            n;


        magnitude[k] =
            20 *
            Math.log10(
                Math.max(
                    1e-10,
                    Math.hypot(
                        re[k],
                        im[k]
                    ) / n
                )
            );
    }


    return {

        frequency,
        magnitude
    };
}


/* ============================================================
   INDIVIDUAL SPECTRUM GRAPH
   ============================================================ */

function drawSpectrum(
    canvasId,
    data,
    fs,
    title,
    cutoff
) {

    const {

        ctx,
        w,
        h

    } =
        clearCanvas(
            $(canvasId)
        );


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Frequency (Hz)",
            "Magnitude (dB)"
        );


    const spectrum =
        magnitudeSpectrum(
            data,
            fs
        );


    const maxFrequency =
        Math.min(
            8000,
            fs / 2
        );


    let minDb =
        Infinity;


    let maxDb =
        -Infinity;


    for (
        let i = 0;
        i <
        spectrum.frequency.length;
        i++
    ) {

        if (
            spectrum.frequency[i]
            >
            maxFrequency
        ) {

            break;
        }


        minDb =
            Math.min(
                minDb,
                spectrum.magnitude[i]
            );


        maxDb =
            Math.max(
                maxDb,
                spectrum.magnitude[i]
            );
    }


    minDb =
        Math.min(
            minDb,
            -100
        );


    maxDb =
        Math.max(
            maxDb,
            0
        );


    if (
        maxDb -
        minDb
        <
        10
    ) {

        maxDb =
            minDb + 10;
    }


    /* Spectrum */

    ctx.strokeStyle =
        "#0b91a6";


    ctx.lineWidth =
        1.25;


    ctx.beginPath();


    let started =
        false;


    for (
        let i = 0;
        i <
        spectrum.frequency.length;
        i++
    ) {

        const frequency =
            spectrum.frequency[i];


        if (
            frequency >
            maxFrequency
        ) {

            break;
        }


        const x =
            axis.left +
            (
                frequency /
                maxFrequency
            ) *
            (
                w -
                axis.left -
                axis.right
            );


        const y =
            axis.top +
            (
                1 -
                (
                    spectrum.magnitude[i] -
                    minDb
                ) /
                (
                    maxDb -
                    minDb
                )
            ) *
            (
                h -
                axis.top -
                axis.bottom
            );


        if (!started) {

            ctx.moveTo(
                x,
                y
            );

            started = true;

        } else {

            ctx.lineTo(
                x,
                y
            );
        }
    }


    ctx.stroke();


    /* Cutoff */

    if (
        cutoff &&
        cutoff <
        maxFrequency
    ) {

        const cutoffX =
            axis.left +
            (
                cutoff /
                maxFrequency
            ) *
            (
                w -
                axis.left -
                axis.right
            );


        ctx.strokeStyle =
            "#e17b20";


        ctx.lineWidth =
            1.5;


        ctx.setLineDash(
            [6, 5]
        );


        ctx.beginPath();


        ctx.moveTo(
            cutoffX,
            axis.top
        );


        ctx.lineTo(
            cutoffX,
            h -
            axis.bottom
        );


        ctx.stroke();


        ctx.setLineDash([]);


        ctx.fillStyle =
            "#c46a16";


        ctx.font =
            "11px Arial";


        ctx.textAlign =
            "left";


        ctx.fillText(
            `Cutoff ${cutoff} Hz`,
            Math.min(
                cutoffX + 6,
                w - 100
            ),
            axis.top + 16
        );
    }


    /* Title */

    ctx.fillStyle =
        "#17203a";


    ctx.font =
        "bold 12px Arial";


    ctx.textAlign =
        "left";


    ctx.fillText(
        title,
        axis.left,
        17
    );
}


/* ============================================================
   OVERALL WAVEFORM COMPARISON
   ============================================================ */

function drawOverallWave(
    canvasId,
    series,
    fs
) {

    const {

        ctx,
        w,
        h

    } =
        clearCanvas(
            $(canvasId)
        );


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Time (s)",
            "Amplitude"
        );


    const maxSeconds =
        Math.min(
            30,
            series[0].data.length / fs
        );


    const end =
        Math.max(
            1,
            Math.floor(
                maxSeconds *
                fs
            )
        );


    const points =
        3000;


    const step =
        Math.max(
            1,
            Math.floor(
                end / points
            )
        );


    /*
       Each series gets its own color.
    */

    const styles = [

        ["#101936", "Mixed Input"],

        ["#6540c7", "FFT Speech"],

        ["#0b91a6", "FIR Speech"],

        ["#d06f19", "IIR Speech"]
    ];


    /*
       Use fixed amplitude range so
       the comparison is visually meaningful.
    */

    let min =
        -1;

    let max =
        1;


    /*
       Draw each waveform.
    */

    for (
        let s = 0;
        s <
        Math.min(
            styles.length,
            series.length
        );
        s++
    ) {

        const data =
            series[s].data;


        ctx.strokeStyle =
            styles[s][0];


        ctx.lineWidth =
            s === 0
                ? 1.25
                : 1.0;


        ctx.beginPath();


        let point = 0;


        for (
            let i = 0;
            i < end;
            i += step
        ) {

            const x =
                axis.left +
                (
                    point /
                    (
                        points - 1
                    )
                ) *
                (
                    w -
                    axis.left -
                    axis.right
                );


            const clipped =
                Math.max(
                    min,
                    Math.min(
                        max,
                        data[i]
                    )
                );


            const y =
                axis.top +
                (
                    1 -
                    (
                        clipped -
                        min
                    ) /
                    (
                        max -
                        min
                    )
                ) *
                (
                    h -
                    axis.top -
                    axis.bottom
                );


            if (
                point === 0
            ) {

                ctx.moveTo(
                    x,
                    y
                );

            } else {

                ctx.lineTo(
                    x,
                    y
                );
            }


            point++;
        }


        ctx.stroke();
    }


    /*
       LEGEND
       This is deliberately placed at the TOP,
       not on the X-axis.
    */

    const legendY =
        axis.top + 5;


    let legendX =
        axis.left;


    ctx.font =
        "bold 11px Arial";


    for (
        let i = 0;
        i <
        Math.min(
            styles.length,
            series.length
        );
        i++
    ) {

        ctx.fillStyle =
            styles[i][0];


        ctx.fillRect(
            legendX,
            legendY - 8,
            20,
            3
        );


        ctx.fillText(
            styles[i][1],
            legendX + 26,
            legendY - 4
        );


        legendX +=
            115;
    }
}


/* ============================================================
   OVERALL FFT SPECTRUM COMPARISON
   ============================================================ */

function drawOverallSpec(
    canvasId,
    series,
    fs
) {

    const {

        ctx,
        w,
        h

    } =
        clearCanvas(
            $(canvasId)
        );


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Frequency (Hz)",
            "Magnitude (dB)"
        );


    const maxFrequency =
        Math.min(
            8000,
            fs / 2
        );


    const styles = [

        ["#101936", "Mixed Input"],

        ["#6540c7", "FFT Speech"],

        ["#0b91a6", "FIR Speech"],

        ["#d06f19", "IIR Speech"]
    ];


    /*
       Find common dB scale.
    */

    const minDb =
        -100;


    const maxDb =
        0;


    /*
       Draw spectra.
    */

    for (
        let s = 0;
        s <
        Math.min(
            styles.length,
            series.length
        );
        s++
    ) {

        const spectrum =
            magnitudeSpectrum(
                series[s].data,
                fs
            );


        ctx.strokeStyle =
            styles[s][0];


        ctx.lineWidth =
            s === 0
                ? 1.25
                : 1.0;


        ctx.beginPath();


        let started =
            false;


        for (
            let i = 0;
            i <
            spectrum.frequency.length;
            i++
        ) {

            const frequency =
                spectrum.frequency[i];


            if (
                frequency >
                maxFrequency
            ) {

                break;
            }


            const x =
                axis.left +
                (
                    frequency /
                    maxFrequency
                ) *
                (
                    w -
                    axis.left -
                    axis.right
                );


            const db =
                Math.max(
                    minDb,
                    Math.min(
                        maxDb,
                        spectrum.magnitude[i]
                    )
                );


            const y =
                axis.top +
                (
                    1 -
                    (
                        db -
                        minDb
                    ) /
                    (
                        maxDb -
                        minDb
                    )
                ) *
                (
                    h -
                    axis.top -
                    axis.bottom
                );


            if (
                !started
            ) {

                ctx.moveTo(
                    x,
                    y
                );

                started = true;

            } else {

                ctx.lineTo(
                    x,
                    y
                );
            }
        }


        ctx.stroke();
    }


    /*
       LEGEND
       Top of graph.
    */

    let legendX =
        axis.left;


    const legendY =
        axis.top + 5;


    ctx.font =
        "bold 11px Arial";


    for (
        let i = 0;
        i <
        Math.min(
            styles.length,
            series.length
        );
        i++
    ) {

        ctx.fillStyle =
            styles[i][0];


        ctx.fillRect(
            legendX,
            legendY - 8,
            20,
            3
        );


        ctx.fillText(
            styles[i][1],
            legendX + 26,
            legendY - 4
        );


        legendX +=
            115;
    }
}


/* ============================================================
   MAIN PROJECT PROCESSING
   ============================================================ */

async function processProject() {

    const mixFile =
        $("mixFile").files[0];


    if (!mixFile) {

        setStatus(
            "Please select the mixed WAV input first."
        );

        return;
    }


    $("startBtn").disabled =
        true;


    $("results").classList.add(
        "hidden"
    );


    /* Remove previous audio URLs */

    currentObjectUrls.forEach(
        URL.revokeObjectURL
    );


    currentObjectUrls = [];


    try {

        /* ====================================================
           LOAD MIX
           ==================================================== */

        setStatus(
            "Decoding mixed audio...",
            2
        );


        const mixDecoded =
            await decodeFile(
                mixFile
            );


        const mix =
            mixDecoded.samples;


        const fs =
            mixDecoded.fs;


        /* ====================================================
           PARAMETERS
           ==================================================== */

        const cutoff =
            Number(
                $("cutoff").value
            );


        const firTaps =
            nextOdd(
                Number(
                    $("firTaps").value
                )
            );


        const iirOrder =
            Math.max(
                2,
                Math.min(
                    8,
                    Number(
                        $("iirOrder").value
                    )
                )
            );


        if (
            cutoff <= 0 ||
            cutoff >= fs / 2
        ) {

            throw new Error(
                `Cutoff must be between 1 and ${Math.floor(fs / 2 - 1)} Hz.`
            );
        }


        /* ====================================================
           FFT
           ==================================================== */

        setStatus(
            `Loaded ${mix.length.toLocaleString()} samples at ${fs} Hz. Running FFT separation...`,
            5
        );


        const fftOut =
            await fftSeparation(
                mix,
                fs,
                cutoff,
                setProgress
            );


        /* ====================================================
           FIR
           ==================================================== */

        setStatus(
            "Running FIR low-pass / high-pass filters...",
            63
        );


        const firLow =
            designFIRLowpass(
                firTaps,
                cutoff,
                fs
            );


        const firHigh =
            designFIRHighpass(
                firTaps,
                cutoff,
                fs
            );


        const firSpeech =
            normalize(
                await convolveOffline(
                    mix,
                    firLow,
                    fs
                )
            );


        const firMusic =
            normalize(
                await convolveOffline(
                    mix,
                    firHigh,
                    fs
                )
            );


        /* ====================================================
           IIR
           ==================================================== */

        setStatus(
            "Running IIR Butterworth filters...",
            77
        );


        await new Promise(
            resolve =>
                setTimeout(
                    resolve,
                    20
                )
        );


        const iirSpeech =
            normalize(
                iirFilter(
                    mix,
                    fs,
                    cutoff,
                    iirOrder,
                    "lowpass"
                )
            );


        const iirMusic =
            normalize(
                iirFilter(
                    mix,
                    fs,
                    cutoff,
                    iirOrder,
                    "highpass"
                )
            );


        /* ====================================================
           REFERENCE FILES
           ==================================================== */

        setStatus(
            "Loading optional reference files...",
            87
        );


        let speechRef =
            null;


        let musicRef =
            null;


        if (
            $("speechFile").files[0]
        ) {

            const reference =
                await decodeFile(
                    $("speechFile").files[0]
                );


            if (
                reference.fs !== fs
            ) {

                throw new Error(
                    "Speech reference sampling rate does not match the mixed input."
                );
            }


            speechRef =
                normalize(
                    reference.samples
                        .slice(
                            0,
                            mix.length
                        )
                );
        }


        if (
            $("musicFile").files[0]
        ) {

            const reference =
                await decodeFile(
                    $("musicFile").files[0]
                );


            if (
                reference.fs !== fs
            ) {

                throw new Error(
                    "Music reference sampling rate does not match the mixed input."
                );
            }


            musicRef =
                normalize(
                    reference.samples
                        .slice(
                            0,
                            mix.length
                        )
                );
        }


        /* ====================================================
           OUTPUTS
           ==================================================== */

        setStatus(
            "Creating audio outputs and graphs...",
            91
        );


        /* FFT */

        attachAudio(
            "fftSpeechAudio",
            "fftSpeechDownload",
            fftOut.speech,
            fs,
            "FFT_Speech.wav"
        );


        attachAudio(
            "fftMusicAudio",
            "fftMusicDownload",
            fftOut.music,
            fs,
            "FFT_Music.wav"
        );


        /* FIR */

        attachAudio(
            "firSpeechAudio",
            "firSpeechDownload",
            firSpeech,
            fs,
            "FIR_Speech.wav"
        );


        attachAudio(
            "firMusicAudio",
            "firMusicDownload",
            firMusic,
            fs,
            "FIR_Music.wav"
        );


        /* IIR */

        attachAudio(
            "iirSpeechAudio",
            "iirSpeechDownload",
            iirSpeech,
            fs,
            "IIR_Speech.wav"
        );


        attachAudio(
            "iirMusicAudio",
            "iirMusicDownload",
            iirMusic,
            fs,
            "IIR_Music.wav"
        );


        /* ====================================================
           INDIVIDUAL WAVEFORMS
           ==================================================== */

        drawWave(
            "waveMix",
            mix,
            fs,
            "Mixed Input"
        );


        drawWave(
            "waveFftSpeech",
            fftOut.speech,
            fs,
            "FFT Speech"
        );


        drawWave(
            "waveFftMusic",
            fftOut.music,
            fs,
            "FFT Music"
        );


        drawWave(
            "waveFirSpeech",
            firSpeech,
            fs,
            "FIR Speech"
        );


        drawWave(
            "waveFirMusic",
            firMusic,
            fs,
            "FIR Music"
        );


        drawWave(
            "waveIirSpeech",
            iirSpeech,
            fs,
            "IIR Speech"
        );


        drawWave(
            "waveIirMusic",
            iirMusic,
            fs,
            "IIR Music"
        );


        /* ====================================================
           INDIVIDUAL SPECTRA
           ==================================================== */

        drawSpectrum(
            "specMix",
            mix,
            fs,
            "Mixed Input Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFftSpeech",
            fftOut.speech,
            fs,
            "FFT Speech Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFftMusic",
            fftOut.music,
            fs,
            "FFT Music Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFirSpeech",
            firSpeech,
            fs,
            "FIR Speech Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFirMusic",
            firMusic,
            fs,
            "FIR Music Spectrum",
            cutoff
        );


        drawSpectrum(
            "specIirSpeech",
            iirSpeech,
            fs,
            "IIR Speech Spectrum",
            cutoff
        );


        drawSpectrum(
            "specIirMusic",
            iirMusic,
            fs,
            "IIR Music Spectrum",
            cutoff
        );


        /* ====================================================
           OVERALL WAVEFORM
           ==================================================== */

        drawOverallWave(
            "overallWave",
            [

                {
                    data: mix
                },

                {
                    data: fftOut.speech
                },

                {
                    data: firSpeech
                },

                {
                    data: iirSpeech
                }

            ],
            fs
        );


        /* ====================================================
           OVERALL SPECTRUM
           ==================================================== */

        drawOverallSpec(
            "overallSpec",
            [

                {
                    data: mix
                },

                {
                    data: fftOut.speech
                },

                {
                    data: firSpeech
                },

                {
                    data: iirSpeech
                }

            ],
            fs
        );


        /* ====================================================
           PERFORMANCE TABLE
           ==================================================== */

        const tbody =
            $("metricsTable")
                .querySelector(
                    "tbody"
                );


        tbody.innerHTML =

            metricsRow(
                "FFT",
                fftOut.speech,
                fftOut.music,
                speechRef,
                musicRef
            )

            +

            metricsRow(
                "FIR",
                firSpeech,
                firMusic,
                speechRef,
                musicRef
            )

            +

            metricsRow(
                "IIR",
                iirSpeech,
                iirMusic,
                speechRef,
                musicRef
            );


        /* ====================================================
           SHOW RESULTS
           ==================================================== */

        $("results")
            .classList
            .remove("hidden");


        setStatus(
            `Completed successfully • ${mix.length.toLocaleString()} samples • ${(mix.length / fs).toFixed(2)} seconds • Cutoff ${cutoff} Hz`,
            100
        );


        window.scrollTo({

            top:
                $("results")
                    .offsetTop - 20,

            behavior:
                "smooth"
        });


        lastResult = {

            fs,

            mix,

            fftOut,

            firSpeech,

            firMusic,

            iirSpeech,

            iirMusic
        };


    } catch (error) {

        console.error(error);


        setStatus(
            "Error: " +
            (
                error.message ||
                error
            )
        );


    } finally {

        $("startBtn").disabled =
            false;
    }
}


/* ============================================================
   PROGRESS CALLBACK
   ============================================================ */

function setProgress(
    value
) {

    setStatus(
        `FFT separation in progress... ${value.toFixed(0)}%`,
        value
    );
}


/* ============================================================
   START BUTTON
   ============================================================ */

$("startBtn")
    .addEventListener(
        "click",
        processProject
    );


/* ============================================================
   WINDOW RESIZE
   Redraw graphs after resizing.
   ============================================================ */

window.addEventListener(
    "resize",
    () => {

        if (!lastResult) {
            return;
        }


        const {
            fs,
            mix,
            fftOut,
            firSpeech,
            firMusic,
            iirSpeech,
            iirMusic
        } =
            lastResult;


        const cutoff =
            Number(
                $("cutoff").value
            );


        /* Waveforms */

        drawWave(
            "waveMix",
            mix,
            fs,
            "Mixed Input"
        );


        drawWave(
            "waveFftSpeech",
            fftOut.speech,
            fs,
            "FFT Speech"
        );


        drawWave(
            "waveFftMusic",
            fftOut.music,
            fs,
            "FFT Music"
        );


        drawWave(
            "waveFirSpeech",
            firSpeech,
            fs,
            "FIR Speech"
        );


        drawWave(
            "waveFirMusic",
            firMusic,
            fs,
            "FIR Music"
        );


        drawWave(
            "waveIirSpeech",
            iirSpeech,
            fs,
            "IIR Speech"
        );


        drawWave(
            "waveIirMusic",
            iirMusic,
            fs,
            "IIR Music"
        );


        /* Spectra */

        drawSpectrum(
            "specMix",
            mix,
            fs,
            "Mixed Input Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFftSpeech",
            fftOut.speech,
            fs,
            "FFT Speech Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFftMusic",
            fftOut.music,
            fs,
            "FFT Music Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFirSpeech",
            firSpeech,
            fs,
            "FIR Speech Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFirMusic",
            firMusic,
            fs,
            "FIR Music Spectrum",
            cutoff
        );


        drawSpectrum(
            "specIirSpeech",
            iirSpeech,
            fs,
            "IIR Speech Spectrum",
            cutoff
        );


        drawSpectrum(
            "specIirMusic",
            iirMusic,
            fs,
            "IIR Music Spectrum",
            cutoff
        );


        /* Overall */

        drawOverallWave(
            "overallWave",
            [

                {
                    data: mix
                },

                {
                    data: fftOut.speech
                },

                {
                    data: firSpeech
                },

                {
                    data: iirSpeech
                }

            ],
            fs
        );


        drawOverallSpec(
            "overallSpec",
            [

                {
                    data: mix
                },

                {
                    data: fftOut.speech
                },

                {
                    data: firSpeech
                },

                {
                    data: iirSpeech
                }

            ],
            fs
        );
    }
);
