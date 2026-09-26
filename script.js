/* ============================================================
   DSP AUDIO SEPARATION STUDIO
   Pure JavaScript / GitHub Pages
   FFT + FIR + IIR
   ============================================================ */

const $ = id => document.getElementById(id);

let objectUrls = [];
let lastResult = null;


/* ============================================================
   STATUS
   ============================================================ */

function setStatus(message, progress = null) {
    const status = $("status");

    if (status) {
        status.textContent = message;
    }

    if (progress !== null) {
        const wrap = $("progressWrap");
        const bar = $("progressBar");

        if (wrap) wrap.classList.remove("hidden");
        if (bar) {
            bar.style.width =
                `${Math.max(0, Math.min(100, progress))}%`;
        }
    }
}


/* ============================================================
   FILE NAMES
   ============================================================ */

function setupFileName(inputId, labelId) {

    const input = $(inputId);

    if (!input) return;

    input.addEventListener("change", () => {

        const file = input.files[0];

        if ($(labelId)) {
            $(labelId).textContent =
                file ? file.name : "Choose a WAV file";
        }
    });
}


setupFileName("mixFile", "mixName");
setupFileName("speechFile", "speechName");
setupFileName("musicFile", "musicName");


/* ============================================================
   UTILITY
   ============================================================ */

function nextOdd(n) {

    n = Math.max(5, Math.floor(n));

    return n % 2 === 0 ? n + 1 : n;
}


function normalize(x, peak = 0.95) {

    let maximum = 0;

    for (let i = 0; i < x.length; i++) {
        maximum = Math.max(
            maximum,
            Math.abs(x[i])
        );
    }

    if (maximum < 1e-12) {
        return x.slice();
    }

    const scale = peak / maximum;

    const out =
        new Float32Array(x.length);

    for (let i = 0; i < x.length; i++) {
        out[i] = x[i] * scale;
    }

    return out;
}


/* ============================================================
   AUDIO LOADING
   ============================================================ */

function downmix(buffer) {

    const channels =
        buffer.numberOfChannels;

    const n =
        buffer.length;

    const output =
        new Float32Array(n);

    for (let c = 0; c < channels; c++) {

        const channel =
            buffer.getChannelData(c);

        for (let i = 0; i < n; i++) {
            output[i] +=
                channel[i] / channels;
        }
    }

    return output;
}


async function decodeFile(file) {

    const AudioContextClass =
        window.AudioContext ||
        window.webkitAudioContext;

    const context =
        new AudioContextClass();

    const buffer =
        await file.arrayBuffer();

    const decoded =
        await context.decodeAudioData(
            buffer.slice(0)
        );

    const samples =
        downmix(decoded);

    const fs =
        decoded.sampleRate;

    await context.close();

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
        new ArrayBuffer(
            44 +
            samples.length * 2
        );

    const view =
        new DataView(buffer);


    function writeString(
        offset,
        string
    ) {

        for (
            let i = 0;
            i < string.length;
            i++
        ) {

            view.setUint8(
                offset + i,
                string.charCodeAt(i)
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

    view.setUint32(
        16,
        16,
        true
    );

    view.setUint16(
        20,
        1,
        true
    );

    view.setUint16(
        22,
        1,
        true
    );

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

    view.setUint16(
        32,
        2,
        true
    );

    view.setUint16(
        34,
        16,
        true
    );

    writeString(36, "data");

    view.setUint32(
        40,
        samples.length * 2,
        true
    );


    let offset = 44;


    for (
        let i = 0;
        i < samples.length;
        i++
    ) {

        const sample =
            Math.max(
                -1,
                Math.min(
                    1,
                    samples[i]
                )
            );


        view.setInt16(
            offset,
            sample < 0
                ? sample * 32768
                : sample * 32767,
            true
        );

        offset += 2;
    }


    return new Blob(
        [buffer],
        {
            type: "audio/wav"
        }
    );
}


/* ============================================================
   AUDIO OUTPUT
   ============================================================ */

function attachAudio(
    audioId,
    downloadId,
    samples,
    fs,
    filename
) {

    const blob =
        encodeWav(
            samples,
            fs
        );

    const url =
        URL.createObjectURL(blob);

    objectUrls.push(url);

    if ($(audioId)) {
        $(audioId).src = url;
    }

    if ($(downloadId)) {

        $(downloadId).href = url;

        $(downloadId).download =
            filename;
    }
}


/* ============================================================
   FIR LOW-PASS
   Hamming Window
   ============================================================ */

function designFIRLowpass(
    taps,
    cutoff,
    fs
) {

    taps = nextOdd(taps);

    const h =
        new Float32Array(taps);

    const center =
        (taps - 1) / 2;

    const fc =
        cutoff / fs;


    for (
        let n = 0;
        n < taps;
        n++
    ) {

        const k =
            n - center;

        let sinc;


        if (k === 0) {

            sinc =
                2 * fc;

        } else {

            sinc =
                Math.sin(
                    2 *
                    Math.PI *
                    fc *
                    k
                )
                /
                (
                    Math.PI * k
                );
        }


        const window =
            0.54 -
            0.46 *
            Math.cos(
                2 *
                Math.PI *
                n /
                (taps - 1)
            );


        h[n] =
            sinc * window;
    }


    let sum = 0;

    for (
        let i = 0;
        i < taps;
        i++
    ) {

        sum += h[i];
    }


    if (Math.abs(sum) > 1e-12) {

        for (
            let i = 0;
            i < taps;
            i++
        ) {

            h[i] /= sum;
        }
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


    for (
        let i = 0;
        i < taps;
        i++
    ) {

        high[i] =
            -low[i];
    }


    high[center] += 1;

    return high;
}


/* ============================================================
   FIR CONVOLUTION
   ============================================================ */

async function convolveOffline(
    input,
    kernel,
    fs
) {

    const length =
        input.length +
        kernel.length -
        1;


    const context =
        new OfflineAudioContext(
            1,
            length,
            fs
        );


    const source =
        context.createBufferSource();


    const inputBuffer =
        context.createBuffer(
            1,
            input.length,
            fs
        );


    inputBuffer.copyToChannel(
        input,
        0
    );


    const impulse =
        context.createBuffer(
            1,
            kernel.length,
            fs
        );


    impulse.copyToChannel(
        kernel,
        0
    );


    const convolver =
        context.createConvolver();

    convolver.normalize =
        false;

    convolver.buffer =
        impulse;


    source.buffer =
        inputBuffer;


    source.connect(
        convolver
    );

    convolver.connect(
        context.destination
    );


    source.start();


    const rendered =
        await context.startRendering();


    return rendered
        .getChannelData(0)
        .slice(
            0,
            input.length
        );
}


/* ============================================================
   IIR BUTTERWORTH
   ============================================================ */

function butterworthQValues(order) {

    const values = [];

    const sections =
        order / 2;


    for (
        let k = 0;
        k < sections;
        k++
    ) {

        const theta =
            Math.PI *
            (2 * k + 1) /
            (2 * order);


        values.push(
            1 /
            (
                2 *
                Math.cos(theta)
            )
        );
    }


    return values;
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


    const a0 =
        1 + alpha;

    let a1 =
        -2 * cos;

    let a2 =
        1 - alpha;


    b0 /= a0;
    b1 /= a0;
    b2 /= a0;

    a1 /= a0;
    a2 /= a0;


    const output =
        new Float32Array(
            input.length
        );


    let x1 = 0;
    let x2 = 0;

    let y1 = 0;
    let y2 = 0;


    for (
        let i = 0;
        i < input.length;
        i++
    ) {

        const x0 =
            input[i];


        const y0 =
            b0 * x0 +
            b1 * x1 +
            b2 * x2 -
            a1 * y1 -
            a2 * y2;


        output[i] =
            y0;


        x2 = x1;
        x1 = x0;

        y2 = y1;
        y1 = y0;
    }


    return output;
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


    let output =
        input.slice();


    for (
        const Q of qValues
    ) {

        output =
            processBiquad(
                output,
                fs,
                cutoff,
                type,
                Q
            );
    }


    return output;
}


/* ============================================================
   FFT
   ============================================================ */

function reverseBits(
    value,
    bits
) {

    let result = 0;


    for (
        let i = 0;
        i < bits;
        i++
    ) {

        result =
            (
                result << 1
            )
            |
            (
                value & 1
            );

        value >>>= 1;
    }


    return result;
}


function fft(
    real,
    imag,
    inverse = false
) {

    const n =
        real.length;


    const bits =
        Math.round(
            Math.log2(n)
        );


    /* Bit reversal */

    for (
        let i = 0;
        i < n;
        i++
    ) {

        const j =
            reverseBits(
                i,
                bits
            );


        if (j > i) {

            let temp =
                real[i];

            real[i] =
                real[j];

            real[j] =
                temp;


            temp =
                imag[i];

            imag[i] =
                imag[j];

            imag[j] =
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


        const angle =
            (
                inverse ? 1 : -1
            )
            *
            2 *
            Math.PI /
            size;


        const cos =
            Math.cos(angle);

        const sin =
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
                    wr * real[odd] -
                    wi * imag[odd];


                const ti =
                    wr * imag[odd] +
                    wi * real[odd];


                const er =
                    real[even];

                const ei =
                    imag[even];


                real[even] =
                    er + tr;

                imag[even] =
                    ei + ti;


                real[odd] =
                    er - tr;

                imag[odd] =
                    ei - ti;


                const nextWr =
                    wr * cos -
                    wi * sin;


                wi =
                    wr * sin +
                    wi * cos;


                wr =
                    nextWr;
            }
        }
    }


    if (inverse) {

        for (
            let i = 0;
            i < n;
            i++
        ) {

            real[i] /= n;
            imag[i] /= n;
        }
    }
}


/* ============================================================
   FFT SEPARATION
   ============================================================ */

async function fftSeparation(
    input,
    fs,
    cutoff,
    report
) {

    const frameSize =
        16384;

    const hop =
        frameSize / 2;


    const frameCount =
        Math.max(
            1,
            Math.ceil(
                (
                    input.length -
                    frameSize
                ) /
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
        frame < frameCount;
        frame++
    ) {

        const start =
            frame * hop;


        const real =
            new Float64Array(
                frameSize
            );


        const imag =
            new Float64Array(
                frameSize
            );


        for (
            let n = 0;
            n < frameSize;
            n++
        ) {

            const index =
                start + n;


            real[n] =
                index < input.length
                    ? input[index] *
                      window[n]
                    : 0;
        }


        fft(
            real,
            imag,
            false
        );


        const speechReal =
            new Float64Array(
                frameSize
            );

        const speechImag =
            new Float64Array(
                frameSize
            );

        const musicReal =
            new Float64Array(
                frameSize
            );

        const musicImag =
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
                    (
                        k -
                        frameSize
                    ) *
                    fs /
                    frameSize;
            }


            if (
                Math.abs(frequency)
                <= cutoff
            ) {

                speechReal[k] =
                    real[k];

                speechImag[k] =
                    imag[k];

            } else {

                musicReal[k] =
                    real[k];

                musicImag[k] =
                    imag[k];
            }
        }


        fft(
            speechReal,
            speechImag,
            true
        );


        fft(
            musicReal,
            musicImag,
            true
        );


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
                    speechReal[n] *
                    window[n];


                music[index] +=
                    musicReal[n] *
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
                frame === frameCount - 1
            )
        ) {

            report(
                20 +
                40 *
                (
                    (frame + 1) /
                    frameCount
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
   METRICS
   ============================================================ */

function findBestLag(
    reference,
    estimated,
    maxLagSamples = 1000
) {

    const step = 16;

    const maxLag =
        Math.floor(
            maxLagSamples / step
        );


    const length =
        Math.min(
            reference.length,
            estimated.length,
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

        let xy = 0;
        let xx = 0;
        let yy = 0;


        for (
            let i = 0;
            i < length;
            i += step
        ) {

            const j =
                i +
                lag * step;


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


            xy += x * y;
            xx += x * x;
            yy += y * y;
        }


        const denominator =
            Math.sqrt(
                xx * yy
            );


        if (
            denominator >
            1e-12
        ) {

            const value =
                xy /
                denominator;


            if (
                value >
                bestCorrelation
            ) {

                bestCorrelation =
                    value;

                bestLag =
                    lag * step;
            }
        }
    }


    return bestLag;
}


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


    let referenceStart = 0;

    let estimatedStart = 0;


    if (lag > 0) {

        estimatedStart =
            lag;

    } else {

        referenceStart =
            -lag;
    }


    const length =
        Math.min(
            n - referenceStart,
            n - estimatedStart
        );


    return {

        reference:
            reference.slice(
                referenceStart,
                referenceStart + length
            ),

        estimated:
            estimated.slice(
                estimatedStart,
                estimatedStart + length
            )
    };
}


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


    if (!n) return 0;


    let meanX = 0;
    let meanY = 0;


    for (
        let i = 0;
        i < n;
        i++
    ) {

        meanX += x[i];
        meanY += y[i];
    }


    meanX /= n;
    meanY /= n;


    let numerator = 0;
    let xx = 0;
    let yy = 0;


    for (
        let i = 0;
        i < n;
        i++
    ) {

        const dx =
            x[i] - meanX;

        const dy =
            y[i] - meanY;


        numerator +=
            dx * dy;

        xx +=
            dx * dx;

        yy +=
            dy * dy;
    }


    const denominator =
        Math.sqrt(
            xx * yy
        );


    return denominator >
        1e-12
        ? numerator / denominator
        : 0;
}


function calculateSNR(
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


    if (!n) return 0;


    let xy = 0;
    let yy = 0;


    for (
        let i = 0;
        i < n;
        i++
    ) {

        xy +=
            x[i] * y[i];

        yy +=
            y[i] * y[i];
    }


    const scale =
        xy /
        (
            yy + 1e-12
        );


    let signalPower = 0;
    let errorPower = 0;


    for (
        let i = 0;
        i < n;
        i++
    ) {

        const estimate =
            scale * y[i];


        const error =
            x[i] - estimate;


        signalPower +=
            x[i] * x[i];


        errorPower +=
            error * error;
    }


    signalPower /= n;
    errorPower /= n;


    if (
        errorPower <
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


function calculateRMSE(
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


    if (!n) return 0;


    let xy = 0;
    let yy = 0;


    for (
        let i = 0;
        i < n;
        i++
    ) {

        xy +=
            x[i] * y[i];

        yy +=
            y[i] * y[i];
    }


    const scale =
        xy /
        (
            yy + 1e-12
        );


    let error = 0;


    for (
        let i = 0;
        i < n;
        i++
    ) {

        const difference =
            x[i] -
            scale * y[i];


        error +=
            difference *
            difference;
    }


    return Math.sqrt(
        error / n
    );
}


function formatMetric(value) {

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

        <td>${formatMetric(
            correlation(
                speechRef,
                speech
            )
        )}</td>

        <td>${formatMetric(
            correlation(
                musicRef,
                music
            )
        )}</td>

        <td>${formatMetric(
            calculateSNR(
                speechRef,
                speech
            )
        )}</td>

        <td>${formatMetric(
            calculateSNR(
                musicRef,
                music
            )
        )}</td>

        <td>${formatMetric(
            calculateRMSE(
                speechRef,
                speech
            )
        )}</td>

        <td>${formatMetric(
            calculateRMSE(
                musicRef,
                music
            )
        )}</td>
    </tr>`;
}


/* ============================================================
   CANVAS SETUP
   ============================================================ */

function clearCanvas(canvas) {

    if (!canvas) {
        return null;
    }


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
            240,
            Math.floor(
                rect.height
            )
        );


    const dpr =
        Math.min(
            window.devicePixelRatio || 1,
            2
        );


    canvas.width =
        Math.floor(
            width * dpr
        );


    canvas.height =
        Math.floor(
            height * dpr
        );


    const ctx =
        canvas.getContext("2d");


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


    /*
       White background makes the graphs
       much sharper when captured/screenshot.
    */

    ctx.fillStyle =
        "#ffffff";

    ctx.fillRect(
        0,
        0,
        width,
        height
    );


    return {
        ctx,
        w: width,
        h: height
    };
}


/* ============================================================
   AXES
   ============================================================ */

function drawAxes(
    ctx,
    w,
    h,
    xLabel,
    yLabel,
    top = 42
) {

    const left = 62;
    const right = 24;
    const bottom = 42;


    ctx.strokeStyle =
        "#d9dee8";

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


    /*
       Horizontal zero/reference grid.
    */

    const middle =
        top +
        (
            h -
            top -
            bottom
        ) / 2;


    ctx.strokeStyle =
        "#edf0f5";

    ctx.beginPath();

    ctx.moveTo(
        left,
        middle
    );

    ctx.lineTo(
        w - right,
        middle
    );

    ctx.stroke();


    /*
       X axis label.
    */

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


    /*
       Y axis label.
    */

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
   CLEAN WAVEFORM GRAPH
   ============================================================ */

function drawWave(
    canvasId,
    data,
    fs,
    title
) {

    const canvas =
        $(canvasId);

    if (!canvas) return;


    const graph =
        clearCanvas(canvas);


    const {
        ctx,
        w,
        h
    } = graph;


    /*
       Show 15 seconds.
       This is enough to demonstrate
       the waveform without making it
       look like a solid block.
    */

    const seconds =
        Math.min(
            15,
            data.length / fs
        );


    const sampleCount =
        Math.max(
            1,
            Math.floor(
                seconds * fs
            )
        );


    /*
       Around 2500 visual points.
    */

    const maxPoints = 2500;

    const step =
        Math.max(
            1,
            Math.floor(
                sampleCount /
                maxPoints
            )
        );


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Time (s)",
            "Amplitude",
            38
        );


    /*
       Fixed amplitude range gives
       consistent graphs.
    */

    const minAmplitude = -1;

    const maxAmplitude = 1;


    /*
       Waveform.
    */

    ctx.strokeStyle =
        "#6540c7";

    ctx.lineWidth =
        1.15;

    ctx.beginPath();


    let point = 0;


    for (
        let i = 0;
        i < sampleCount;
        i += step
    ) {

        const time =
            i / fs;


        const x =
            axis.left +
            (
                time /
                seconds
            ) *
            (
                w -
                axis.left -
                axis.right
            );


        const value =
            Math.max(
                minAmplitude,
                Math.min(
                    maxAmplitude,
                    data[i]
                )
            );


        const y =
            axis.top +
            (
                1 -
                (
                    value -
                    minAmplitude
                ) /
                (
                    maxAmplitude -
                    minAmplitude
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


    /*
       Title.
    */

    ctx.fillStyle =
        "#17203a";

    ctx.font =
        "bold 12px Arial";

    ctx.textAlign =
        "left";

    ctx.fillText(
        title,
        axis.left,
        18
    );


    /*
       Small time ticks.
    */

    ctx.fillStyle =
        "#7b8498";

    ctx.font =
        "10px Arial";

    ctx.textAlign =
        "center";


    const ticks = 5;


    for (
        let i = 0;
        i <= ticks;
        i++
    ) {

        const time =
            seconds *
            i /
            ticks;


        const x =
            axis.left +
            (
                i / ticks
            ) *
            (
                w -
                axis.left -
                axis.right
            );


        ctx.fillText(
            `${time.toFixed(1)}`,
            x,
            h - 26
        );
    }
}


/* ============================================================
   SPECTRUM CALCULATION
   ============================================================ */

function magnitudeSpectrum(
    data,
    fs
) {

    /*
       Use first 30 seconds,
       but limit FFT size.
    */

    const maximumSamples =
        Math.min(
            data.length,
            Math.floor(
                30 * fs
            ),
            262144
        );


    let n = 1;


    while (
        (n << 1) <= maximumSamples
    ) {

        n <<= 1;
    }


    const real =
        new Float64Array(n);

    const imag =
        new Float64Array(n);


    /*
       Hann window.
    */

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


        real[i] =
            data[i] *
            window;
    }


    fft(
        real,
        imag,
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
                        real[k],
                        imag[k]
                    ) /
                    n
                )
            );
    }


    return {
        frequency,
        magnitude
    };
}


/* ============================================================
   CLEAN SPECTRUM GRAPH
   ============================================================ */

function drawSpectrum(
    canvasId,
    data,
    fs,
    title,
    cutoff
) {

    const canvas =
        $(canvasId);

    if (!canvas) return;


    const graph =
        clearCanvas(canvas);


    const {
        ctx,
        w,
        h
    } = graph;


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Frequency (Hz)",
            "Magnitude (dB)",
            40
        );


    const spectrum =
        magnitudeSpectrum(
            data,
            fs
        );


    /*
       Show 0–8 kHz,
       exactly like your current project.
    */

    const maxFrequency =
        Math.min(
            8000,
            fs / 2
        );


    const minDb = -90;

    const maxDb = 0;


    /*
       Maximum number of visual
       points. This is the main
       anti-blur improvement.
    */

    const maxPoints = 1800;


    const totalBins =
        spectrum.frequency.length;


    const step =
        Math.max(
            1,
            Math.ceil(
                totalBins /
                maxPoints
            )
        );


    ctx.strokeStyle =
        "#0798ad";

    ctx.lineWidth =
        1.2;

    ctx.beginPath();


    let started = false;


    for (
        let i = 0;
        i < totalBins;
        i += step
    ) {

        const frequency =
            spectrum.frequency[i];


        if (
            frequency >
            maxFrequency
        ) {
            break;
        }


        const db =
            Math.max(
                minDb,
                Math.min(
                    maxDb,
                    spectrum.magnitude[i]
                )
            );


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


    /*
       Cutoff line.
    */

    if (
        cutoff > 0 &&
        cutoff < maxFrequency
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
            1.4;

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
                cutoffX + 7,
                w - 110
            ),
            axis.top + 16
        );
    }


    /*
       Title.
    */

    ctx.fillStyle =
        "#17203a";

    ctx.font =
        "bold 12px Arial";

    ctx.textAlign =
        "left";


    ctx.fillText(
        title,
        axis.left,
        18
    );


    /*
       Frequency ticks.
    */

    ctx.fillStyle =
        "#7b8498";

    ctx.font =
        "10px Arial";

    ctx.textAlign =
        "center";


    const frequencyTicks =
        [0, 2000, 4000, 6000, 8000];


    for (
        const frequency of frequencyTicks
    ) {

        if (
            frequency >
            maxFrequency
        ) {
            continue;
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


        ctx.fillText(
            `${frequency}`,
            x,
            h - 26
        );
    }
}


/* ============================================================
   OVERALL WAVEFORM COMPARISON
   ============================================================ */

function drawOverallWave(
    canvasId,
    series,
    fs
) {

    const canvas =
        $(canvasId);

    if (!canvas) return;


    const graph =
        clearCanvas(canvas);


    const {
        ctx,
        w,
        h
    } = graph;


    /*
       12 seconds gives a clean
       side-by-side comparison.
    */

    const seconds =
        Math.min(
            12,
            series[0].data.length / fs
        );


    const sampleCount =
        Math.floor(
            seconds * fs
        );


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Time (s)",
            "Amplitude",
            58
        );


    const colors = [
        "#17203a",
        "#6540c7",
        "#0798ad",
        "#d27618"
    ];


    const labels = [
        "Mixed Input",
        "FFT Speech",
        "FIR Speech",
        "IIR Speech"
    ];


    /*
       Draw clean legend ABOVE the graph.
    */

    let legendX =
        axis.left;


    const legendY = 31;


    ctx.font =
        "bold 11px Arial";

    ctx.textAlign =
        "left";


    for (
        let i = 0;
        i < Math.min(
            series.length,
            labels.length
        );
        i++
    ) {

        ctx.strokeStyle =
            colors[i];

        ctx.lineWidth = 3;


        ctx.beginPath();

        ctx.moveTo(
            legendX,
            legendY
        );

        ctx.lineTo(
            legendX + 18,
            legendY
        );

        ctx.stroke();


        ctx.fillStyle =
            colors[i];


        ctx.fillText(
            labels[i],
            legendX + 24,
            legendY + 4
        );


        legendX += 135;
    }


    /*
       Draw each waveform.
    */

    const maxPoints = 1800;


    const step =
        Math.max(
            1,
            Math.floor(
                sampleCount /
                maxPoints
            )
        );


    for (
        let s = 0;
        s < Math.min(
            series.length,
            colors.length
        );
        s++
    ) {

        const data =
            series[s].data;


        ctx.strokeStyle =
            colors[s];


        ctx.lineWidth =
            s === 0
                ? 1.25
                : 1.05;


        ctx.globalAlpha =
            s === 0
                ? 0.65
                : 0.78;


        ctx.beginPath();


        let point = 0;


        for (
            let i = 0;
            i < sampleCount;
            i += step
        ) {

            const time =
                i / fs;


            const x =
                axis.left +
                (
                    time /
                    seconds
                ) *
                (
                    w -
                    axis.left -
                    axis.right
                );


            const value =
                Math.max(
                    -1,
                    Math.min(
                        1,
                        data[i]
                    )
                );


            const y =
                axis.top +
                (
                    1 -
                    (
                        value + 1
                    ) / 2
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


    ctx.globalAlpha = 1;


    /*
       Title.
    */

    ctx.fillStyle =
        "#17203a";

    ctx.font =
        "bold 13px Arial";

    ctx.textAlign =
        "left";


    ctx.fillText(
        "Overall Waveform Comparison",
        axis.left,
        17
    );
}


/* ============================================================
   OVERALL FFT SPECTRUM
   ============================================================ */

function drawOverallSpec(
    canvasId,
    series,
    fs
) {

    const canvas =
        $(canvasId);

    if (!canvas) return;


    const graph =
        clearCanvas(canvas);


    const {
        ctx,
        w,
        h
    } = graph;


    const axis =
        drawAxes(
            ctx,
            w,
            h,
            "Frequency (Hz)",
            "Magnitude (dB)",
            58
        );


    const maxFrequency =
        Math.min(
            8000,
            fs / 2
        );


    const colors = [
        "#17203a",
        "#6540c7",
        "#0798ad",
        "#d27618"
    ];


    const labels = [
        "Mixed Input",
        "FFT Speech",
        "FIR Speech",
        "IIR Speech"
    ];


    /*
       Legend at top.
    */

    let legendX =
        axis.left;


    const legendY = 31;


    ctx.font =
        "bold 11px Arial";

    ctx.textAlign =
        "left";


    for (
        let i = 0;
        i < Math.min(
            series.length,
            labels.length
        );
        i++
    ) {

        ctx.strokeStyle =
            colors[i];

        ctx.lineWidth = 3;


        ctx.beginPath();

        ctx.moveTo(
            legendX,
            legendY
        );

        ctx.lineTo(
            legendX + 18,
            legendY
        );

        ctx.stroke();


        ctx.fillStyle =
            colors[i];


        ctx.fillText(
            labels[i],
            legendX + 24,
            legendY + 4
        );


        legendX += 135;
    }


    /*
       Draw spectra.
    */

    const minDb = -90;

    const maxDb = 0;


    for (
        let s = 0;
        s < Math.min(
            series.length,
            colors.length
        );
        s++
    ) {

        const spectrum =
            magnitudeSpectrum(
                series[s].data,
                fs
            );


        const totalBins =
            spectrum.frequency.length;


        const maxPoints = 1500;


        const step =
            Math.max(
                1,
                Math.ceil(
                    totalBins /
                    maxPoints
                )
            );


        ctx.strokeStyle =
            colors[s];


        ctx.lineWidth =
            s === 0
                ? 1.25
                : 1.05;


        ctx.globalAlpha =
            s === 0
                ? 0.55
                : 0.70;


        ctx.beginPath();


        let started = false;


        for (
            let i = 0;
            i < totalBins;
            i += step
        ) {

            const frequency =
                spectrum.frequency[i];


            if (
                frequency >
                maxFrequency
            ) {
                break;
            }


            const db =
                Math.max(
                    minDb,
                    Math.min(
                        maxDb,
                        spectrum.magnitude[i]
                    )
                );


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
    }


    ctx.globalAlpha = 1;


    /*
       Title.
    */

    ctx.fillStyle =
        "#17203a";

    ctx.font =
        "bold 13px Arial";

    ctx.textAlign =
        "left";


    ctx.fillText(
        "Overall FFT Spectrum Comparison",
        axis.left,
        17
    );


    /*
       Frequency ticks.
    */

    ctx.fillStyle =
        "#7b8498";

    ctx.font =
        "10px Arial";

    ctx.textAlign =
        "center";


    const ticks =
        [0, 2000, 4000, 6000, 8000];


    for (
        const frequency of ticks
    ) {

        if (
            frequency >
            maxFrequency
        ) {
            continue;
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


        ctx.fillText(
            `${frequency}`,
            x,
            h - 26
        );
    }
}


/* ============================================================
   MAIN PROCESS
   ============================================================ */

async function processProject() {

    const mixFile =
        $("mixFile")?.files[0];


    if (!mixFile) {

        setStatus(
            "Please select the mixed WAV input first."
        );

        return;
    }


    $("startBtn").disabled =
        true;


    if ($("results")) {

        $("results")
            .classList
            .add("hidden");
    }


    /*
       Clean old object URLs.
    */

    objectUrls.forEach(
        url =>
            URL.revokeObjectURL(url)
    );


    objectUrls = [];


    try {

        /* ----------------------------------------------------
           MIXED INPUT
           ---------------------------------------------------- */

        setStatus(
            "Loading mixed audio...",
            2
        );


        const mixAudio =
            await decodeFile(
                mixFile
            );


        const mix =
            mixAudio.samples;

        const fs =
            mixAudio.fs;


        /* ----------------------------------------------------
           PARAMETERS
           ---------------------------------------------------- */

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
            Number(
                $("iirOrder").value
            );


        if (
            cutoff <= 0 ||
            cutoff >= fs / 2
        ) {

            throw new Error(
                `Cutoff must be between 1 and ${Math.floor(fs / 2 - 1)} Hz.`
            );
        }


        /* ----------------------------------------------------
           FFT
           ---------------------------------------------------- */

        setStatus(
            "Running FFT frequency-domain separation...",
            5
        );


        const fftOutput =
            await fftSeparation(
                mix,
                fs,
                cutoff,
                progress => {

                    setStatus(
                        "FFT separation in progress...",
                        progress
                    );
                }
            );


        /* ----------------------------------------------------
           FIR
           ---------------------------------------------------- */

        setStatus(
            "Designing FIR filters...",
            65
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


        setStatus(
            "Applying FIR filters...",
            70
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


        /* ----------------------------------------------------
           IIR
           ---------------------------------------------------- */

        setStatus(
            "Applying IIR Butterworth filters...",
            80
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


        /* ----------------------------------------------------
           REFERENCE FILES
           ---------------------------------------------------- */

        setStatus(
            "Loading optional reference signals...",
            86
        );


        let speechReference =
            null;

        let musicReference =
            null;


        if (
            $("speechFile")?.files[0]
        ) {

            const reference =
                await decodeFile(
                    $("speechFile").files[0]
                );


            if (
                reference.fs !== fs
            ) {

                throw new Error(
                    "Speech reference sampling rate does not match the mixed signal."
                );
            }


            speechReference =
                normalize(
                    reference.samples
                        .slice(
                            0,
                            mix.length
                        )
                );
        }


        if (
            $("musicFile")?.files[0]
        ) {

            const reference =
                await decodeFile(
                    $("musicFile").files[0]
                );


            if (
                reference.fs !== fs
            ) {

                throw new Error(
                    "Music reference sampling rate does not match the mixed signal."
                );
            }


            musicReference =
                normalize(
                    reference.samples
                        .slice(
                            0,
                            mix.length
                        )
                );
        }


        /* ----------------------------------------------------
           AUDIO OUTPUTS
           ---------------------------------------------------- */

        setStatus(
            "Creating audio outputs...",
            90
        );


        attachAudio(
            "fftSpeechAudio",
            "fftSpeechDownload",
            fftOutput.speech,
            fs,
            "FFT_Speech.wav"
        );


        attachAudio(
            "fftMusicAudio",
            "fftMusicDownload",
            fftOutput.music,
            fs,
            "FFT_Music.wav"
        );


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


        /* ----------------------------------------------------
           INDIVIDUAL WAVEFORMS
           ---------------------------------------------------- */

        setStatus(
            "Drawing time-domain waveforms...",
            93
        );


        drawWave(
            "waveMix",
            mix,
            fs,
            "Mixed Input"
        );


        drawWave(
            "waveFftSpeech",
            fftOutput.speech,
            fs,
            "FFT Speech"
        );


        drawWave(
            "waveFftMusic",
            fftOutput.music,
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


        /* ----------------------------------------------------
           INDIVIDUAL SPECTRA
           ---------------------------------------------------- */

        setStatus(
            "Drawing frequency spectra...",
            95
        );


        drawSpectrum(
            "specMix",
            mix,
            fs,
            "Mixed Input Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFftSpeech",
            fftOutput.speech,
            fs,
            "FFT Speech Spectrum",
            cutoff
        );


        drawSpectrum(
            "specFftMusic",
            fftOutput.music,
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


        /* ----------------------------------------------------
           OVERALL COMPARISONS
           ---------------------------------------------------- */

        drawOverallWave(
            "overallWave",
            [
                {
                    data: mix
                },
                {
                    data: fftOutput.speech
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
                    data: fftOutput.speech
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


        /* ----------------------------------------------------
           PERFORMANCE TABLE
           ---------------------------------------------------- */

        if (
            $("metricsTable")
        ) {

            const tbody =
                $("metricsTable")
                    .querySelector("tbody");


            if (tbody) {

                tbody.innerHTML =

                    metricsRow(
                        "FFT",
                        fftOutput.speech,
                        fftOutput.music,
                        speechReference,
                        musicReference
                    )

                    +

                    metricsRow(
                        "FIR",
                        firSpeech,
                        firMusic,
                        speechReference,
                        musicReference
                    )

                    +

                    metricsRow(
                        "IIR",
                        iirSpeech,
                        iirMusic,
                        speechReference,
                        musicReference
                    );
            }
        }


        /* ----------------------------------------------------
           SAVE RESULT
           ---------------------------------------------------- */

        lastResult = {

            fs,
            mix,

            fftSpeech:
                fftOutput.speech,

            fftMusic:
                fftOutput.music,

            firSpeech,
            firMusic,

            iirSpeech,
            iirMusic,

            cutoff
        };


        /* ----------------------------------------------------
           SHOW RESULTS
           ---------------------------------------------------- */

        if ($("results")) {

            $("results")
                .classList
                .remove("hidden");
        }


        setStatus(
            `Completed successfully • ${(mix.length / fs).toFixed(2)} seconds • ${fs} Hz • Cutoff ${cutoff} Hz`,
            100
        );


        /*
           Scroll to results.
        */

        setTimeout(() => {

            if ($("results")) {

                $("results").scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

        }, 150);


    } catch (error) {

        console.error(
            "DSP processing error:",
            error
        );


        setStatus(
            "Error: " +
            (
                error.message ||
                "Unable to process audio."
            )
        );

    } finally {

        $("startBtn").disabled =
            false;
    }
}


/* ============================================================
   START BUTTON
   ============================================================ */

if ($("startBtn")) {

    $("startBtn")
        .addEventListener(
            "click",
            processProject
        );
}


/* ============================================================
   RESPONSIVE REDRAW
   ============================================================ */

let resizeTimer = null;


window.addEventListener(
    "resize",
    () => {

        clearTimeout(
            resizeTimer
        );


        resizeTimer =
            setTimeout(
                () => {

                    if (!lastResult) {
                        return;
                    }


                    const {

                        fs,
                        mix,
                        fftSpeech,
                        fftMusic,
                        firSpeech,
                        firMusic,
                        iirSpeech,
                        iirMusic,
                        cutoff

                    } =
                        lastResult;


                    /* Waveforms */

                    drawWave(
                        "waveMix",
                        mix,
                        fs,
                        "Mixed Input"
                    );


                    drawWave(
                        "waveFftSpeech",
                        fftSpeech,
                        fs,
                        "FFT Speech"
                    );


                    drawWave(
                        "waveFftMusic",
                        fftMusic,
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
                        fftSpeech,
                        fs,
                        "FFT Speech Spectrum",
                        cutoff
                    );


                    drawSpectrum(
                        "specFftMusic",
                        fftMusic,
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
                                data: fftSpeech
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
                                data: fftSpeech
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

                },
                250
            );
    }
);
