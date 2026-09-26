# DSP Audio Separation Studio — GitHub Pages

This version runs entirely in the browser using **HTML + CSS + JavaScript**.

It does **not** require:
- Gradio
- Python
- Flask
- Hugging Face
- a backend server

## Files

- `index.html` — user interface
- `style.css` — UI design
- `script.js` — browser-side DSP processing
- `.nojekyll` — tells GitHub Pages to serve the files directly

## How it works

The user supplies the mixed WAV file. The browser decodes it with the Web Audio API and performs:

1. FFT-based frequency-domain separation
2. Hamming-window FIR low-pass/high-pass filtering
3. Butterworth IIR low-pass/high-pass filtering
4. Time-domain waveform plotting
5. Frequency-spectrum plotting
6. Optional reference-based correlation, SNR and RMSE
7. WAV generation and download for all six outputs

The speech and music reference WAV files are **optional** and are used only for evaluation metrics. They are not used as separator inputs.

## GitHub Pages

In the repository:

1. Open **Settings**
2. Open **Pages**
3. Under **Build and deployment**, choose **Deploy from a branch**
4. Select `main` and `/ (root)`
5. Click **Save**
6. Wait for GitHub Pages to deploy
7. Open the generated `github.io` URL

GitHub Pages is a static hosting service and can publish HTML, CSS and JavaScript directly from a repository.

## Important

The browser version intentionally uses memory-conscious visualization. Long recordings are processed in FFT frames rather than one enormous FFT of the complete recording.

For the best browser performance, Chrome/Edge and a reasonably powerful computer are recommended.
