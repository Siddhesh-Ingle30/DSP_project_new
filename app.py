import gc, numpy as np, pandas as pd, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from scipy import signal
from scipy.signal import firwin, butter, lfilter, sosfilt

BG = "https://i.ibb.co/9mMML0H9/background.png"

def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim > 1: x = x.mean(axis=1)
    x = x.astype(np.float64)
    x -= x.mean()
    return x, fs

def norm(x, peak=0.95):
    m = np.max(np.abs(x))
    return x if m == 0 else x / m * peak

def corr(r, e):
    r, e = r-r.mean(), e-e.mean()
    d = np.sqrt(np.sum(r*r)*np.sum(e*e))
    return 0 if d == 0 else np.sum(r*e)/d

def snr(r, e):
    s = np.dot(r,e)/(np.dot(e,e)+1e-12)
    err = r-s*e
    return np.inf if np.mean(err*err)==0 else 10*np.log10(np.mean(r*r)/np.mean(err*err))

def rmse(r, e):
    s = np.dot(r,e)/(np.dot(e,e)+1e-12)
    return np.sqrt(np.mean((r-s*e)**2))

# FFT frequency-domain separation.
def fft_sep(x, fs, cut):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1/fs)
    return np.fft.irfft(X*(f<=cut), n=len(x)), np.fft.irfft(X*(f>cut), n=len(x))

# FIR Hamming-window low-pass/high-pass separation.
def fir_sep(x, fs, cut, taps):
    lo = firwin(int(taps), cut, fs=fs, window="hamming", pass_zero="lowpass")
    hi = firwin(int(taps), cut, fs=fs, window="hamming", pass_zero="highpass")
    return lfilter(lo,[1.0],x), lfilter(hi,[1.0],x)

# IIR Butterworth low-pass/high-pass separation.
def iir_sep(x, fs, order, cut):
    lo = butter(int(order), cut, btype="lowpass", fs=fs, output="sos")
    hi = butter(int(order), cut, btype="highpass", fs=fs, output="sos")
    return sosfilt(lo,x), sosfilt(hi,x)

def waveform(x, fs, title, filename):
    n = min(len(x), int(30*fs))
    step = max(1, n//5000)
    fig = plt.figure(figsize=(10,4.5), dpi=120)
    plt.plot(np.arange(n)[::step]/fs, x[:n:step], lw=.8)
    plt.xlabel("Time (seconds)"); plt.ylabel("Amplitude"); plt.title(title, fontweight="bold")
    plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(filename, dpi=140, bbox_inches="tight"); plt.close(fig)
    return filename

def spectrum(x, fs, title, filename, cut):
    n = min(len(x), int(60*fs), 262144)
    x = x[:n]; N = len(x)
    X = np.fft.rfft(x*np.hanning(N))
    f = np.fft.rfftfreq(N, 1/fs); m = f <= 8000
    fig = plt.figure(figsize=(10,4.5), dpi=120)
    plt.plot(f[m], np.abs(X)[m]/N, lw=.8)
    plt.axvline(cut, ls="--", lw=2, label=f"Cutoff = {int(cut)} Hz")
    plt.legend(); plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude")
    plt.title(title, fontweight="bold"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(filename, dpi=140, bbox_inches="tight"); plt.close(fig)
    return filename

def overall_waveform(items, fs):
    n=min(len(items[0][1]),int(30*fs)); step=max(1,n//5000); t=np.arange(n)[::step]/fs
    fig=plt.figure(figsize=(12,6),dpi=120)
    for label,x in items: plt.plot(t,x[:n:step],lw=.7,label=label)
    plt.xlabel("Time (seconds)"); plt.ylabel("Amplitude"); plt.title("Overall Waveform Comparison",fontweight="bold")
    plt.legend(fontsize=8,ncol=2); plt.grid(alpha=.25); plt.tight_layout()
    name="Overall_Waveform_Comparison.png"; plt.savefig(name,dpi=140,bbox_inches="tight"); plt.close(fig); return name

def overall_spectrum(items, fs, cut):
    fig=plt.figure(figsize=(12,6),dpi=120)
    for label,x in items:
        n=min(len(x),int(60*fs),262144); N=n
        X=np.fft.rfft(x[:n]*np.hanning(N)); f=np.fft.rfftfreq(N,1/fs); m=f<=8000
        plt.plot(f[m],np.abs(X)[m]/N,lw=.7,label=label)
    plt.axvline(cut,ls="--",lw=2,label=f"Cutoff = {int(cut)} Hz")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude"); plt.title("Overall FFT Spectrum Comparison",fontweight="bold")
    plt.legend(fontsize=8,ncol=2); plt.grid(alpha=.25); plt.tight_layout()
    name="Overall_Spectrum_Comparison.png"; plt.savefig(name,dpi=140,bbox_inches="tight"); plt.close(fig); return name

def process_audio(mixed, ref_speech, ref_music, cut, taps, fir_order, iir_order):
    if mixed is None: raise gr.Error("Please upload the mixed audio file first.")
    x,fs=load_audio(mixed); samples=len(x); duration=samples/fs

    fft_s,fft_m=fft_sep(x,fs,cut)
    fir_s,fir_m=fir_sep(x,fs,cut,taps)
    iir_s,iir_m=iir_sep(x,fs,iir_order,cut)
    fft_s,fft_m,fir_s,fir_m,iir_s,iir_m=[norm(z) for z in (fft_s,fft_m,fir_s,fir_m,iir_s,iir_m)]

    audio_files=[
        ("FFT_Speech.wav",fft_s),("FFT_Music.wav",fft_m),
        ("FIR_Speech.wav",fir_s),("FIR_Music.wav",fir_m),
        ("IIR_Speech.wav",iir_s),("IIR_Music.wav",iir_m)]
    for name,z in audio_files: sf.write(name,z,fs)

    names=[("Mixed Input",x),("FFT Speech",fft_s),("FFT Music",fft_m),
           ("FIR Speech",fir_s),("FIR Music",fir_m),("IIR Speech",iir_s),("IIR Music",iir_m)]

    wave_files=[
        waveform(x,fs,"Mixed Input — Speech + Music","Mixed_Waveform.png"),
        waveform(fft_s,fs,"FFT Separated Speech — Waveform","FFT_Speech_Waveform.png"),
        waveform(fft_m,fs,"FFT Separated Music — Waveform","FFT_Music_Waveform.png"),
        waveform(fir_s,fs,"FIR Separated Speech — Waveform","FIR_Speech_Waveform.png"),
        waveform(fir_m,fs,"FIR Separated Music — Waveform","FIR_Music_Waveform.png"),
        waveform(iir_s,fs,"IIR Separated Speech — Waveform","IIR_Speech_Waveform.png"),
        waveform(iir_m,fs,"IIR Separated Music — Waveform","IIR_Music_Waveform.png")]

    spec_files=[
        spectrum(x,fs,"Mixed Input — Frequency Spectrum","Mixed_Spectrum.png",cut),
        spectrum(fft_s,fs,"FFT Separated Speech — Spectrum","FFT_Speech_Spectrum.png",cut),
        spectrum(fft_m,fs,"FFT Separated Music — Spectrum","FFT_Music_Spectrum.png",cut),
        spectrum(fir_s,fs,"FIR Separated Speech — Spectrum","FIR_Speech_Spectrum.png",cut),
        spectrum(fir_m,fs,"FIR Separated Music — Spectrum","FIR_Music_Spectrum.png",cut),
        spectrum(iir_s,fs,"IIR Separated Speech — Spectrum","IIR_Speech_Spectrum.png",cut),
        spectrum(iir_m,fs,"IIR Separated Music — Spectrum","IIR_Music_Spectrum.png",cut)]

    ow=overall_waveform(names,fs)
    os=overall_spectrum(names,fs,cut)

    rows=[]
    if ref_speech is not None and ref_music is not None:
        sr,sfs=load_audio(ref_speech); mr,mfs=load_audio(ref_music)
        if sfs!=fs: sr=signal.resample_poly(sr,fs,sfs)
        if mfs!=fs: mr=signal.resample_poly(mr,fs,mfs)
        n=min(len(sr),len(mr),len(fft_s),len(fft_m),len(fir_s),len(fir_m),len(iir_s),len(iir_m))
        sr,mr=sr[:n],mr[:n]
        for method,a,b in [("FFT",fft_s[:n],fft_m[:n]),("FIR",fir_s[:n],fir_m[:n]),("IIR",iir_s[:n],iir_m[:n])]:
            rows.append([method,round(corr(sr,a),4),round(corr(mr,b),4),round(snr(sr,a),2),round(snr(mr,b),2),round(rmse(sr,a),5),round(rmse(mr,b),5)])

    table=pd.DataFrame(rows,columns=["Method","Speech Correlation","Music Correlation","Speech SNR (dB)","Music SNR (dB)","Speech RMSE","Music RMSE"])
    delay=(int(taps)-1)//2
    result=f"""<div class="result-title">✅ PROCESSING COMPLETED</div>
<div class="result-subtitle">Audio separation and DSP analysis have been completed successfully.</div>
<div class="stats-grid">
<div class="stat-card"><div class="stat-label">SAMPLING RATE</div><div class="stat-value">{fs:,} Hz</div></div>
<div class="stat-card"><div class="stat-label">SAMPLES</div><div class="stat-value">{samples:,}</div></div>
<div class="stat-card"><div class="stat-label">DURATION</div><div class="stat-value">{duration:.2f} sec</div></div>
<div class="stat-card"><div class="stat-label">CUTOFF</div><div class="stat-value">{int(cut)} Hz</div></div></div>
<div class="result-heading">⚙️ DSP CONFIGURATION</div>
<div class="config-row"><span>FIR Taps <b>{int(taps)}</b></span><span>FIR Order <b>{int(fir_order)}</b></span><span>IIR Order <b>{int(iir_order)}</b></span><span>FIR Delay <b>{delay} samples</b></span></div>
<div class="result-heading">🔬 SEPARATION METHODS</div><div class="method-row">
<div class="method-item">🟣 <b>FFT</b><small>Frequency-domain separation</small></div>
<div class="method-item">💜 <b>FIR</b><small>Low-pass Speech / High-pass Music</small></div>
<div class="method-item">🔵 <b>IIR</b><small>Butterworth filtering</small></div></div>
<div class="success-box">🟢 <b>STATUS:</b> Separation and analysis completed successfully.</div>"""
    gc.collect()
    return (result,*[x[0] for x in audio_files],*wave_files,*spec_files,ow,os,table)

CSS = """
html,body{margin:0!important;padding:0!important;background:#080b25!important}
body::before{content:"";position:fixed;inset:0;z-index:-2;background-image:linear-gradient(rgba(5,8,30,.58),rgba(9,8,35,.68)),url("https://i.ibb.co/9mMML0H9/background.png");background-size:cover;background-position:center;background-repeat:no-repeat}
#root,.gradio-container{background:transparent!important}.gradio-container{max-width:1450px!important}
.hero,.glass-card,#result-panel,.analysis-card,.graph-card{background:rgba(248,250,252,.97)!important;border-radius:22px!important;box-shadow:0 14px 40px rgba(0,0,0,.32)!important;color:#172033!important}
.hero{margin:25px auto 30px;padding:32px;text-align:center}.hero-title{font-size:40px;font-weight:900;color:#172554!important}.hero-subtitle{color:#334155!important;font-size:17px;margin:5px}
.section-title{color:#172554!important;font-size:21px;font-weight:900;padding-bottom:10px;margin-bottom:18px;border-bottom:3px solid #c4b5fd}
.glass-card .prose,.glass-card .prose *,.glass-card .markdown,.glass-card .markdown *, .glass-card p{color:#334155!important}
.glass-card h1,.glass-card h2,.glass-card h3,.glass-card b,.glass-card label,.graph-card label,.analysis-card label{color:#172554!important}
#process-btn{width:100%!important;min-height:60px!important;border:0!important;border-radius:17px!important;color:#fff!important;font-size:19px!important;font-weight:900!important;background:linear-gradient(90deg,#6d28d9,#7c3aed,#0891b2)!important}
#result-panel{margin-top:25px;padding:28px}.result-title{color:#172554!important;font-size:29px;font-weight:900}.result-subtitle{color:#475569!important}
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat-card{background:linear-gradient(135deg,#eef2ff,#ecfeff);border:1px solid #c7d2fe;border-radius:16px;padding:18px}
.stat-label{color:#64748b!important;font-size:11px;font-weight:800}.stat-value{color:#0f172a!important;font-size:21px;font-weight:900}
.result-heading,.performance-heading{color:#4338ca!important;font-size:19px;font-weight:900;margin:25px 0 12px}.config-row{display:flex;flex-wrap:wrap;gap:12px}.config-row span{background:#f1f5f9;color:#475569!important;padding:10px 15px;border-radius:10px}.config-row b{color:#0f172a!important}
.method-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.method-item{padding:16px;border-radius:15px;border:1px solid #e2e8f0;background:#fff}.method-item b{color:#172554!important}.method-item small{display:block;color:#64748b!important;margin-top:6px}
.success-box{margin-top:20px;padding:14px 18px;border-radius:12px;background:#ecfdf5;border:1px solid #86efac;color:#166534!important}.graph-card{padding:12px!important}
.footer{margin-top:25px;padding:25px;text-align:center;color:#e2e8f0!important;background:rgba(15,23,42,.82);border-radius:18px}
@media(max-width:900px){.stats-grid{grid-template-columns:repeat(2,1fr)}.method-row{grid-template-columns:1fr}}@media(max-width:600px){.hero-title{font-size:29px}.stats-grid{grid-template-columns:1fr}}
"""

with gr.Blocks(title="DSP Audio Separation Studio",css=CSS) as demo:
    gr.HTML('<div class="hero"><div class="hero-title">🎧 DSP AUDIO SEPARATION STUDIO</div><div class="hero-subtitle">Speech & Music Separation using <b>FFT • FIR • IIR Filters</b></div><div class="hero-subtitle">Digital Signal Processing Analysis System</div></div>')
    with gr.Row():
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<div class="section-title">🎵 AUDIO INPUT</div>')
            mixed=gr.Audio(label="🎵 Mixed Audio — Speech + Music",type="filepath")
            ref_speech=gr.Audio(label="🗣️ Reference Speech — Optional",type="filepath")
            ref_music=gr.Audio(label="🎼 Reference Music — Optional",type="filepath")
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<div class="section-title">⚙️ DSP PARAMETERS</div>')
            cut=gr.Slider(500,3000,value=1200,step=100,label="🎚️ Cutoff Frequency (Hz)")
            taps=gr.Slider(51,301,value=101,step=10,label="🔧 FIR Number of Taps")
            fir_order=gr.Slider(50,300,value=100,step=10,label="📐 FIR Order")
            iir_order=gr.Slider(2,10,value=6,step=1,label="🔵 IIR Butterworth Order")

    gr.HTML("<br>")
    button=gr.Button("🚀  START AUDIO SEPARATION",elem_id="process-btn")
    info=gr.HTML('<div class="result-title">🎛️ SYSTEM READY</div><div class="result-subtitle">Upload the mixed audio and press <b>START AUDIO SEPARATION</b>.</div><div class="success-box">🟢 Ready to process FFT, FIR and IIR separation.</div>',elem_id="result-panel")

    gr.HTML('<div class="section-title" style="color:#fff!important;background:rgba(15,23,42,.82);padding:15px 20px;border-radius:15px">🔊 SEPARATED AUDIO OUTPUTS</div>')
    audio_out=[]
    for method,icon in [("FFT","🟣"),("FIR","💜"),("IIR","🔵")]:
        with gr.Column(elem_classes=["glass-card"]):
            gr.Markdown(f"## {icon} {method} METHOD")
            gr.Markdown("Low-pass → **Speech**\\n\\nHigh-pass → **Music**")
            audio_out += [gr.Audio(label=f"🗣️ {method} — Separated Speech",type="filepath",buttons=["download"]),
                          gr.Audio(label=f"🎼 {method} — Separated Music",type="filepath",buttons=["download"])]

    gr.HTML('<div class="section-title" style="color:#fff!important;background:rgba(15,23,42,.82);padding:15px 20px;border-radius:15px;margin-top:25px">📊 SIGNAL ANALYSIS</div>')
    gr.HTML('<div class="performance-heading">🎵 INDIVIDUAL TIME-DOMAIN WAVEFORMS</div>')
    wave_out=[]
    wave_labels=["📈 Mixed Input — Waveform","🗣️ FFT Speech — Waveform","🎼 FFT Music — Waveform","🗣️ FIR Speech — Waveform","🎼 FIR Music — Waveform","🗣️ IIR Speech — Waveform","🎼 IIR Music — Waveform"]
    for i in range(0,7,3):
        with gr.Row():
            for label in wave_labels[i:i+3]: wave_out.append(gr.Image(label=label,elem_classes=["graph-card"]))

    gr.HTML('<div class="performance-heading">📡 INDIVIDUAL FREQUENCY SPECTRA</div>')
    spec_out=[]
    spec_labels=["📡 Mixed Input — Spectrum","🗣️ FFT Speech — Spectrum","🎼 FFT Music — Spectrum","🗣️ FIR Speech — Spectrum","🎼 FIR Music — Spectrum","🗣️ IIR Speech — Spectrum","🎼 IIR Music — Spectrum"]
    for i in range(0,7,3):
        with gr.Row():
            for label in spec_labels[i:i+3]: spec_out.append(gr.Image(label=label,elem_classes=["graph-card"]))

    gr.HTML('<div class="performance-heading">🔬 OVERALL ANALYSIS & COMPARISON</div>')
    with gr.Row():
        overall_w=gr.Image(label="📈 Overall Waveform Comparison",elem_classes=["analysis-card"])
        overall_s=gr.Image(label="📡 Overall FFT Spectrum Comparison",elem_classes=["analysis-card"])

    gr.HTML('<div class="performance-heading">📊 PERFORMANCE EVALUATION</div>')
    table=gr.Dataframe(headers=["Method","Speech Correlation","Music Correlation","Speech SNR (dB)","Music SNR (dB)","Speech RMSE","Music RMSE"],interactive=False)

    gr.HTML('<div class="footer">🎧 <b>DSP Audio Separation Studio</b><br><br>FFT • FIR • IIR • Waveform Analysis • Spectrum Analysis<br><br>Digital Signal Processing Mini Project</div>')

    button.click(process_audio,[mixed,ref_speech,ref_music,cut,taps,fir_order,iir_order],
                 [info,*audio_out,*wave_out,*spec_out,overall_w,overall_s,table])

if __name__=="__main__":
    demo.launch()
