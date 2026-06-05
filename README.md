# 🎹 AI Music Generation with Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![music21](https://img.shields.io/badge/music21-Corpus-8B4513?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge)
![LSTM](https://img.shields.io/badge/Bidirectional-LSTM-6C5CE7?style=for-the-badge)
![MIDI](https://img.shields.io/badge/Export-MIDI%20%2F%20WAV-00B894?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

**A deep learning system that learns musical patterns from Bach chorales and generates original piano compositions — powered by Bidirectional LSTM, temperature sampling, and music21.**

[🚀 Quick Start](#getting-started) · [✨ Features](#features) · [🏗️ Architecture](#architecture) · [🎛️ How It Works](#how-it-works) · [🧪 Experiments](#experiment-ideas)

</div>

---

## 📌 Project Overview

Built as **Task 03** of the AI Internship at **CodeAlpha** to explore deep learning for creative sequence generation, Bidirectional LSTM networks, and AI-driven music composition.

> *"The model doesn't understand music the way humans do — it learns patterns. By studying enough structure, it generates something surprisingly musical."*
> — Rashid Ahmad

The core insight behind this project: music composition and next-word prediction are the same problem. Notes follow rules, patterns, and tendencies — and a sequence model can learn all of them from data. This system trains on Bach chorales from the `music21` corpus, learns what notes commonly follow others, and generates brand-new piano compositions note by note.

| | Details |
|---|---|
| **Developer** | Rashid Ahmad |
| **Internship** | CodeAlpha — AI Internship Task 03 |
| **Domain** | Deep Learning · Sequential Modeling · Generative AI |
| **Model** | Bidirectional LSTM (TensorFlow / Keras) |
| **Dataset** | Bach Chorales — `music21` built-in corpus |
| **Output** | MIDI + WAV audio files |
| **Interface** | Gradio (browser-based, Colab-compatible) |
| **API Key** | ❌ Not required |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎵 **AI Piano Music Generation** | Generates original note sequences using a trained deep learning model |
| 🧠 **Bidirectional LSTM** | Learns musical patterns from both past and future context in the training sequence |
| 🌡️ **Temperature Sampling** | Controls creativity level — low = conservative, high = experimental |
| 🎼 **MIDI Export** | Downloads generated composition as a standard `.mid` file |
| 🔊 **WAV Audio Playback** | Converts MIDI to 44.1 kHz WAV via FluidSynth for in-browser playback |
| 🎚️ **Adjustable BPM** | Controls the tempo of the generated composition |
| 📊 **Training Visualization** | Loss curves and accuracy metrics displayed during training |
| 🖥️ **Interactive Gradio UI** | Clean browser interface for training, generation, and playback |
| 💾 **Model Checkpointing** | Saves best model weights automatically during training |

---

## 🏗️ Architecture

```
Bach Chorale Corpus (music21)
        │
        ▼
┌─────────────────────────────────────┐
│         Note Tokenisation           │
│                                     │
│  Note   → "C4", "G#3", "B2"        │
│  Chord  → "0.4.7" (normal order)   │
│  Rest   → "R"                       │
│                                     │
│  Vocabulary: ~100–200 unique tokens │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Sliding Window Sequences       │
│                                     │
│  Window size : 100 tokens           │
│  Stride      : 1                    │
│  X → [t₀, t₁, ..., t₉₉]           │
│  y → [t₁₀₀]  (next token)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│               Bidirectional LSTM Model               │
│                                                      │
│  Embedding(vocab_size, 128)                          │
│       ↓                                              │
│  Bidirectional(LSTM(256, return_sequences=True))     │
│       ↓                                              │
│  Dropout(0.3)                                        │
│       ↓                                              │
│  Bidirectional(LSTM(256))                            │
│       ↓                                              │
│  Dropout(0.3)                                        │
│       ↓                                              │
│  BatchNormalization()                                │
│       ↓                                              │
│  Dense(256, activation='relu')                       │
│       ↓                                              │
│  Dropout(0.3)                                        │
│       ↓                                              │
│  Dense(vocab_size, activation='softmax')             │
└──────────────┬──────────────────────────────────────┘
               │  Training: categorical_crossentropy + Adam
               │
               ▼
┌─────────────────────────────────────┐
│       Autoregressive Generation     │
│                                     │
│  Seed sequence (100 tokens)         │
│       ↓                             │
│  Predict next token                 │
│  Apply temperature sampling         │
│  Append → slide window → repeat     │
│       ↓                             │
│  N new tokens generated             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│          Audio Export               │
│                                     │
│  Tokens → music21 Note/Chord/Rest   │
│        → MIDI (.mid)                │
│        → FluidSynth                 │
│        → WAV (.wav) @ 44,100 Hz     │
└─────────────────────────────────────┘
```

---

## 🎛️ How It Works

### Step 1 — Data Loading & Tokenisation

The `music21` corpus provides access to 300+ Bach chorales — no downloading required. Each score is parsed element by element:

```python
for element in parts.parts[0].recurse():
    if isinstance(element, music21.note.Note):
        notes.append(str(element.pitch))        # e.g. "C4"
    elif isinstance(element, music21.chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))  # e.g. "0.4.7"
```

These string tokens form a vocabulary. Each token is mapped to an integer index — exactly like word tokenisation in NLP.

### Step 2 — Sequence Preparation

```python
# Sliding window: 100 input tokens → predict the 101st
for i in range(len(notes) - sequence_length):
    X.append([note_to_int[n] for n in notes[i : i + sequence_length]])
    y.append(note_to_int[notes[i + sequence_length]])
```

### Step 3 — Bidirectional LSTM

A standard LSTM reads a sequence left-to-right. A **Bidirectional LSTM** reads it in both directions simultaneously, giving the model context from both earlier and later in the training sequence — capturing richer musical relationships between notes.

```python
tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(256, return_sequences=True)
)
```

### Step 4 — Temperature Sampling

Temperature controls the "creativity" of generation by scaling the model's output probabilities before sampling:

```python
prediction = np.log(prediction + 1e-7) / temperature
exp_preds  = np.exp(prediction)
prediction = exp_preds / np.sum(exp_preds)
index      = np.argmax(np.random.multinomial(1, prediction, 1))
```

| Temperature | Effect |
|-------------|--------|
| `0.3 – 0.6` | Conservative — repeats learned patterns closely |
| `0.8 – 1.2` | Balanced — musical and moderately varied |
| `1.5 – 2.0` | Experimental — unpredictable, sometimes dissonant |

### Step 5 — MIDI & WAV Export

```python
# Tokens → music21 objects → MIDI
midi_stream = music21.stream.Stream(output_notes)
midi_stream.write('midi', fp=midi_path)

# MIDI → WAV via FluidSynth
subprocess.run([
    "fluidsynth", "-ni", soundfont_path,
    midi_path, "-F", wav_path, "-r", "44100"
])
```

---

## 🗂️ Repository Structure

```
ai-music-generator/
│
├── ai_music_generator.py          # Main Colab script (self-contained)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── src/
│   ├── data_loader.py             # Bach corpus loading & tokenisation
│   ├── model.py                   # Bidirectional LSTM architecture
│   ├── generator.py               # Autoregressive generation + sampling
│   └── exporter.py                # MIDI → WAV audio export
│
├── notebooks/
│   └── MusicGeneration_Colab.ipynb   # Annotated Colab notebook version
│
└── outputs/                       # Generated MIDI/WAV files (gitignored)
```

---

## 🚀 Getting Started

### Option 1 — Google Colab (Recommended)

1. Open the notebook in Colab
2. Run all cells from top to bottom — dependencies install automatically
3. Click **"Train Model"** in the Gradio UI (first run: ~20–40 min on CPU, ~8 min on T4 GPU)
4. Once training completes, click **"Generate Music"**
5. Play the WAV in-browser or download the MIDI

> ⚡ **Tip:** Enable GPU — Runtime → Change Runtime Type → T4 GPU

### Option 2 — Local Setup

**Prerequisites:** Python 3.10+, FluidSynth

```bash
# Install FluidSynth (Ubuntu / Debian)
sudo apt-get install fluidsynth fluid-soundfont-gm

# macOS
brew install fluidsynth

# Clone & install
git clone https://github.com/rashidahmad/ai-music-generator.git
cd ai-music-generator
pip install -r requirements.txt
python ai_music_generator.py
```

---

## 📦 Requirements

```
tensorflow>=2.12
music21>=9.0
gradio>=4.0
pretty_midi>=0.2
numpy>=1.24
```

```bash
pip install -r requirements.txt
```

---

## 🖥️ Gradio Interface

The app launches a browser-based UI with two main controls:

### Training Panel
| Control | Description |
|---------|-------------|
| **Bach Chorales Slider** | Number of training pieces (5–50). More = richer vocabulary, longer training |
| **Train Model Button** | Loads data, builds model, starts training with early stopping |

### Generation Panel
| Control | Description |
|---------|-------------|
| **Number of Notes** | Length of the generated piece (50–300 tokens) |
| **Temperature** | Creativity level (0.1 = safe, 2.0 = chaotic) |
| **Generate Music Button** | Runs autoregressive generation → exports MIDI + WAV |
| **Audio Player** | In-browser WAV playback |
| **Download MIDI** | Export the `.mid` file for use in any DAW |

---

## 🧪 Experiment Ideas

| Experiment | What to Change | Expected Outcome |
|------------|---------------|------------------|
| Strict classical style | `temperature = 0.4` | Closely mirrors Bach's patterns |
| Creative variation | `temperature = 1.4` | Novel combinations, occasional dissonance |
| Short motif | `num_notes = 50` | ~10 seconds — good for ringtones/loops |
| Full composition | `num_notes = 300` | ~60 seconds of continuous piano |
| Larger corpus | `num_chorales = 50` | Bigger vocabulary, more varied output |
| Faster training | Reduce `epochs` to `20` | Less accurate but quicker prototyping |

---

## 📊 What the Model Learns

The Bidirectional LSTM doesn't learn music theory explicitly — it learns **statistical patterns** from the training data:

- Which notes commonly follow a given sequence
- How chord progressions tend to resolve
- The rhythmic regularity of Bach's chorale style
- Which note combinations appear together frequently

This is why the output sounds *musical* even though the model has no concept of harmony, melody, or emotion. Pattern learning at scale produces structure that our ears recognise as coherent music.

---

## ⚠️ Limitations

- Training from scratch on CPU is slow (~40 min for 20 chorales, 40 epochs) — GPU strongly recommended
- The model only generates notes at a fixed time offset (`0.5` per step) — rhythm variation requires additional modeling
- Output quality depends heavily on training duration and corpus size
- Very low temperature values can cause repetitive loops; very high values produce incoherent sequences

---

## 🗺️ Roadmap

- [ ] Rhythm modeling (variable note durations)
- [ ] Multi-instrument generation (piano + violin)
- [ ] Transformer-based architecture for longer context
- [ ] Conditional generation (specify mood or key)
- [ ] Real-time MIDI keyboard input as seed

---

## 📚 References

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
- Schuster, M. & Paliwal, K. (1997). *Bidirectional Recurrent Neural Networks.* IEEE Transactions.
- Cuthbert, M. & Ariza, C. (2010). *music21: A Toolkit for Computer-Aided Musicology.* ISMIR.
- Boulanger-Lewandowski et al. (2012). *Modeling Temporal Dependencies in High-Dimensional Sequences.* ICML.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- **CodeAlpha** for the AI Internship structure and task brief
- The `music21` team at MIT for open-access corpus data
- The TensorFlow / Keras team for the deep learning framework

---

<div align="center">

**Developed by Rashid Ahmad · CodeAlpha AI Internship · Task 03**

*"AI creativity often comes from pattern learning — and by learning enough structure, it can still generate something surprisingly musical."*

</div>
