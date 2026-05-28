# Arnold Weak Resonance Ornament Explorer

Single-file Streamlit prototype for exploring Greek ornamental design phase portraits based on Arnold's weak resonance equation and the Berezovskaya–Karev ornament paper.

## Run locally

```bash
cd ornaments_app
pip install -r requirements.txt
streamlit run app.py
```

## Current status

This is a first functional Python/Streamlit prototype. It ports the polar-coordinate model from the MATLAB scripts and includes:

- paper preset selector
- custom parameter explorer
- one-parameter sweep/zoom workflow
- dense render/export page
- forward + backward integration toggle
- sector-symmetry toggle
- equilibrium markers
- PNG and parameter JSON export

Outputs should be treated as qualitative until selected presets are visually checked against the MATLAB figures.

## Model

The app integrates

```text
dr/dt   = eps1*r + sum_k A2[k]*r^(2k+1) + B*r^(n-1)*cos(n*phi)
dphi/dt = eps2   + sum_k A2[k]*r^(2k)   - B*r^(n-2)*sin(n*phi)
```

where `s = floor(n/2) - 1` determines the expected number of A2 terms.
