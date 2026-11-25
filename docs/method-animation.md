# ELP method animation

A 16-second, 1200 × 720 looping GIF introduces the method in four steps. It is a schematic explanation of the implementation, not benchmark evidence. No model inference, weights, external API calls, or experiment logs are used to render it.

An illustrated example of program proposal, execution, feedback, and frozen replay.

## Regenerate

Run from the repository root with Python 3.10 or newer:

```bash
python -m pip install "Pillow>=10"
python docs/render_method_intro.py
```

The script uses Arial/Menlo on macOS or DejaVu Sans/Mono on Linux. Install DejaVu fonts if neither is available. It writes `docs/assets/method-intro.gif` and `docs/assets/method-overview.png`.

The still image provides an alternative to the animation. Edit the script’s step labels, captions, and drawing function to update the explanation.

## Implementation references

- [elp/core/program.py](../elp/core/program.py)
- [elp/agent/coordinator.py](../elp/agent/coordinator.py)
- [elp/core/pareto.py](../elp/core/pareto.py)

## Visual reference

The compact, progressively revealed animation format was inspired by [Noise’s README animation](https://github.com/qlabs-eng/noise). All diagrams and frames here are drawn from code for this project; no Noise assets or curves are reused.
