A project to generate synthetic Classical music annotated with chord changes.

The approach is purely rule-based. My goal here was to create augmented data for training Roman-numeral analysis models like the one in my [2024 ISMIR paper](TODO). 

The general intuition behind the project is that the composition of music is a little bit like a one-way function. In other words, it is relatively easy to write rules to go from a chord progression to a musical surface, but harder to write rules to go from the musical surface to the chord progression.

While it was a fun project and the generated music can be amusing to listen to, I didn't find that it improved the test performance of my trained models, presumably because the synthetic data lies too far out of distribution. It's possible that refining the code to produce better music would render it useful, but for now I do not plan to pursue this research direction further.

There is further discussion of this project in my upcoming dissertation.

# Usage

There are four main scripts:

```
scripts/run_incremental_contrapuntist.py
scripts/run_incremental_contrapuntist_with_accomps.py
scripts/run_incremental_contrapuntist_with_prefabs.py
scripts/run_incremental_contrapuntist_with_prefabs_and_accomps.py
```

Each of these generate music in batch, taking a chord progression or progressions as inputs. They each follow a slightly different procedure:

- `run_incremental_contrapuntist.py`: makes a four-part chorale-style realization. The realization may contain suspensions but is otherwise unornamented. [Listen to an example (score below)](docs/mozart_structural_0000192.mp3).

![Example of the output of `run_incremental_contrapuntist.py`](docs/mozart_structural_0000192.jpg)

- ``
