# Important TODOs

- suspensions in Bass
- when "prefab voice" is bass, supposed suspensions are not realized in upper parts
- suspensions in accompaniment would be good
- if there is a tie in the "melody", avoid accompaniments that have no downbeat

Because (tied) suspensions are hard to learn, we want to tie as many suspensions as possible.
(It would also be good to allow for tied "anticipations".)

Chord lengths that should be allowed in

- 2/4
  - from downbeat: 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, ...
    - or: demisemibeat, semibeat, demisemibeat * 3, beat, 3*semibeat, 2 _
      bar, 3 _ beat, 2 _ bar, 3 _ bar, 4 \* bar, ...
- 4/4
  - from downbeat: 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, ...
    - or: demisemibeat, semibeat, demisemibeat _ 3, beat, 3 _ semibeat,
      superbeat, 3 _ beat, bar, 3 _ superbeat, 2 _ bar, 3 _ bar, ...
- 3/4
  - from downbeat: 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5? 6, 9, 12, 15 ...
    - or: demisemibeat, semibeat, demisemibeat _ 3, beat, semibeat _ 3, 2 _
      beat (superbeat?), 1 _ bar, 3 _ superbeat??, 2 _ bar, 3 \* bar, ...
  - from beat 2 (max_len 2---will be trimmed at barline): 0.25, 0.5, 0.75,
    1.0, 1.5, 2
    - or: demisemibeat, semibeat, demisemibeat _ 3, beat, semibeat _ 3, 2 \*
      beat (superbeat?)
- 6/8
  - from downbeat: 0.25, 0.5, 0.75, 1.0, 1.5, 2.5?, 3.0, 4.5, 6, 9, 12, ...
    - or: demisemibeat, semibeat, demisemibeat _ 3, semibeat _ 2, beat, ???,
      bar, beat _ 2, bar _ 2, etc.
  - from beat 2=1.5 (max_len 1.5---will be trimmed at barline): 0.25, 0.5, 0.75,
    1.0, 1.5
    - or: demisemibeat, semibeat, demisemibeat _ 3, semibeat _ 2, beat
- 9/8
  - from downbeat: 0.25, 0.5, 0.75, 1.0, 1.5, 2.5?, 3.0, 4.5, 7.5?, 9, 13.5,
    ...
    - or: demisemibeat, semibeat, demisemibeat _ 3, semibeat _ 2, beat, ???,
      beat _ 2, bar, ???, bar _ 2, ...
  - from beat 2 (max_len 3---will be trimmed at barline)): 0.25, 0.5, 0.75,
    1.0, 1.5, 2.5?, 3.0
    - or: demisemibeat, semibeat, demisemibeat _ 3, semibeat _ 2, beat, ???,
      beat \* 2
  - from beat 3 (max_len 1.5---will be trimmed at barline): 0.25, 0.5, 0.75,
    1.0, 1.5
    - or: demisemibeat, semibeat, demisemibeat _ 3, semibeat _ 2, beat
- 12/8
  - from downbeat: 0.25, 0.5, 0.75, 1.0, 1.5, 2.5?, 3.0, 4.5, 6.0, 9.0, 12,
    18, ...
    - or: demisemibeat, semibeat, demisemibeat _ 3, semibeat _ 2, beat, ???,
      superbeat, beat _ 3, bar, superbeat _ 3, bar \* 2, ...

if there are more than two prefab voices, raise an error

if "soprano" is among prefab voices, accompaniment is below it
if "bass" is among prefab voices, accompaniment is above it

S -> accompaniment below
A -> accompaniment below
T -> accompaniment above [or above and below TODO]
B -> accompaniment above

SA -> accompaniment below alto
ST -> accompaniment below tenor [or between tenor and soprano with bass TODO]
SB -> accompaniment between voices
AT -> accompaniment below tenor [or above alto with bass TODO]
AB -> accompaniment above alto [or divided between?]
TB -> accompaniment above tenor

It would be nice to allow this to change on a per-measure basis.

===

2023-08-03

Accompaniment procedure:

1. choose pattern
2. set appropriate ranges
   - it seems best to set ranges dynamically. E.g., for alberti bass, set a certain range for the bass, then allow the rest to be within an octave above that, with optional minimums/maximums
3. create 4-part structural counterpoint
4. double parts as required by structure
5. realize melody
6. realize accompaniment

Alberti:

- 4 "real" parts:
  - melody
  - 3 in accompaniment in close position
- only w/ melody in top voice? or bottom voice too?

1-3-5 patterns:

- same part constraints as alberti
  - seems like 5-3-1 or similar might be better in case of bass melody?

Solid chords:

- 4 "real" parts
- optional doublings

Oom-pah etc:

- 4 real parts
- optional doublings

Arpeggios:

- need to think more

Tremolos:

- independent of bass/melody:
  - requires 2 different pitches (we need to enforce no unisons)
- including bass/melody
  - could be in 3 real parts
  - or 4 real parts with possible doublings, with multiple notes in one or the other leg of the tremolo

# suspensions

Suspensions to avoid:

General:

- 6/4 to 6/3

Bass:
7-8

Inner voices:
6-5 unless there is a 4-3 or something elsewhere
4-3 when 3 is already in melody or other voice
Maybe:

- certain pitches on diminished 7th chords
- 9-8 on a change of harmony?

```
foobar test
```

```python
def foobar():
    return
```
