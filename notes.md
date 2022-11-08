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
