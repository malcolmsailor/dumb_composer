# Rhythm prefabs

- `total_dur: TIME_TYPE`
- `onsets: t.List[TimeStamp]`
- `metric_strength_str: str`
- `releases: t.Optional[t.Sequence[TimeStamp]] = None`: if omitted, generated automatically

- `endpoint_metric_strength_str: str = "ss"`: indicates the metric strengths of the onset of the rhythm and the onset of the start of the next rhythm. E.g., if it goes from beat 1 to beat 1, should be "ss"; if it goes from beat 2 to beat 1, should be "ws"
- `allow_suspension: t.Optional[Allow] = None`: if omitted, set to True if onsets begins with 0.0, otherwise False
- `allow_preparation: Allow = Allow.NO`
- `allow_resolution: t.Optional[Allow] = None`
- `allow_after_tie: t.Optional[Allow] = None- `:  if omitted, set to True if onsets begins with 0.0, otherwise False
- `allow_next_to_start_with_rest: t.Optional[Allow] = None`: if omitted, set according to the following heuristic:
    - if the rhythm ends with a rest, set to Allow.YES
    - else if last onset is <= 1/4 the length of the rhythm, or less
        than MIN_DEFAULT_RHYTHM_BEFORE_REST set to allow.NO
    - else, set to allow.YES

```yaml
- total_dur: 4
  onsets: [0.0, 1.5, 2.0, 2.5, 3.0]
  metric_strength_str: "swsws"
```
# Pitch prefabs

Required keys:
- `interval_to_next: int | Sequence[int] | None`
- `metric_strength_str: str`
- `relative_degrees: Sequence[Union[int, str]]`

Optional keys (default value indicated):
- `constraints: Sequence[int] = ()`: specifies intervals relative to the initial pitch that must be contained in the chord. Thus if constraints=[-2], the prefab could occur starting on the 3rd or fifth of a triad, but not on the root. If constraints = [-2, -4], it could occur on only the fifth of a triad. Etc.
- `negative_constraints: Sequence[int] = ()`: specifies intervals relative to the initial pitch that must NOT be contained in the chord.
- `allow_suspension: Literal["YES", "NO", "ONLY"] = "YES"`
- `allow_preparation: Literal["YES", "NO", "ONLY"] = "NO"`
- `avoid_interval_to_next: Sequence[int] = ()`
- `avoid_voices: Container[Voice] = frozenset()`
