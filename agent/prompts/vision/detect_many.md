Detect up to 12 instances of the <target_object> from the provided info about it - the ones CLOSEST to the centre of the image. Each box_2d is four coordinates normalized to 0-1000, in this model's own native box_2d coordinate order. Return a JSON array with one entry per instance and nothing else; return [] only when no matching target is genuinely visible. Never return masks or extra code fencing. Example output:

```json
[{'box_2d': box_2d, 'label': target_object}, {'box_2d': box_2d, 'label': target_object}]
```

