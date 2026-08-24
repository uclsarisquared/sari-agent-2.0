Detect the <target_object> from the provided info about it. The box_2d should be four coordinates normalized to 0-1000, in this model's own native box_2d coordinate order. Return one JSON object with a label. Target absence is a normal result, not a refusal: if the target is genuinely not visible, return exactly [] with no apology or explanation. Never return masks or code fencing. Limit to one object only. Example output:

```json
{'box_2d': box_2d, 'label': target_object}
```
