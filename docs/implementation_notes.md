# Implementation notes

Details of how this implementation realizes the method described in the paper, the design
choices made along the way, and the improvements planned for future versions. Default
parameters reproduce the results of the original research code.

## Registration

- **Window aggregation.** Scans of a window are aggregated in the session frame given by the
  odometry poses. This is equivalent to aggregating in the frame of the window's first scan
  (paper eq. 1) up to a fixed transform.
- **Verification radius.** Before any window is trusted, there is no prior to track, so the
  first window's fitness is evaluated with a wide inlier radius (`first_eval_distance`,
  50 m); later windows use `eval_distance` (2 m). A tighter first check is planned.
- **Pose fusion.** Scans covered by trusted windows average their estimates. Remaining scans
  use an inverse-squared-distance average of the trusted transforms, with distances measured
  between scan centroids.
- **Window overlap.** The default `stride` equals `window_size`, so each scan belongs to one
  window. A smaller stride gives overlapping windows and smoother transitions between them.
- **Determinism.** KISS-Matcher is not deterministic; aligned poses can differ by a few
  centimeters between runs.

## Dynamic object removal

- **Ray origin.** Exposure is computed with the median of each scan's points as the ray
  origin, which approximates the sensor position when the surroundings are roughly symmetric.
  Using the odometry pose directly is planned.
- **Stability score.** The score is computed as R = H / max(H, E), where the exposure count E
  also counts hits; this corresponds to H / (H + E + η) in the paper with E counting only
  traversals. Hits add the number of points in a voxel per scan, exposure adds one per scan.
- **Ground.** The local ground height comes from 5 m anchor cells (k-nearest anchors), with an
  asymmetric band of +0.1 m / −1.0 m around it.
- **Clustering.** Candidate regions are formed by 6-connected label propagation on voxels up
  to 4.5 m above the ground, with labels capped at 100 voxels so that large static
  structures are not merged with small objects.
- **Additional refinements.** Besides the stages described in the paper, the classifier
  removes small floating fragments, propagates dynamic labels down ground columns, and
  smooths weakly observed voxels by majority vote. Each stage can be switched off; see
  [parameters.md](parameters.md).

## Map integration

- **Stored statistics.** Each Voxel Ratio Map stores every observed voxel, including voxels
  that were only traversed by rays. This keeps the free-space evidence of a new session, so
  objects that left the scene can be removed from the lifelong map.
- **Update rule.** Hit and exposure counts are currently summed across sessions. The
  density-adaptive decay of older evidence (paper eq. 8–9) is planned.
- **Lifelong points.** The previous lifelong map and the new session's points are classified
  together with the merged statistics. Points accumulate across sessions; voxel-level
  deduplication is planned for long session sequences.
