# Original no-wall speed and transition bank reconstruction

Run `python3 run.py rebuild-reproduce --smoke`, then `python3 run.py rebuild-reproduce`.
The registered full pipeline retrains speed, regenerates the offline bank, checks
numeric equality against frozen inputs, trains TravelCost and evaluates it.
All outputs are new; frozen baseline assets are never overwritten.

Source inputs: original and augmentation training pose pairs and physical labels,
verified by source_manifest.json. No evaluation poses or trajectories are read by
preparation. Extract dist/no-wall-training-inputs.tar.gz at the repository root
on another machine. These original pose pools remain the declared source dataset;
this workflow does not recreate the historical pose-pool sampling process.

Speed: random initialization, seed87358, 20000 AdamW updates, 512 pairs from each
source per update, pair index modulo100==0 held out, log-speed MSE, lr0.0003,
weight_decay1e-6, gradient clipping2. Preserve original labels for exact historical
training. Every label is also regenerated from physical geometry and audited with
maximum absolute error <=2e-5; regenerated labels are saved separately. Physical
labels are numerically reproducible, not necessarily bitwise identical.

Bank: select100000safe training endpoints using the speed-training NumPy RNG
stream, then seed87344, 50000walkers, 8generations, 4paired directions and step
sizes0.01/0.05/0.15/0.25, 17quadrature points. Reject unsafe edges at0.02rad,
20degree OR, table and self-collision, no wall/workspace bound. Minimum speed0.015.
Choose a safe scale then direction uniformly. No goal/path/value-based selection.

The full run stops before TravelCost training if any speed parameter or bank array
differs from the frozen baseline. Metadata and serialized checkpoint hashes can
change even with identical tensor values; compare tensors and NumPy arrays.
A different GPU/environment has not been certified.
