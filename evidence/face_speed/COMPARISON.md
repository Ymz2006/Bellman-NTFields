# Cabinet task: per-piece (box-face) speed vs the release speed

Generated 2026-10-25 18:17 UTC. Frozen suites `experiments/cabinet/tests` (6 x 100 certified queries; MPPI rows count 3 planning seeds). All runs below were made on this workstation through the host backend (bwrap, not the pinned image): {"python": "3.12.3", "torch": "2.7.1+cu128", "cuda": "12.8", "cudnn": 90701, "gpus": ["NVIDIA GeForce RTX 3090"], "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.39", "curobo": "unavailable: No module named 'curobo'", "numpy": "2.2.6"}. Rows marked *pinned* are the original machine's results in the pinned image (evidence/cabinet), for reference only.

## Success rate, all 600 queries

|Run|Method|Speed cover|RK4 (native)|Release MPPI|Release MPPI per seed|ntrl-demo MPPI|
|---|---|---|---:|---:|---|---:|
|`registered_20261025T065945Z-3c53b360`|metric|release|181/600 (30.2%)|613/1800 (34.1%)|[204, 204, 205]|520/1800 (28.9%)|
|`registered_20261025T065830Z-8e5788c0`|metric|faces|173/600 (28.8%)|618/1800 (34.3%)|[206, 206, 206]|520/1800 (28.9%)|
|`registered_20261025T072528Z-2e175159`|metric|faces|184/600 (30.7%)|626/1800 (34.8%)|[211, 206, 209]|502/1800 (27.9%)|
|`registered_20261025T070429Z-b9a2383c`|bellman|release|228/600 (38.0%)|682/1800 (37.9%)|[227, 227, 228]|–|
|`registered_20261025T070501Z-aaa42561`|bellman|faces|248/600 (41.3%)|722/1800 (40.1%)|[238, 242, 242]|–|
|`registered_20261025T072358Z-932f514d`|bellman|spheres|231/600 (38.5%)|684/1800 (38.0%)|[229, 229, 226]|–|
|`registered_20261025T074323Z-331f4da7`|bellman|links|231/600 (38.5%)|681/1800 (37.8%)|[228, 227, 226]|–|
|`registered_20261025T175901Z-4ef50717`|bellman|split (separate models)|227/600 (37.8%)|669/1800 (37.2%)|[224, 223, 222]|–|
|`registered_20261025T175946Z-fe060952`|bellman|split|227/600 (37.8%)|690/1800 (38.3%)|[230, 229, 231]|–|
|`registered_20261019T041026Z-39d9dafe` *pinned*|bellman|release|229/600 (38.2%)|688/1800 (38.2%)|[230, 229, 229]|–|
|`registered_20261019T041045Z-9baa751f` *pinned*|metric|release|182/600 (30.3%)|595/1800 (33.1%)|[197, 198, 200]|509/1800 (28.3%)|

## Per suite, RK4 (native)

|Suite|metric release|metric faces|metric faces|bellman release|bellman faces|bellman spheres|bellman links|bellman split (separate models)|bellman split|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|free_to_free|79/100 (79.0%)|72/100 (72.0%)|79/100 (79.0%)|90/100 (90.0%)|92/100 (92.0%)|89/100 (89.0%)|91/100 (91.0%)|89/100 (89.0%)|88/100 (88.0%)|
|free_to_compartment|8/100 (8.0%)|13/100 (13.0%)|8/100 (8.0%)|22/100 (22.0%)|23/100 (23.0%)|27/100 (27.0%)|24/100 (24.0%)|20/100 (20.0%)|23/100 (23.0%)|
|compartment_to_free|9/100 (9.0%)|7/100 (7.0%)|10/100 (10.0%)|20/100 (20.0%)|30/100 (30.0%)|20/100 (20.0%)|21/100 (21.0%)|21/100 (21.0%)|20/100 (20.0%)|
|compartment_to_compartment|3/100 (3.0%)|1/100 (1.0%)|1/100 (1.0%)|2/100 (2.0%)|4/100 (4.0%)|2/100 (2.0%)|0/100 (0.0%)|2/100 (2.0%)|2/100 (2.0%)|
|same_compartment|6/100 (6.0%)|6/100 (6.0%)|11/100 (11.0%)|7/100 (7.0%)|7/100 (7.0%)|7/100 (7.0%)|8/100 (8.0%)|7/100 (7.0%)|6/100 (6.0%)|
|anywhere|76/100 (76.0%)|74/100 (74.0%)|75/100 (75.0%)|87/100 (87.0%)|92/100 (92.0%)|86/100 (86.0%)|87/100 (87.0%)|88/100 (88.0%)|88/100 (88.0%)|

## Per suite, release MPPI

|Suite|metric release|metric faces|metric faces|bellman release|bellman faces|bellman spheres|bellman links|bellman split (separate models)|bellman split|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|free_to_free|262/300 (87.3%)|253/300 (84.3%)|263/300 (87.7%)|268/300 (89.3%)|279/300 (93.0%)|266/300 (88.7%)|273/300 (91.0%)|266/300 (88.7%)|267/300 (89.0%)|
|free_to_compartment|53/300 (17.7%)|63/300 (21.0%)|63/300 (21.0%)|67/300 (22.3%)|71/300 (23.7%)|68/300 (22.7%)|65/300 (21.7%)|63/300 (21.0%)|69/300 (23.0%)|
|compartment_to_free|22/300 (7.3%)|36/300 (12.0%)|29/300 (9.7%)|64/300 (21.3%)|83/300 (27.7%)|57/300 (19.0%)|55/300 (18.3%)|57/300 (19.0%)|59/300 (19.7%)|
|compartment_to_compartment|4/300 (1.3%)|3/300 (1.0%)|3/300 (1.0%)|3/300 (1.0%)|4/300 (1.3%)|6/300 (2.0%)|2/300 (0.7%)|5/300 (1.7%)|7/300 (2.3%)|
|same_compartment|33/300 (11.0%)|34/300 (11.3%)|40/300 (13.3%)|22/300 (7.3%)|12/300 (4.0%)|27/300 (9.0%)|25/300 (8.3%)|25/300 (8.3%)|24/300 (8.0%)|
|anywhere|239/300 (79.7%)|229/300 (76.3%)|228/300 (76.0%)|258/300 (86.0%)|273/300 (91.0%)|260/300 (86.7%)|261/300 (87.0%)|253/300 (84.3%)|264/300 (88.0%)|

## Per suite, ntrl-demo MPPI

|Suite|metric release|metric faces|metric faces|
|---|---:|---:|---:|
|free_to_free|238/300 (79.3%)|220/300 (73.3%)|226/300 (75.3%)|
|free_to_compartment|20/300 (6.7%)|35/300 (11.7%)|25/300 (8.3%)|
|compartment_to_free|24/300 (8.0%)|29/300 (9.7%)|27/300 (9.0%)|
|compartment_to_compartment|2/300 (0.7%)|1/300 (0.3%)|0/300 (0.0%)|
|same_compartment|11/300 (3.7%)|8/300 (2.7%)|8/300 (2.7%)|
|anywhere|225/300 (75.0%)|227/300 (75.7%)|216/300 (72.0%)|

## Same planner, same queries: variant vs its same-environment release baseline

|Method|Cover|Planner|Both solve|Variant only|Baseline only|Neither|
|---|---|---|---:|---:|---:|---:|
|metric|faces|RK4 (native)|152|21|29|398|
|metric|faces|release MPPI|564|54|49|1133|
|metric|faces|ntrl-demo MPPI|442|78|78|1202|
|metric|faces|RK4 (native)|151|33|30|386|
|metric|faces|release MPPI|548|78|65|1109|
|metric|faces|ntrl-demo MPPI|434|68|86|1212|
|bellman|faces|RK4 (native)|202|46|26|326|
|bellman|faces|release MPPI|614|108|68|1010|
|bellman|spheres|RK4 (native)|213|18|15|354|
|bellman|spheres|release MPPI|644|40|38|1078|
|bellman|links|RK4 (native)|211|20|17|352|
|bellman|links|release MPPI|631|50|51|1068|
|bellman|split (separate models)|RK4 (native)|213|14|15|358|
|bellman|split (separate models)|release MPPI|644|25|38|1093|
|bellman|split|RK4 (native)|215|12|13|360|
|bellman|split|release MPPI|654|36|28|1082|

## Failure reasons (scorer)

|Run|Planner|collision|endpoint|joint_limits|
|---|---|---:|---:|---:|
|metric release|RK4 (native)|409|18|0|
|metric release|release MPPI|1187|0|0|
|metric release|ntrl-demo MPPI|1272|0|64|
|metric faces|RK4 (native)|409|35|0|
|metric faces|release MPPI|1182|0|0|
|metric faces|ntrl-demo MPPI|1253|0|80|
|metric faces|RK4 (native)|406|24|0|
|metric faces|release MPPI|1174|0|0|
|metric faces|ntrl-demo MPPI|1289|0|79|
|bellman release|RK4 (native)|368|5|0|
|bellman release|release MPPI|1118|0|0|
|bellman faces|RK4 (native)|343|11|0|
|bellman faces|release MPPI|1078|0|0|
|bellman spheres|RK4 (native)|367|4|0|
|bellman spheres|release MPPI|1116|0|0|
|bellman links|RK4 (native)|366|4|0|
|bellman links|release MPPI|1119|0|0|
|bellman split (separate models)|RK4 (native)|371|3|0|
|bellman split (separate models)|release MPPI|1131|0|0|
|bellman split|RK4 (native)|371|4|0|
|bellman split|release MPPI|1110|0|0|

## Training inputs

- metric release (`registered_20261025T065945Z-3c53b360`): 500000 pairs from 46600000 uniform draws; x0 in the cupboard 0.83%; x1 at the speed floor 35.1%; final epoch loss 0.9954; data 32 s.
- metric faces (`registered_20261025T065830Z-8e5788c0`): 500000 pairs from 51600000 uniform draws; x0 in the cupboard 0.04%; x1 at the speed floor 41.0%; mean speed 0.566 vs release 0.599, 11.1% of endpoints below the release speed; final epoch loss 1.013; data 47 s.
- metric faces (`registered_20261025T072528Z-2e175159`): 500000 pairs from 43400000 uniform draws; x0 in the cupboard 0.90%; x1 at the speed floor 43.5%; mean speed 0.513 vs release 0.573, 18.5% of endpoints below the release speed; final epoch loss 0.9664; data 43 s.
- bellman release (`registered_20261025T070429Z-b9a2383c`): 2 outputs, 20000 steps; held-out original: outputs MAE 0.0355, min speed MAE 0.0355 (vs release 0.0355); augmentation: outputs MAE 0.0508, min speed MAE 0.0508 (vs release 0.0508); bank anchors 399581; bank-source speed MAE vs release 0.0964, vs cover truth 0.0964.
- bellman faces (`registered_20261025T070501Z-aaa42561`): 31 outputs, 20000 steps; held-out original: outputs MAE 0.0064, min speed MAE 0.0310 (vs release 0.0970); augmentation: outputs MAE 0.0103, min speed MAE 0.0244 (vs release 0.1839); bank anchors 399581; bank-source speed MAE vs release 0.1419, vs cover truth 0.0916.
- bellman spheres (`registered_20261025T072358Z-932f514d`): 52 outputs, 20000 steps; held-out original: outputs MAE 0.0032, min speed MAE 0.0404 (vs release 0.0404); augmentation: outputs MAE 0.0113, min speed MAE 0.0465 (vs release 0.0465); bank anchors 399581; bank-source speed MAE vs release 0.0924, vs cover truth 0.0924.
- bellman links (`registered_20261025T074323Z-331f4da7`): 18 outputs, 20000 steps; held-out original: outputs MAE 0.0049, min speed MAE 0.0341 (vs release 0.0341); augmentation: outputs MAE 0.0142, min speed MAE 0.0396 (vs release 0.0396); bank anchors 399581; bank-source speed MAE vs release 0.0839, vs cover truth 0.0839.
- bellman split (separate models) (`registered_20261025T175901Z-4ef50717`): 2 outputs, 20000 steps; held-out original: outputs MAE 0.0215, min speed MAE 0.0271 (vs release 0.0271); augmentation: outputs MAE 0.0294, min speed MAE 0.0453 (vs release 0.0453); bank anchors 399581; bank-source speed MAE vs release 0.0596, vs cover truth 0.0596.
- bellman split (`registered_20261025T175946Z-fe060952`): 2 outputs, 20000 steps; held-out original: outputs MAE 0.0242, min speed MAE 0.0325 (vs release 0.0325); augmentation: outputs MAE 0.0318, min speed MAE 0.0475 (vs release 0.0475); bank anchors 399581; bank-source speed MAE vs release 0.0751, vs cover truth 0.0751.

## Wall time (seconds)

- metric release [device=1; NVIDIA GeForce RTX 3090]: training 661, evaluate 492; total 1153
- metric faces [device=0; NVIDIA GeForce RTX 5070]: data 48, training 502, evaluate 643; total 1193
- metric faces [device=0; NVIDIA GeForce RTX 5070]: data 44, training 418, evaluate 207; total 670
- bellman release [device=1; NVIDIA GeForce RTX 3090]: prepare 504, training 249, evaluate 206; total 958
- bellman faces [device=0; NVIDIA GeForce RTX 5070]: prepare 694, training 193, evaluate 161; total 1048
- bellman spheres [device=1; NVIDIA GeForce RTX 3090]: prepare 355, training 158, evaluate 135; total 647
- bellman links [device=0; NVIDIA GeForce RTX 5070]: prepare 623, training 194, evaluate 209; total 1025
- bellman split (separate models) [device=1; NVIDIA GeForce RTX 3090]: prepare 543, training 182, evaluate 143; total 869
- bellman split [device=0; NVIDIA GeForce RTX 5070]: prepare 676, training 196, evaluate 193; total 1066

## Speed network accuracy (`registered_20261025T073447Z-a5d29435`)

MAE of the predicted body speed against the release clearance speed (the scorer's rule); for per-piece networks also against their own cover label. Full tables: the run's accuracy/REPORT.md.

|Network|Cover|uniform|heldout_original|heldout_augmentation|witness_inside_cupboard|witness_outside_cupboard|uniform_valid|
|---|---|---:|---:|---:|---:|---:|---:|
|`registered_20261025T070429Z-b9a2383c`|release|0.0900|0.0355|0.0508|0.0849|0.0968|0.1652|
|`registered_20261025T070501Z-aaa42561`|faces|0.0966 / 0.0813|0.0970 / 0.0310|0.1839 / 0.0244|0.1654 / 0.0032|0.1516 / 0.0801|0.1857 / 0.1546|
|`registered_20261025T072358Z-932f514d`|spheres|0.0792 / 0.0792|0.0404 / 0.0404|0.0465 / 0.0465|0.0824 / 0.0824|0.0889 / 0.0889|0.1436 / 0.1436|
|`registered_20261019T034213Z-f9c501f9`|release|0.0906|0.0358|0.0510|0.0852|0.0976|0.1664|

Cells: MAE vs release / MAE vs own cover. Sets: uniform n=20000, release speed at floor 51%; heldout_original n=20000, release speed at floor 8%; heldout_augmentation n=20002, release speed at floor 20%; witness_inside_cupboard n=37357, release speed at floor 1%; witness_outside_cupboard n=140539, release speed at floor 6%; uniform_valid n=20000, release speed at floor 6%.

## Step filter at planning time (cspace_face_filter Section 3)

The trained field is unchanged; each planner step is projected onto the half-spaces of the active pieces (`guard.py`). RK4: the native RK4 rollout; MPPI: the ntrl-demo MPPI settings for either field (the release MPPI planner is frozen and not filtered). "converged" counts rollouts that reached the goal; the scorer decides success.

|Field (run)|Filter cover|Rows|alpha|floor|RK4 success|RK4 converged|MPPI success (3 seeds)|MPPI converged|Filter active|Mean step change|Halvings|Stalled steps|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|bellman faces (`registered_20261025T070501Z-aaa42561`)|–|none|0.5|0.15|248/600 (41.3%)|589|710/1800 (39.4%)|1789|–|–|–|–|
|bellman faces (`registered_20261025T070501Z-aaa42561`)|faces|pieces|0.5|0.15|263/600 (43.8%)|288|1064/1800 (59.1%)|990|96%|0.548|160906|93996|
|bellman faces (`registered_20261025T070501Z-aaa42561`)|spheres|pieces|0.5|0.15|375/600 (62.5%)|406|1217/1800 (67.6%)|1333|94%|0.449|77545|63059|
|bellman release (`registered_20261025T070429Z-b9a2383c`)|spheres|body|0.5|0.15|311/600 (51.8%)|487|992/1800 (55.1%)|1574|95%|0.461|200712|19027|
|bellman release (`registered_20261025T070429Z-b9a2383c`)|spheres|body|0.5|0.15|311/600 (51.8%)|486|991/1800 (55.1%)|1568|95%|0.462|202295|19030|
|bellman release (`registered_20261025T070429Z-b9a2383c`)|–|none|0.5|0.15|228/600 (38.0%)|595|641/1800 (35.6%)|1788|–|–|–|–|
|bellman release (`registered_20261025T070429Z-b9a2383c`)|faces|pieces|0.5|0.15|267/600 (44.5%)|288|1028/1800 (57.1%)|966|97%|0.627|282345|77204|
|bellman release (`registered_20261025T070429Z-b9a2383c`)|faces|pieces|0.5|0.15|267/600 (44.5%)|288|1030/1800 (57.2%)|965|97%|0.625|282230|76994|
|bellman release (`registered_20261025T070429Z-b9a2383c`)|spheres|pieces|0.5|0.1|337/600 (56.2%)|384|1095/1800 (60.8%)|1255|96%|0.524|158113|60211|
|bellman release (`registered_20261025T070429Z-b9a2383c`)|spheres|pieces|0.5|0.15|354/600 (59.0%)|382|1147/1800 (63.7%)|1247|96%|0.537|168605|61172|
|metric release (`registered_20261025T065945Z-3c53b360`)|spheres|body|0.5|0.15|282/600 (47.0%)|463|919/1800 (51.1%)|1500|93%|0.526|262105|20690|
|metric release (`registered_20261025T065945Z-3c53b360`)|–|none|0.5|0.15|181/600 (30.2%)|582|527/1800 (29.3%)|1750|–|–|–|–|
|metric release (`registered_20261025T065945Z-3c53b360`)|faces|pieces|0.5|0.15|243/600 (40.5%)|272|970/1800 (53.9%)|904|96%|0.659|338648|73969|
|metric release (`registered_20261025T065945Z-3c53b360`)|faces|pieces|0.5|0.15|243/600 (40.5%)|272|957/1800 (53.2%)|903|96%|0.661|341196|74007|
|metric release (`registered_20261025T065945Z-3c53b360`)|spheres|pieces|0.5|0.1|306/600 (51.0%)|369|998/1800 (55.4%)|1163|95%|0.585|225471|57655|
|metric release (`registered_20261025T065945Z-3c53b360`)|spheres|pieces|0.5|0.15|317/600 (52.8%)|362|1050/1800 (58.3%)|1154|95%|0.601|237215|57838|

|Field|Filter|RK4 per suite|MPPI per suite|
|---|---|---|---|
|bellman faces|none  a0.5 f0.15|free_to_free 92/100, free_to_compartment 23/100, compartment_to_free 30/100, compartment_to_compartment 4/100, same_compartment 7/100, anywhere 92/100|free_to_free 284/300, free_to_compartment 59/300, compartment_to_free 75/300, compartment_to_compartment 8/300, same_compartment 11/300, anywhere 273/300|
|bellman faces|pieces faces a0.5 f0.15|free_to_free 90/100, free_to_compartment 5/100, compartment_to_free 67/100, compartment_to_compartment 2/100, same_compartment 8/100, anywhere 91/100|free_to_free 293/300, free_to_compartment 94/300, compartment_to_free 224/300, compartment_to_compartment 65/300, same_compartment 96/300, anywhere 292/300|
|bellman faces|pieces spheres a0.5 f0.15|free_to_free 92/100, free_to_compartment 39/100, compartment_to_free 80/100, compartment_to_compartment 33/100, same_compartment 38/100, anywhere 93/100|free_to_free 294/300, free_to_compartment 136/300, compartment_to_free 253/300, compartment_to_compartment 107/300, same_compartment 135/300, anywhere 292/300|
|bellman release|body spheres a0.5 f0.15|free_to_free 92/100, free_to_compartment 36/100, compartment_to_free 56/100, compartment_to_compartment 10/100, same_compartment 24/100, anywhere 93/100|free_to_free 288/300, free_to_compartment 116/300, compartment_to_free 180/300, compartment_to_compartment 34/300, same_compartment 87/300, anywhere 287/300|
|bellman release|body spheres a0.5 f0.15|free_to_free 92/100, free_to_compartment 36/100, compartment_to_free 56/100, compartment_to_compartment 10/100, same_compartment 24/100, anywhere 93/100|free_to_free 290/300, free_to_compartment 120/300, compartment_to_free 173/300, compartment_to_compartment 32/300, same_compartment 88/300, anywhere 288/300|
|bellman release|none  a0.5 f0.15|free_to_free 90/100, free_to_compartment 22/100, compartment_to_free 20/100, compartment_to_compartment 2/100, same_compartment 7/100, anywhere 87/100|free_to_free 265/300, free_to_compartment 54/300, compartment_to_free 50/300, compartment_to_compartment 3/300, same_compartment 9/300, anywhere 260/300|
|bellman release|pieces faces a0.5 f0.15|free_to_free 91/100, free_to_compartment 7/100, compartment_to_free 66/100, compartment_to_compartment 2/100, same_compartment 10/100, anywhere 91/100|free_to_free 287/300, free_to_compartment 101/300, compartment_to_free 217/300, compartment_to_compartment 42/300, same_compartment 90/300, anywhere 291/300|
|bellman release|pieces faces a0.5 f0.15|free_to_free 91/100, free_to_compartment 7/100, compartment_to_free 66/100, compartment_to_compartment 2/100, same_compartment 10/100, anywhere 91/100|free_to_free 288/300, free_to_compartment 99/300, compartment_to_free 215/300, compartment_to_compartment 43/300, same_compartment 94/300, anywhere 291/300|
|bellman release|pieces spheres a0.5 f0.1|free_to_free 96/100, free_to_compartment 35/100, compartment_to_free 63/100, compartment_to_compartment 13/100, same_compartment 36/100, anywhere 94/100|free_to_free 291/300, free_to_compartment 120/300, compartment_to_free 218/300, compartment_to_compartment 48/300, same_compartment 128/300, anywhere 290/300|
|bellman release|pieces spheres a0.5 f0.15|free_to_free 93/100, free_to_compartment 38/100, compartment_to_free 74/100, compartment_to_compartment 16/100, same_compartment 40/100, anywhere 93/100|free_to_free 291/300, free_to_compartment 121/300, compartment_to_free 249/300, compartment_to_compartment 57/300, same_compartment 136/300, anywhere 293/300|
|metric release|body spheres a0.5 f0.15|free_to_free 86/100, free_to_compartment 25/100, compartment_to_free 49/100, compartment_to_compartment 8/100, same_compartment 25/100, anywhere 89/100|free_to_free 278/300, free_to_compartment 83/300, compartment_to_free 160/300, compartment_to_compartment 26/300, same_compartment 92/300, anywhere 280/300|
|metric release|none  a0.5 f0.15|free_to_free 79/100, free_to_compartment 8/100, compartment_to_free 9/100, compartment_to_compartment 3/100, same_compartment 6/100, anywhere 76/100|free_to_free 237/300, free_to_compartment 23/300, compartment_to_free 25/300, compartment_to_compartment 2/300, same_compartment 12/300, anywhere 228/300|
|metric release|pieces faces a0.5 f0.15|free_to_free 84/100, free_to_compartment 3/100, compartment_to_free 57/100, compartment_to_compartment 1/100, same_compartment 12/100, anywhere 86/100|free_to_free 282/300, free_to_compartment 73/300, compartment_to_free 203/300, compartment_to_compartment 36/300, same_compartment 94/300, anywhere 282/300|
|metric release|pieces faces a0.5 f0.15|free_to_free 84/100, free_to_compartment 3/100, compartment_to_free 57/100, compartment_to_compartment 1/100, same_compartment 12/100, anywhere 86/100|free_to_free 283/300, free_to_compartment 74/300, compartment_to_free 200/300, compartment_to_compartment 27/300, same_compartment 92/300, anywhere 281/300|
|metric release|pieces spheres a0.5 f0.1|free_to_free 92/100, free_to_compartment 27/100, compartment_to_free 59/100, compartment_to_compartment 9/100, same_compartment 30/100, anywhere 89/100|free_to_free 287/300, free_to_compartment 89/300, compartment_to_free 203/300, compartment_to_compartment 33/300, same_compartment 107/300, anywhere 279/300|
|metric release|pieces spheres a0.5 f0.15|free_to_free 89/100, free_to_compartment 27/100, compartment_to_free 63/100, compartment_to_compartment 11/100, same_compartment 38/100, anywhere 89/100|free_to_free 288/300, free_to_compartment 94/300, compartment_to_free 220/300, compartment_to_compartment 42/300, same_compartment 124/300, anywhere 282/300|

One training seed per run; planning seeds 0/1/2 are not training seeds; no significance claim. Modified environment (host backend), see environment_override.json in each run.
