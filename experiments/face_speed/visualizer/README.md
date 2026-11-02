# Collision-cover viewer

Interactive 3D page showing the three collision models of the face-speed experiment on the UR5e + Robotiq arm in the
cabinet: the 53 padded cuRobo spheres (release label), the per-link pieces, and the 5-box / 30-face cover, plus the
link collision meshes, the closest self-collision sphere pair and the cabinet cuboids. Each piece is coloured by its
speed clip(d, 5 mm, 50 mm)/50 mm, recomputed live in the browser as joints move or a path plays (certified witness
paths and the Bellman RK4 paths of the release-speed and box-face-speed fields for 8 queries per suite).

Published: https://claude.ai/artifact/1Qsg2Nz7gDQuDSC4NTwufK

```bash
~/.venvs/bellman-ntfield/bin/python experiments/face_speed/visualizer/export.py   # geometry.json from robot.py / clearance.py
node experiments/face_speed/visualizer/check.cjs                                  # JS clearances vs Python (48 configurations)
python3 experiments/face_speed/visualizer/build.py                                # -> collision_viewer.html (open in a browser)
```

`geometry.js` is a port of robot.py / faces.py / clearance.py; check.cjs found it within 4e-7 m of the Python
clearances for all three covers and the gripper position.
