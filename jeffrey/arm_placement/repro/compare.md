| p_d055_s46_h60 (0.55 m, ±0.46 m, 0.60 m high) | original (4090, original queries) | original checkpoints re-planned here | retrained here, original data + queries | re-created from scratch here (new suite) |
|---|---|---|---|---|
| Euclid (control), ntrl-demo planner 600 | 153 |  | 151 | 210 |
| lift, ntrl-demo planner 600 | 474 (5) | 475 (6) | 464 (2) | 527 (0) |
| lift, reverse 600 | 483 (4) | 490 (5) | 504 (2) | 450 (2) |
| lift, two-ended 300 | 582 (13) | 589 (18) | 602 (14) | 557 (10) |
| lift, two-ended 600 | 638 (20) | 650 (23) | 671 (27) | 627 (27) |
| lift, two-ended 1,200 | 698 (42) | 704 (43) | 704 (35) | 674 (38) |
| **lift, two-ended 2,400** | 733 (59) | 742 (60) | 741 (53) | 712 (62) |
| ntrl-demo, own planner 600 | 42 (0) | 42 (0) | 40 (0) | 54 (0) |
| ntrl-demo, reverse 600 | 44 (0) | 48 (0) | 43 (0) | 65 (0) |
| ntrl-demo, two-ended 600 | 45 (0) | 47 (0) | 39 (0) | 61 (0) |
| ntrl-demo, two-ended 2,400 | 45 (0) | 47 (0) | 39 (0) | 61 (0) |
| lift seed 87359, ntrl-demo planner 600 |  |  |  | 518 (5) |
| lift seed 87359, two-ended 2,400 |  |  |  | 691 (43) |
| ntrl-demo seed 87359, own planner 600 | 37 (0) |  |  | 27 (0) |

Runs (re-created): {"tests": "20261123T191615Z-b687b76d", "prep": "20261123T193757Z-11356432", "euclid": "20261123T193854Z-e5462df2", "ntrl_s87358": "20261123T193917Z-e343a228", "lift_s87358": "20261123T193854Z-089d42cd", "plan_lift_s87358": "20261123T200524Z-33106626", "budget_lift_s87358": "20261123T201517Z-78ffc5d5", "ntrl_s87359": "20261123T200524Z-910f5cd9", "picks_lift": "20261123T202723Z-1991fced", "plan_ntrl_s87358": "20261123T203248Z-d67cd04c", "budget_ntrl_s87358": "20261123T203534Z-e62c527f", "picks_ntrl": "20261123T203955Z-31202de5", "lift_s87359": "20261123T200524Z-357d53ee", "budget_lift_s87359": "20261123T204434Z-2ada4e29", "demo": "20261123T204929Z-e274f0e8"} 
Runs (frozen): {"prep": "FROZEN", "euclid": "20261123T205002Z-389ddb9c", "ntrl_s87358": "20261123T205027Z-2401133c", "lift_s87358": "20261123T205002Z-e96c3257", "plan_lift_s87358": "20261123T211509Z-daebfa31", "budget_lift_s87358": "20261123T211755Z-c33852e1", "picks_lift": "20261123T212201Z-a4632d02", "plan_ntrl_s87358": "20261123T212439Z-36f8b67e", "budget_ntrl_s87358": "20261123T212543Z-fda053f2", "picks_ntrl": "20261123T212742Z-c1dc078c"}
