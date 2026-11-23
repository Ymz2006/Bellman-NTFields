# LBS benchmark: registered runs

| scene | arm | seed | solved % | through a wall % | stalled % | T within 10 % | median T / T* | detours seen through % | train min | run | from |
|---|---|---|---|---|---|---|---|---|---|---|---|
| maze 10 x 10 | LBS (lift + Bellman + segment) | 87358 | 100.0 (600/600) | 0.0 | 0.0 | 56.3 | 1.092 | 0.0 | 41 | 20261120T024416Z-e62e1e8c | jeffrey/lbs |
| maze 10 x 10 | no lift (Bellman + segment) | 87358 | 92.0 (552/600) | 7.5 | 0.5 | 10.7 | 1.202 | 0.0 | 34 | 20261120T032606Z-55728e7e | jeffrey/lbs |
| maze 10 x 10 | no Bellman (lift + segment) | 87358 | 94.3 (566/600) | 0.0 | 5.7 | 95.5 | 0.999 | 1.7 | 28 | 20261120T032116Z-db1f8f06 | jeffrey/lbs |
| maze 10 x 10 | no segment (lift + Bellman) | 87358 | 94.5 (567/600) | 0.0 | 5.5 | 20.5 | 1.169 | 0.0 | 29 | 20261120T025056Z-7cefce43 | jeffrey/lbs |
| maze 10 x 10 | ntrl-demo | 87358 | 83.3 (500/600) | 14.2 | 2.5 | 11.5 | 1.314 | 0.0 | 5 | 20261120T024436Z-582ae344 | jeffrey/lbs |
| maze 13 x 13 | LBS (lift + Bellman + segment) | 87358 | 99.5 (597/600) | 0.5 | 0.0 | 31.0 | 1.124 | 0.0 | 56 | 20261120T035026Z-84d35844 | jeffrey/lbs |
| maze 13 x 13 | no lift (Bellman + segment) | 87358 | 69.2 (415/600) | 30.8 | 0.0 | 6.8 | 1.241 | 0.0 | 47 | 20261120T044716Z-a9cdbc56 | jeffrey/lbs |
| maze 13 x 13 | no Bellman (lift + segment) | 87358 | 88.7 (532/600) | 0.0 | 11.3 | 90.2 | 1.015 | 3.8 | 26 | 20261120T043456Z-7da93be4 | jeffrey/lbs |
| maze 13 x 13 | no segment (lift + Bellman) | 87358 | 98.3 (590/600) | 0.7 | 1.0 | 12.1 | 1.186 | 0.0 | 27 | 20261120T040706Z-7f5125b2 | jeffrey/lbs |
| maze 13 x 13 | ntrl-demo | 87358 | 64.2 (385/600) | 34.8 | 1.0 | 6.8 | 1.693 | 0.0 | 5 | 20261120T040046Z-fedc4e9d | jeffrey/lbs |
| maze 18 x 18 | LBS (lift + Bellman + segment) | 87358 | 94.6 (681/720) | 0.6 | 4.9 | 19.8 | 1.140 | 0.1 | 60 | 20261120T053456Z-a02310b0 | jeffrey/lbs |
| maze 18 x 18 | no lift (Bellman + segment) | 87358 | 49.4 (356/720) | 44.6 | 6.0 | 0.6 | 1.388 | 0.0 | 49 | 20261120T073256Z-a5c96950 | jeffrey/lbs |
| maze 18 x 18 | no Bellman (lift + segment) | 87358 | 46.9 (338/720) | 0.0 | 53.1 | 52.0 | 0.921 | 31.9 | 23 | 20261120T070806Z-8a82d98c | jeffrey/lbs |
| maze 18 x 18 | no segment (lift + Bellman) | 87358 | 86.5 (623/720) | 0.0 | 13.5 | 11.3 | 1.201 | 0.5 | 30 | 20261120T063626Z-e0b71352 | jeffrey/lbs |
| maze 18 x 18 | ntrl-demo | 87358 | 35.4 (255/720) | 48.2 | 16.4 | 2.0 | 1.519 | 0.2 | 5 | 20261120T060816Z-2cdfe357 | jeffrey/lbs |
| gap maze x1 (10 x 10 base, 16 cells) | LBS (lift + Bellman + segment) | 87358 | 100.0 (900/900) | 0.0 | 0.0 | 55.5 | 0.925 | 0.0 | 47 | 20261120T082336Z-33650f07 | jeffrey/lbs |
| gap maze x1 (10 x 10 base, 16 cells) | no lift (Bellman + segment) | 87358 | 81.8 (736/900) | 15.8 | 2.4 | 45.2 | 0.933 | 0.0 | 40 | 20261120T091226Z-155e6d58 | jeffrey/lbs |
| gap maze x1 (10 x 10 base, 16 cells) | no Bellman (lift + segment) | 87358 | 71.0 (639/900) | 0.0 | 29.0 | 26.3 | 0.851 | 36.6 | 21 | 20261120T090236Z-ae312b28 | jeffrey/lbs |
| gap maze x1 (10 x 10 base, 16 cells) | no segment (lift + Bellman) | 87358 | 100.0 (900/900) | 0.0 | 0.0 | 80.1 | 1.044 | 0.0 | 26 | 20261120T083516Z-e92ba09c | jeffrey/lbs |
| gap maze x1 (10 x 10 base, 16 cells) | ntrl-demo | 87358 | 70.2 (632/900) | 24.4 | 5.3 | 62.3 | 1.073 | 0.4 | 5 | 20261120T082856Z-8764ef43 | jeffrey/lbs |
| gap maze x2 (22 cells) | LBS (lift + Bellman + segment) | 87358 | 99.9 (899/900) | 0.1 | 0.0 | 64.5 | 0.968 | 0.0 | 52 | 20261120T095336Z-719315ba | jeffrey/lbs |
| gap maze x2 (22 cells) | no lift (Bellman + segment) | 87358 | 78.9 (710/900) | 12.8 | 8.3 | 68.4 | 0.942 | 0.0 | 35 | 20261120T111956Z-c8fd8006 | jeffrey/lbs |
| gap maze x2 (22 cells) | no Bellman (lift + segment) | 87358 | 41.6 (374/900) | 0.0 | 58.4 | 39.6 | 0.883 | 89.8 | 29 | 20261120T105436Z-020c02f1 | jeffrey/lbs |
| gap maze x2 (22 cells) | no segment (lift + Bellman) | 87358 | 99.8 (898/900) | 0.0 | 0.2 | 90.7 | 1.049 | 0.0 | 26 | 20261120T105216Z-295ce890 | jeffrey/lbs |
| gap maze x2 (22 cells) | ntrl-demo | 87358 | 45.8 (412/900) | 17.2 | 37.0 | 72.4 | 1.054 | 6.1 | 5 | 20261120T104726Z-007aeff7 | jeffrey/lbs |
| gap maze x4 (34 cells) | LBS (lift + Bellman + segment) | 87358 | 99.3 (894/900) | 0.0 | 0.7 | 89.8 | 1.004 | 0.0 | 76 | 20261120T112717Z-11b4d878 | jeffrey/lbs |
| gap maze x4 (34 cells) | no lift (Bellman + segment) | 87358 | 63.9 (575/900) | 30.4 | 5.7 | 93.6 | 0.985 | 0.0 | 67 | 20261120T142436Z-b2939fac | jeffrey/lbs |
| gap maze x4 (34 cells) | no Bellman (lift + segment) | 87358 | 27.0 (243/900) | 0.0 | 73.0 | 62.2 | 0.944 | 99.7 | 40 | 20261120T133856Z-4cacbcaa | jeffrey/lbs |
| gap maze x4 (34 cells) | no segment (lift + Bellman) | 87358 | 94.1 (847/900) | 0.0 | 5.9 | 86.1 | 0.988 | 0.0 | 42 | 20261120T125416Z-f6893741 | jeffrey/lbs |
| gap maze x4 (34 cells) | ntrl-demo | 87358 | 33.0 (297/900) | 28.7 | 38.3 | 76.2 | 1.049 | 8.8 | 5 | 20261120T124626Z-146b5fcf | jeffrey/lbs |
| Gibson Allensville (43 cells) | LBS (lift + Bellman + segment) | 1 | 97.1 (699/720) | 0.0 | 2.9 | 92.0 | 0.992 | 0.1 | 63 | 20261120T064946Z-ef11536e | jeffrey/lbs |
| Gibson Allensville (43 cells) | LBS (lift + Bellman + segment) | 2 | 90.7 (653/720) | 2.5 | 6.8 | 84.5 | 0.974 | 3.2 | 64 | 20261120T171156Z-04b846cf | jeffrey/lbs |
| Gibson Allensville (43 cells) | LBS (lift + Bellman + segment) | 87358 | 92.5 (666/720) | 0.0 | 7.5 | 81.0 | 0.987 | 5.6 | 65 | 20261120T050206Z-a97e6309 | jeffrey/lbs |
| Gibson Allensville (43 cells) | no lift (Bellman + segment) | 87358 | 98.8 (711/720) | 1.2 | 0.0 | 91.1 | 0.969 | 1.0 | 53 | 20261120T143926Z-5cdf0c70 | jeffrey/lbs |
| Gibson Allensville (43 cells) | no Bellman (lift + segment) | 87358 | 84.6 (609/720) | 0.0 | 15.4 | 61.1 | 0.932 | 69.4 | 40 | 20261117T182927Z-0a0eca6d | jeffrey/lift_midpoint |
| Gibson Allensville (43 cells) | no segment (lift + Bellman) | 1 | 99.6 (717/720) | 0.0 | 0.4 | 70.3 | 0.944 | 9.2 | 33 | 20261120T075416Z-6b3e7bec | jeffrey/lbs |
| Gibson Allensville (43 cells) | no segment (lift + Bellman) | 2 | 91.2 (657/720) | 0.0 | 8.8 | 76.9 | 0.966 | 8.3 | 47 | 20261120T225917Z-68b6362a | jeffrey/lbs |
| Gibson Allensville (43 cells) | no segment (lift + Bellman) | 87358 | 94.2 (678/720) | 0.0 | 5.8 | 84.6 | 0.979 | 2.5 | 33 | 20261120T061506Z-7b0ee268 | jeffrey/lbs |
| Gibson Allensville (43 cells) | ntrl-demo | 87358 | 83.9 (604/720) | 15.6 | 0.6 | 66.5 | 1.064 | 0.3 | 9 | 20261117T121605Z-9c558da3 | jeffrey/lift_scale |
| Gibson Lakeville (83 cells) | LBS (lift + Bellman + segment) | 87358 | 90.8 (763/840) | 0.0 | 9.2 | 75.3 | 0.960 | 22.7 | 84 | 20261120T092547Z-bde76789 | jeffrey/lbs |
| Gibson Lakeville (83 cells) | no lift (Bellman + segment) | 87358 | 44.9 (377/840) | 0.5 | 54.6 | 2.7 | 0.180 | 99.0 | 44 | 20261120T153347Z-a3c68687 | jeffrey/lbs |
| Gibson Lakeville (83 cells) | no Bellman (lift + segment) | 87358 | 60.7 (510/840) | 0.0 | 39.3 | 14.3 | 0.690 | 97.8 | 45 | 20261117T191038Z-55bb04e3 | jeffrey/lift_midpoint |
| Gibson Lakeville (83 cells) | no segment (lift + Bellman) | 87358 | 97.0 (815/840) | 0.0 | 3.0 | 66.9 | 0.930 | 14.7 | 33 | 20261120T132707Z-abf43613 | jeffrey/lbs |
| Gibson Lakeville (83 cells) | ntrl-demo | 87358 | 72.7 (611/840) | 4.6 | 22.6 | 73.8 | 0.958 | 33.3 | 5 | 20261120T171347Z-f77982a7 | jeffrey/lbs |
| Gibson Allensville in 3-D | LBS (lift + Bellman + segment) | 87358 | 78.3 (564/720) | 0.1 | 21.5 | 67.2 | 0.958 | 83.5 | 65 | 20261120T235517Z-71d6a2c5 | jeffrey/lbs |
| Gibson Allensville in 3-D | ntrl-demo | 87358 | 64.7 (466/720) | 35.3 | 0.0 | 65.4 | 0.991 | 89.1 | 5 | 20261121T005037Z-585c4d0f | jeffrey/lbs |
| Gibson Mifflinburg in 3-D | LBS (lift + Bellman + segment) | 87358 | 90.6 (652/720) | 0.0 | 9.4 | 89.0 | 0.990 | 37.3 | 86 | 20261120T232237Z-6dfa59f7 | jeffrey/lbs |
| Gibson Mifflinburg in 3-D | ntrl-demo | 87358 | 68.2 (491/720) | 28.9 | 2.9 | 77.2 | 1.035 | 35.3 | 5 | 20261120T234857Z-2727e491 | jeffrey/lbs |
| Gibson Noxapater in 3-D | LBS (lift + Bellman + segment) | 87358 | 95.4 (687/720) | 0.0 | 4.6 | 81.2 | 1.010 | 8.6 | 81 | 20261121T010917Z-fd0e606d | jeffrey/lbs |
| Gibson Noxapater in 3-D | ntrl-demo | 87358 | 75.6 (544/720) | 18.2 | 6.2 | 77.9 | 1.042 | 19.2 | 5 | 20261121T005627Z-d6eee0aa | jeffrey/lbs |
| Gibson Markleeville in 3-D | LBS (lift + Bellman + segment) | 87358 | 92.5 (666/720) | 0.0 | 7.5 | 86.2 | 1.010 | 56.5 | 69 | 20261121T021707Z-9645d217 | jeffrey/lbs |
| Gibson Markleeville in 3-D | ntrl-demo | 87358 | 79.3 (571/720) | 17.6 | 3.1 | 87.2 | 1.024 | 51.0 | 6 | 20261121T010147Z-51711f85 | jeffrey/lbs |
| Gibson Lakeville in 3-D | LBS (lift + Bellman + segment) | 87358 | 68.0 (571/840) | 0.2 | 31.8 | 72.5 | – | – | 66 | 20261121T010828Z-3fc6f1af | jeffrey/lbs |
| Gibson Lakeville in 3-D | ntrl-demo | 87358 | 58.2 (489/840) | 41.8 | 0.0 | 67.5 | – | – | 5 | 20261121T010238Z-6ac4101f | jeffrey/lbs |
| maze 13 x 13 | control lbsP | 87358 | 94.5 (567/600) | 0.0 | 5.5 | 50.1 | 1.098 | 0.1 | 56 | 20261120T172256Z-7dbca24f | jeffrey/lbs |
| maze 18 x 18 | control lbsP | 87358 | 91.9 (662/720) | 0.4 | 7.6 | 9.5 | 1.181 | 0.1 | 59 | 20261120T181936Z-b587cb68 | jeffrey/lbs |
| gap maze x1 (10 x 10 base, 16 cells) | control lbsP | 87358 | 100.0 (900/900) | 0.0 | 0.0 | 53.4 | 0.936 | 0.0 | 48 | 20261120T192046Z-32364492 | jeffrey/lbs |
| gap maze x2 (22 cells) | control lbsP | 87358 | 100.0 (900/900) | 0.0 | 0.0 | 82.0 | 0.980 | 0.0 | 52 | 20261120T200957Z-14e899e7 | jeffrey/lbs |
| gap maze x4 (34 cells) | control lbsP | 87358 | 93.4 (841/900) | 0.3 | 6.2 | 86.4 | 1.017 | 0.0 | 77 | 20261120T210347Z-1843335d | jeffrey/lbs |
| maze 10 x 10 | control lbsP | 87358 | 88.0 (528/600) | 4.2 | 7.8 | 42.4 | 1.105 | 0.8 | 40 | 20261120T181717Z-6a238a9c | jeffrey/lbs |
| Gibson Allensville (43 cells) | control lbsP | 87358 | 88.6 (638/720) | 0.0 | 11.4 | 79.9 | 0.964 | 23.9 | 63 | 20261120T185826Z-cd4d99a6 | jeffrey/lbs |
| Gibson Lakeville (83 cells) | control lbsP | 87358 | 95.5 (802/840) | 0.0 | 4.5 | 63.4 | 0.919 | 15.5 | 86 | 20261120T200237Z-49ce09a5 | jeffrey/lbs |
| Gibson Klickitat (96 cells) | control lbsP | 87358 | 89.3 (750/840) | 0.0 | 10.7 | 32.0 | 0.853 | 53.8 | 86 | 20261120T213107Z-8d07cff5 | jeffrey/lbs |
| Gibson Lakeville (83 cells) | control noLw | 87358 | 48.9 (411/840) | 2.5 | 48.6 | 8.3 | 0.285 | 87.9 | 54 | 20261120T222357Z-f16d151d | jeffrey/lbs |
