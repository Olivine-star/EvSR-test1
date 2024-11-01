# Overview
This repo provides the code of [Neuromorphic Imaging with Super-Resolution](https://doi.org/10.1109/TIP.2024.3374074).
```
@article{zhang2024tcsvt,
  title    =  {Neuromorphic Imaging with Super-Resolution},
  author   =  {Pei Zhang and Shuo Zhu and Chutian Wang and Yaping Zhao and Edmund Y. Lam},
  journal  =  {IEEE Transactions on Circuits and Systems for Video Technology},
  doi      =  {10.1109/TCSVT.2024.3482436},
}
```
![DEMO](./imgs/workflow.png)

![DEMO](./imgs/ex.png)

## Implementation
1. Prepare your event sample (with `t, x, y, p` entries) in the `data` folder.
2. Run the program
   ```
   CUDA_VISIBLE_DEVICES=0 python run_task.py
   ```
3. Check the `result` folder for your output files.
