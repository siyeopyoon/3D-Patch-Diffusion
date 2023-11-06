## High-resolution 3D CT synthesis from Single X-ray Image using 3D Diffusion Models

**High-resolution 3D CT synthesis from Single X-ray Image using 3D Diffusion Models**<br>
Siyeop Yoon, Jay Sairam Pratap,Wen-Chih Liu, Matthew Tivnan, Hui Ren, Quanzheng Li, Neal Chen, Xiang Li


Abstract: 3D Computed Tomography (CT) offers invaluable geometric insights into bone structures, but the high radiation dose and medical cost constraints are significant barriers. Moreover, CT reconstruction demands multiple X-ray projections, necessitating a dedicated scanning system, albeit bi-directional X-rays are already the front-line diagnostic tool in routine practice. Therefore, reconstructing 3D bone structures from bidirectional X-ray data can reduce the need for additional CT scans, provide rapid access to 3D information, and lower medical costs. Recently, diffusion models have emerged as potent tools for generating high-fidelity images. However, their high computational, especially in 3D volumetric generation tasks, limited their utility. Collecting large-scale datasets for training diffusion models in clinical environments presents another challenge. In this study, we introduce a novel approach to synthesize 3D CT volumes from a bi-directional X-ray projection using a 3D diffusion model. To reduce the computational burden and the need for a large dataset, our 3D diffusion model was trained using patch-wise loss. A conditional score function of our model incorporates 2D bi-directional X-ray images and patch coordinate information to synthesize high-resolution CT. Initial findings indicate that our diffusion model synthesizes 3D CT volumes from a bi-directional X-ray, effectively capturing 3D geometric correlations while enabling single-GPU training and rapid 3D volumetric sampling.


## Acknowledgement
We build our 3D Patch Diffusion upon the [EDM](https://github.com/NVlabs/edm) and [Patch Diffusion] https://github.com/Zhendong-Wang/Patch-Diffusion



We thank [EDM](https://github.com/NVlabs/edm) and [Patch Diffusion] https://github.com/Zhendong-Wang/Patch-Diffusion authors for providing the great code base.
