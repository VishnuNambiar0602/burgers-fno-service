---
title: Burgers FNO Service
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# burgers-fno-service

A Fourier Neural Operator (FNO) that predicts the solution of the 1D viscous Burgers equation
(nu = 0.01, periodic domain) from an initial condition. FastAPI backend: `/health`, `/model-info`, `/predict`, `/docs`.

**Status: skeleton deployment.** Trained only at nu = 0.01 on one family of initial conditions at 256 points; it does not
conserve mass and is least accurate near shocks. An out-of-distribution flag is returned with every prediction;
no flag does not certify accuracy. See `/model-info` for measured metrics and limitations.

Data: PDEBench (Takamoto et al., 2022), DaRUS DOI 10.18419/darus-2986, licence CC BY 4.0.
