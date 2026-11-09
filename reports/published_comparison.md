# Published comparison (web search, NOT verified against the primary tables)

Searched 2026-11-09 (three web searches, one extended). The tables I could reach were garbled text extractions,
so **no number below should be quoted as a published result until checked in the PDEBench paper appendix
(NeurIPS 2022 Datasets & Benchmarks, "Detailed Baseline Score")**.

What the searches did say:
- PDEBench (Takamoto et al., arXiv 2210.07182): FNO is generally the best baseline, but its error grows as the
  Burgers viscosity decreases (strong discontinuities, Gibbs phenomenon). U-Net baseline uses the pushforward trick
  (autoregressive training).
- A Burgers nu=0.01 table block listed nRMSE of about 2.2e-1 (U-Net), 7.8e-3 (FNO), 8.5e-1 (PINN). The same text
  extraction repeated those numbers under a Darcy caption, so the labelling may be wrong. The 2.2e-1 U-Net value
  is also reported as 0.2222 for Burgers nu=0.01 in PDEformer-1 (arXiv 2407.06664).

Ours (test set, 1000 samples, relative L2 over the whole trajectory, one-shot initial condition -> 201 frames,
256 points, 4000 training samples, 300 epochs): FNO 0.0267 +- 0.0004 (3 seeds), U-Net 0.0904 +- 0.0053.

Why these are NOT comparable as-is:
- different training protocol (PDEBench: autoregressive rollouts, more training samples, 1024 points; ours: one-shot, 4000 samples, 256 points);
- the exact definition of nRMSE (normalisation, averaging over time/space/samples) was not confirmed;
- the label question above.

Honest reading: the possibly-mislabelled numbers would put our FNO about 3.4x above the PDEBench FNO and our
U-Net about 2.4x below the PDEBench U-Net. Neither claim is established. The ordering FNO < U-Net matches the
paper's general FNO-over-U-Net statement, but the "FNO degrades at low viscosity" remark was not reproduced as a
disadvantage here (our FNO beats our U-Net at nu=0.01 by ~3.4x).

Sources: https://arxiv.org/pdf/2210.07182 , https://arxiv.org/pdf/2407.06664 ,
https://alphaxiv.org/overview/2210.07182v7 , https://arxiv.org/pdf/2407.08615 (MgFNO: FNO on a different
Burgers setting, nu=0.1, resolution 1024, relative error about 1.6e-2 -- a different dataset, not comparable).
