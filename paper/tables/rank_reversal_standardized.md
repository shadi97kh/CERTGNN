# ABLATIONS 1.8 with per-instance standardization

git SHA `d2e7fed` (DIRTY), config `4cf6e87a`, 10 seeds, mean [95% bootstrap CI].

Cross-instance reversal rate: among pairs of instances at different baseline
rates where instance a truly has the larger latent effect, the fraction the
arm's score orders the other way. Scores are single-node-removal scores over
every candidate node; standardized arms divide each instance's score vector
by its own spread, as SQUID does before comparing attribution maps across
loci.

| baseline pair (b vs a) | probability, raw | probability, standardized | latent, raw | latent, standardized |
|---|---|---|---|---|
| 0.5 vs 0.9 | 0.540 [0.516, 0.563] (n=10) | 0.498 [0.476, 0.523] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.433 [0.408, 0.462] (n=10) |
| 0.5 vs 0.95 | 0.717 [0.699, 0.737] (n=10) | 0.504 [0.475, 0.534] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.474 [0.453, 0.495] (n=10) |
| 0.5 vs 0.99 | 0.941 [0.930, 0.951] (n=10) | 0.516 [0.477, 0.551] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.558 [0.532, 0.587] (n=10) |

## Reading

- The latent arms are zero by construction on this substrate: the latent
  removal score *is* the quantity being ranked. They are a consistency check,
  not evidence.

- Chance level is 0.500: a score carrying no information about the true effect
  orders instances at random. Below chance means it tracks the truth; above
  chance means it is systematically reversed.

- Raw probability-space: 0.941 [0.930, 0.951] — reliably worse than chance, i.e. systematically reversed.
- Standardized probability-space: 0.516 [0.477, 0.551], Spearman with the truth in [0.298, 0.404].
- Standardized probability minus standardized latent, paired per seed: 0.5 vs 0.9 0.065 [0.025, 0.102] (n=10); 0.5 vs 0.95 0.030 [0.002, 0.058] (n=10); 0.5 vs 0.99 -0.042 [-0.096, 0.004] (n=10) — the spaces still differ after standardization.
- Standardized latent-space: 0.433 [0.408, 0.462] — the same standardization applied in latent space sits at chance too, which is the tell that standardization removes magnitude information rather than a link artifact.

**The raw probability-space score is systematically reversed and gets worse as the baselines separate (up to 0.941 vs a chance level of 0.500), so row 1.8's premise reproduces under per-node scores. The per-instance standardization SQUID performs removes almost all of that bias, but it lands at chance rather than at the correct ordering: it neutralises the artifact by destroying magnitude information, not by recovering it. The latent score ranks correctly. Clause (ii) therefore survives the prior-art objection, but only in this narrowed form -- standardization is not a substitute for the latent transform when effect magnitudes must be compared across instances, which is a different use than the map-shape consistency SQUID uses it for. Scope: the latent arm is exact by construction on this substrate, so the informative contrast is raw versus standardized, not either against latent.**
