# data/raw provenance

Every file actually on disk, with size and SHA-256. Nothing here is
synthetic. Anything not obtained is listed under "Not obtained" with the
reason, not silently substituted.

Recorded 2026-08-29. Regenerate the checksum block with:

    cd data/raw && find . -type f -not -path './otari_repo/.git/*' \
      -exec sha256sum {} \; | sort -k2

---

## 1. Otari framework

**Repository** `github.com/FunctionLab/otari`, cloned to `otari_repo/`
at commit `0b2925c1ff5cdde50836399274e5806f74bf48ab` (2025-12-17).

**Resources bundle** DOI `10.5281/zenodo.16433544`. That concept DOI
redirects to version record **16433545**, "Otari framework resources",
CC-BY-4.0, a single file:

| file | size | MD5 (from Zenodo) |
|---|---|---|
| `otari_resources.tar.gz` | 5.094 GB | `bef09c5bc5b3b7eb59042c1a1291d6e4` |

Per the otari README this bundle contains the ConvSplice, Sei and Seqweaver
model weights, the hg38 FASTA with `pyfaidx` index files, GENCODE v47
annotations, and the transcriptome datasets used for training and
validation. **STATUS: download in progress at the time of writing.** The
contents, their individual sizes and their SHA-256 sums are NOT yet listed
below, and no claim about them may be made until they are. The downloaded
archive must be verified against the MD5 above before extraction.

---

## 2. Massively parallel splicing assays

### 2.1 FAS/CD95 exon 6 — OBTAINED, but not the library that was requested

Julien P, Miñana B, Baeza-Centurion P, Valcárcel J, Lehner B. "The complete
local genotype-phenotype landscape for the alternative splicing of a human
exon." *Nature Communications* 7:11558 (2016). PMID 27161764, PMC4866304,
open access. Supplementary retrieved from the Europe PMC
`supplementaryFiles` endpoint for PMC4866304.

**What the public supplementary actually contains** (verified by reading
the files, not by reading the abstract):

| table | file | rows | content |
|---|---|---|---|
| Supp. Table 1 | `ncomms11558-s2.xlsx` | 189 | ALL single mutants: 63 positions x 3 alternatives, with `EnrichmentScore`, p-value, FDR, SD, median coverage and 3 replicate scores |
| Extended Data Table 2 | `ncomms11558-s3.xlsx` | 16,728 | double mutants, `Sequence` + `ID` + `EnrichmentScore` |
| Supp. Table 3 | `ncomms11558-s4.xlsx` | 16,728 | pairwise epistasis: single and double scores, expected sum, empirical epistasis and p-value |

Wild-type exon length **63 nt**; **16,917 distinct genotypes** in total
(189 singles + 16,728 doubles). Phenotype is `EnrichmentScore`, a
log-scale selection score, range -6.7 to +1.5 for singles; it is NOT a PSI
value and any conversion must be stated explicitly.

**Discrepancy with the request, stated plainly.** The task described this
as "all 3072 genotypes at 11 positions". The cited paper's public
supplementary is not that: it is a complete single-and-double-mutant
landscape over all 63 positions. 3072 genotypes at 11 positions is a
combinatorial design, which matches Baeza-Centurion et al., "Combinatorial
Genetics Reveals a Scaling Law for the Effects of Mutations on Splicing",
*Cell* 176:549-563 (2019), a later study of the same exon by the same
group. That dataset has NOT been downloaded. **Decide which is wanted
before any modelling: they are different libraries and support different
claims.**

### 2.2 BRCA2 exons 17-19 5' splice sites — OBTAINED

Wong MS, Kinney JB, Krainer AR. "Quantitative activity profile and context
dependence of all human 5' splice sites." *Molecular Cell* 71:1012-1026.e3
(2018). Obtained as the processed dataset shipped with MAVE-NN
(`github.com/jbkinney/mavenn`, `mavenn/examples/datasets/`), fetched
directly over HTTPS without installing the package.

| file | size | rows |
|---|---|---|
| `mpsa_data.csv.gz` | 431,713 B | 30,483 |
| `mpsa_replicate_data.csv.gz` | 449,817 B | 30,697 |

Columns `set, tot_ct, ex_ct, y, x`. `x` is the variant 5' splice site,
**9 nt** in RNA alphabet, matching `NNN/GYNNNN` where `/` is the
exon-intron boundary. `y` is **log10 percent-spliced-in**. Library 1,
replicate 1, BRCA2 exons 17-19.

**On the count.** The request said 32,768 variants. That is the *designed*
library size: 7 free N positions and one Y position gives 4^7 x 2 =
32,768. **30,483** is the number actually measured and released. Both
numbers are correct for different things; only 30,483 exist as data.

### 2.3 Not obtained

- **MFASS** (Cheung et al. 2019), **MaPSy** (Soemedi et al. 2017),
  **Vex-seq** (Adamson et al. 2018): not attempted. 2.1 and 2.2 satisfy
  the requirement of at least one MPSA with position-resolved
  measurements, so nothing was substituted for a failed download.
- **Baeza-Centurion et al. 2019 combinatorial FAS library**: not
  downloaded, pending the decision in 2.1.

---

## 3. Checksums

SHA-256 for every file except `otari_resources.tar.gz`, which is still
downloading and is covered by the Zenodo MD5 above.

```
56baef7e143f8906051c1627ce252b13ff54e749883430c773874e94941910ea     5816132  fas_PMC4866304_supp.zip
db6bd53b6e50e63f1e01a325a5d0b7d3ade3b7f3f39bfa5e1f2f87798453b0b8       14608  fas_supp/ncomms11558-f1.gif
6398ec9b8f3e3d868370b27d56fa8e68fa59f005653509650b5caea6d3c085cd      112243  fas_supp/ncomms11558-f1.jpg
0762676c39e4267225f9d9fa55b3f4aa72845ba91f3fa24c4c9b0c70902d81bc       17955  fas_supp/ncomms11558-f2.gif
8335c7bbf488606d2c17b3cad06979eb8b6d1d8f710ea46ded9e779c3cd203f6      158822  fas_supp/ncomms11558-f2.jpg
5c1739347da1d40db35331d96034794298288de06e67c31d0d542cb41e8ffa9e       18829  fas_supp/ncomms11558-f3.gif
80552be000c8f4d52c9456db926d63c2a85d552bb4131f847c45349b9b135795      182986  fas_supp/ncomms11558-f3.jpg
1ff5350224091c2a1861ae7b2ef624fbefde328f6ceeda77e24db5ad1b58d54e       22043  fas_supp/ncomms11558-f4.gif
6247bc2364781ddcb3fef473c1a46da39e7f8b453c06a8118f512a782b778488      256117  fas_supp/ncomms11558-f4.jpg
4a0c0c4984628e87b6a32035beab14a422fd6c2ee631c65bf8565bdf94437b2f       53764  fas_supp/ncomms11558-i1.jpg
60446a93fcb27b7a63bc73d830b1eab8fbec283a525348c29f4507d9b659e5b0     2784906  fas_supp/ncomms11558-s1.pdf
bd9754a9827c75fea5eaa0ccb500a90cb298a8726b828e92bc7a58312d5cb114       35933  fas_supp/ncomms11558-s2.xlsx
0618cdfa8a37f9228a68764948cee959401b9a4f0336227d20e3b5a1e3651fbe      649940  fas_supp/ncomms11558-s3.xlsx
a179ccc85dd9f61d8e2a252e2dfd3fc4391821b6f581e9e4846ec3f8de027868     1997449  fas_supp/ncomms11558-s4.xlsx
df1251425d7bae15cbae5922c9b05e7765344fc4bb13970fb44b51370c09cf74      431713  mpsa_data.csv.gz
c18a824798617fff07806132aadf9e8c1e6547c770dce8b8d515bb3189e80415      449817  mpsa_replicate_data.csv.gz
```

`otari_repo/` is a git checkout pinned at the commit above; its file
checksums are the commit's and are not duplicated here.

---

## 4. Gate on modelling

Per the standing instruction, no modelling proceeds until this file lists
real files with checksums for the data a model would consume. As of this
writing that is satisfied for the two MPSA datasets (section 2) and NOT
satisfied for the Otari resources bundle (section 1), whose contents are
still downloading and unverified.
