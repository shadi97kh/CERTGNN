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

**STATUS: downloaded and VERIFIED.** `md5sum -c` against the Zenodo MD5
above returns OK. Extracted to `resources/` (5.1 GB, 24 files). hg38 was
confirmed readable through `pyfaidx`: 455 contigs, `chr1` length
248,956,422, random access working through the bgzip `.gzi` index without
decompressing the FASTA.

Extracted contents, SHA-256 truncated to 16 hex characters (full sums via
the regeneration command at the top of this file):

| file | bytes | SHA-256 |
|---|---:|---|
| `resources/CTX_isoform_data.tsv.gz` | 2,360,737 | `4025386630d077ec…` |
| `resources/ESPRESSO_isoform_data.tsv.gz` | 326,206,717 | `585dd77fada500d6…` |
| `resources/GTEx_isoform_data.tsv.gz` | 353,815,006 | `e3117139f0303997…` |
| `resources/gencode.v47.basic.annotation.clean.gtf.gz` | 37,803,573 | `3f803fad25c0dffb…` |
| `resources/gencode.v47.basic.annotation.gtf.gz` | 36,228,629 | `b08c4dae664350fe…` |
| `resources/gene2chrom.pkl` | 1,566,383 | `3ceb3ca6b793967f…` |
| `resources/gene2transcripts.pkl` | 4,529,361 | `1759731b2193fbe2…` |
| `resources/hg38.fa.gz` | 1,006,351,693 | `c5c78f86d98dbb3e…` |
| `resources/hg38.fa.gz.fai` | 19,381 | `3b425de206296a5c…` |
| `resources/hg38.fa.gz.gzi` | 799,192 | `86194d38eacde5ed…` |
| `resources/model_weights/ConvSplice_model_1.pt` | 4,606,866 | `f4db10589bcde9a4…` |
| `resources/model_weights/ConvSplice_model_2.pt` | 4,606,866 | `db55581627d55176…` |
| `resources/model_weights/ConvSplice_model_3.pt` | 4,606,866 | `23e8b552458ea9ce…` |
| `resources/model_weights/ConvSplice_model_4.pt` | 4,606,866 | `bc8df1badb37a090…` |
| `resources/model_weights/ConvSplice_model_5.pt` | 4,606,866 | `0496bcef108ed649…` |
| `resources/model_weights/__init__.py` | 0 | `e3b0c44298fc1c14…` |
| `resources/model_weights/histone_features.csv` | 480,693 | `c99117c4861ebde8…` |
| `resources/model_weights/human_seqweaver.pth` | 28,851,556 | `2c5838962f276e0e…` |
| `resources/model_weights/sei.pth` | 3,559,928,168 | `9d4771d5363e4955…` |
| `resources/model_weights/sei.target.names` | 905,833 | `3a2cece0a7877dad…` |
| `resources/model_weights/seqclass.names` | 692 | `5276a0a39f4d0364…` |
| `resources/model_weights/seqweaver.colnames` | 6,151 | `1a3f49d4e1bf453f…` |
| `resources/transcript2gene.pkl` | 5,701,274 | `cd98dfc1b7be5be1…` |
| `resources/transcripts.pkl` | 62,529,377 | `af1c5ef900cdf7ec…` |

The three model weight sets (ConvSplice x5, Seqweaver, Sei) are the
sequence-to-function predictors Otari uses for node features; `sei.pth`
alone is 3.56 GB. `transcripts.pkl`, `transcript2gene.pkl`,
`gene2transcripts.pkl` and `gene2chrom.pkl` are the GENCODE-derived
indices. `GTEx_isoform_data.tsv.gz`, `ESPRESSO_isoform_data.tsv.gz` and
`CTX_isoform_data.tsv.gz` are the isoform abundance targets Otari trains
on, which are a different phenotype from the splicing measurements in
section 2.

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

### 2.3 MFASS — OBTAINED, and the only acquired dataset with topological variation

Cheung R, Insigne KD, Yao D, Burghard CP, Wang J, Hsiao Y-HE, Jones EM,
Goodman DB, Xiao X, Kosuri S. "A Multiplexed Assay for Exon Recognition
Reveals that an Unappreciated Fraction of Rare Genetic Variants Cause
Large-Effect Splicing Disruptions." *Molecular Cell* 73:183-194 (2019).
From `github.com/KosuriLab/MFASS`, `processed_data/snv/snv_data_clean.txt`.

| file | size | SHA-256 |
|---|---|---|
| `mfass_snv_data_clean.txt` | 41,445,250 B | `a637ca0e307e66ff48811ec7efa22b9ce453bc7883b04f0cacb867f7283132d8` |

| quantity | value |
|---|---|
| variant rows | 32,669 |
| distinct exons (`ensembl_id`) | **2,339** |
| exon length | 18-99 nt, 28 distinct values |
| intron 1 / intron 2 length | 30-80 / 30-81 nt, 34 / 33 distinct |
| distinct (exon, intron1, intron2) length triples | **53** |
| measurement | splicing index in [0, 1]; `v2_index` n = 31,031, median 0.991 |
| assayed fragment | 170 nt (`sequence`) |
| coordinates | hg38 (`chr`, `start`, `end`, `strand`, `snp_position_hg38_*`) |

**Why this dataset was added.** Sections 2.1 and 2.2 are single-locus
libraries: every FAS genotype is the same exon with the same flanking
introns, and every BRCA2 variant is the same splice site. Under
Definition 1 that yields ONE graph topology shared by every instance, so
ABLATIONS rows 0.6 (degree-preserving shuffle) and 0.7 (edge-free MLP)
are null by construction on them and would return "topology is
decorative" as an artifact of the library design. MFASS has 2,339 exons
across 53 distinct structural configurations, which is the topological
variation those rows require.

**A caveat that bears on Definition 1.** MFASS exons are short (median 81
nt) with short flanking introns (median 45 nt) inside a 170 nt assayed
fragment. Definition 1's 300 nt intronic radius therefore exceeds the
available context: tiled windows would extend past the sequence that was
actually measured. Either the radius is reduced to the construct, or the
windows are drawn from hg38 beyond the minigene, which is a different
object from the thing assayed. This must be decided before graphs are
built.

### 2.4 Not obtained

- **MaPSy** (Soemedi et al. 2017), **Vex-seq** (Adamson et al. 2018):
  not attempted. Nothing was substituted for a failed download; every
  dataset listed above downloaded successfully.
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
real files with checksums for the data a model would consume. As of this writing that
is SATISFIED: the Otari resources bundle is downloaded, its MD5 matches
Zenodo, its contents are extracted and checksummed above, and hg38 is
confirmed readable through pyfaidx. The splicing datasets in section 2
are downloaded and checksummed. The gate is open.

What remains before graphs can be built is not data but two decisions,
recorded in sections 2.3 and 2.1: which library the tier-0 topology
controls should run on, and how Definition 1's 300 nt intronic radius is
reconciled with a 170 nt assayed fragment.
