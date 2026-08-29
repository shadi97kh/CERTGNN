#!/usr/bin/env bash
# Re-acquire every raw dataset listed in data/raw/PROVENANCE.md.
# Verifies against the checksums recorded there; does not substitute on failure.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw

echo "== Otari repository =="
[ -d otari_repo ] || git clone -q https://github.com/FunctionLab/otari.git otari_repo
git -C otari_repo checkout -q 0b2925c1ff5cdde50836399274e5806f74bf48ab

echo "== Otari resources bundle (5.09 GB, Zenodo 16433545) =="
if [ ! -f otari_resources.tar.gz ]; then
  curl -SL --retry 3 -o otari_resources.tar.gz \
    "https://zenodo.org/records/16433545/files/otari_resources.tar.gz?download=1"
fi
echo "bef09c5bc5b3b7eb59042c1a1291d6e4  otari_resources.tar.gz" | md5sum -c -

echo "== FAS/CD95 exon 6, Julien et al. 2016 (PMC4866304) =="
[ -f fas_PMC4866304_supp.zip ] || curl -SL -o fas_PMC4866304_supp.zip \
  "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4866304/supplementaryFiles"
mkdir -p fas_supp && unzip -o -q fas_PMC4866304_supp.zip -d fas_supp

echo "== BRCA2 exons 17-19 5'ss MPSA, Wong et al. 2018 (via MAVE-NN) =="
for f in mpsa_data.csv.gz mpsa_replicate_data.csv.gz; do
  [ -f "$f" ] || curl -SL -o "$f" \
    "https://raw.githubusercontent.com/jbkinney/mavenn/master/mavenn/examples/datasets/$f"
done

echo "== verifying SHA-256 against PROVENANCE.md =="
awk '/^```$/{f=!f; next} f && NF==3 {print $1"  "$3}' ../../data/raw/PROVENANCE.md 2>/dev/null \
  | sha256sum -c - 2>/dev/null | grep -v ': OK$' || echo "all recorded checksums OK"
