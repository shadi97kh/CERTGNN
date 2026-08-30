# Gate 1: can measurement variance be estimated per ProteinGym assay?

Inventory only. No ceilings computed, no analysis. ProteinGym v1.1 (`substitutions_raw_DMS.zip` sha256 `6d83b165…`, `DMS_ProteinGym_substitutions.zip` sha256 `3a837662…`); v1.3 serves byte-identical files for both.

**The processed benchmark ships no reliability data at all.** Every one of the 217 processed assay files has the schema `mutant, mutated_sequence, DMS_score, DMS_score_bin`. Any sigma must come from the separately hosted raw assays, which are the original authors' tables and have no common schema.

| assay | variants | group | raw shipped | what is available | sigma computable |
|---|---:|---|---|---|---|
| A0A140D2T1_ZIKV_Sourisseau_2019 | 9,576 | OrganismalFitness | yes | point scores only | **NO** |
| A0A192B1T2_9HIV1_Haddox_2018 | 12,577 | OrganismalFitness | yes | point scores only | **NO** |
| A0A1I9GEU1_NEIME_Kennouche_2019 | 922 | Activity | yes | point scores only | **NO** |
| A0A247D711_LISMN_Stadelmann_2021 | 1,653 | Activity | yes | point scores only | **NO** |
| A0A2Z5U3Z0_9INFA_Doud_2016 | 10,715 | OrganismalFitness | yes | point scores only | **NO** |
| A0A2Z5U3Z0_9INFA_Wu_2014 | 2,350 | OrganismalFitness | yes | point scores only | **NO** |
| A4D664_9INFA_Soh_2019 | 14,421 | OrganismalFitness | yes | point scores only | **NO** |
| A4GRB6_PSEAI_Chen_2020 | 5,004 | OrganismalFitness | yes | point scores only | **NO** |
| A4_HUMAN_Seuma_2022 | 14,811 | Stability | yes | per-variant SE/SD (1) | **YES** |
| AACC1_PSEAI_Dandage_2018 | 1,801 | OrganismalFitness | yes | point scores only | **NO** |
| ACE2_HUMAN_Chan_2020 | 2,223 | Binding | yes | replicate scores (2) | **YES** |
| ADRB2_HUMAN_Jones_2020 | 7,800 | Activity | yes | point scores only | **NO** |
| AICDA_HUMAN_Gajula_2014_3cycles | 209 | Activity | yes | point scores only | **NO** |
| AMFR_HUMAN_Tsuboyama_2023_4G3O | 2,972 | Stability | yes | point scores only | **NO** |
| AMIE_PSEAE_Wrenbeck_2017 | 6,227 | Activity | yes | point scores only | **NO** |
| ANCSZ_Hobbs_2022 | 4,670 | Activity | yes | point scores only | **NO** |
| ARGR_ECOLI_Tsuboyama_2023_1AOY | 1,287 | Stability | yes | point scores only | **NO** |
| B2L11_HUMAN_Dutta_2010_binding-Mcl-1 | 170 | Binding | yes | point scores only | **NO** |
| BBC1_YEAST_Tsuboyama_2023_1TG0 | 2,069 | Stability | yes | point scores only | **NO** |
| BCHB_CHLTE_Tsuboyama_2023_2KRU | 1,572 | Stability | yes | point scores only | **NO** |
| BLAT_ECOLX_Deng_2012 | 4,996 | OrganismalFitness | yes | point scores only | **NO** |
| BLAT_ECOLX_Firnberg_2014 | 4,783 | OrganismalFitness | yes | point scores only | **NO** |
| BLAT_ECOLX_Jacquier_2013 | 989 | OrganismalFitness | yes | point scores only | **NO** |
| BLAT_ECOLX_Stiffler_2015 | 4,996 | OrganismalFitness | yes | point scores only | **NO** |
| BRCA1_HUMAN_Findlay_2018 | 1,837 | OrganismalFitness | yes | point scores only | **NO** |
| BRCA2_HUMAN_Erwood_2022_HEK293T | 265 | OrganismalFitness | yes | point scores only | **NO** |
| C6KNH7_9INFA_Lee_2018 | 10,754 | OrganismalFitness | yes | point scores only | **NO** |
| CALM1_HUMAN_Weile_2017 | 1,813 | OrganismalFitness | yes | point scores only | **NO** |
| CAPSD_AAV2S_Sinai_2021 | 42,328 | OrganismalFitness | yes | point scores only | **NO** |
| CAR11_HUMAN_Meitlis_2020_gof | 2,374 | OrganismalFitness | yes | point scores only | **NO** |
| CAR11_HUMAN_Meitlis_2020_lof | 2,395 | OrganismalFitness | yes | point scores only | **NO** |
| CAS9_STRP1_Spencer_2017_positive | 8,117 | Activity | yes | read counts (20), per-variant SE/SD (2) | **YES** |
| CASP3_HUMAN_Roychowdhury_2020 | 1,567 | Activity | yes | per-variant SE/SD (1) | **YES** |
| CASP7_HUMAN_Roychowdhury_2020 | 1,680 | Activity | yes | per-variant SE/SD (1) | **YES** |
| CATR_CHLRE_Tsuboyama_2023_2AMI | 1,903 | Stability | yes | point scores only | **NO** |
| CBPA2_HUMAN_Tsuboyama_2023_1O6X | 2,068 | Stability | yes | point scores only | **NO** |
| CBS_HUMAN_Sun_2020 | 7,217 | OrganismalFitness | yes | per-variant SE/SD (2) | **YES** |
| CBX4_HUMAN_Tsuboyama_2023_2K28 | 2,282 | Stability | yes | point scores only | **NO** |
| CCDB_ECOLI_Adkar_2012 | 1,176 | Activity | yes | point scores only | **NO** |
| CCDB_ECOLI_Tripathi_2016 | 1,663 | OrganismalFitness | yes | point scores only | **NO** |
| CCR5_HUMAN_Gill_2023 | 6,137 | Binding | yes | replicate scores (2) | **YES** |
| CD19_HUMAN_Klesmith_2019_FMC_singles | 3,761 | Binding | yes | read counts (5) | **YES** |
| CP2C9_HUMAN_Amorosi_2021_abundance | 6,370 | Expression | no | processed scores only | **NO** |
| CP2C9_HUMAN_Amorosi_2021_activity | 6,142 | Binding | no | processed scores only | **NO** |
| CSN4_MOUSE_Tsuboyama_2023_1UFM | 3,295 | Stability | yes | point scores only | **NO** |
| CUE1_YEAST_Tsuboyama_2023_2MYX | 1,580 | Stability | yes | point scores only | **NO** |
| D7PM05_CLYGR_Somermeyer_2022 | 24,515 | Activity | yes | replicate scores (2), read counts (1) | **YES** |
| DLG4_HUMAN_Faure_2021 | 6,976 | OrganismalFitness | yes | per-variant SE/SD (1) | **YES** |
| DLG4_RAT_McLaughlin_2012 | 1,576 | Binding | yes | point scores only | **NO** |
| DN7A_SACS2_Tsuboyama_2023_1JIC | 1,008 | Stability | yes | point scores only | **NO** |
| DNJA1_HUMAN_Tsuboyama_2023_2LO1 | 2,264 | Stability | yes | point scores only | **NO** |
| DOCK1_MOUSE_Tsuboyama_2023_2M0Y | 2,915 | Stability | yes | point scores only | **NO** |
| DYR_ECOLI_Nguyen_2023 | 2,916 | OrganismalFitness | yes | per-variant SE/SD (2) | **YES** |
| DYR_ECOLI_Thompson_2019 | 2,363 | OrganismalFitness | yes | per-variant SE/SD (1) | **YES** |
| ENVZ_ECOLI_Ghose_2023 | 1,121 | Activity | yes | point scores only | **NO** |
| ENV_HV1B9_DuenasDecamp_2016 | 375 | OrganismalFitness | yes | point scores only | **NO** |
| ENV_HV1BR_Haddox_2016 | 12,863 | OrganismalFitness | yes | point scores only | **NO** |
| EPHB2_HUMAN_Tsuboyama_2023_1F0M | 1,960 | Stability | yes | point scores only | **NO** |
| ERBB2_HUMAN_Elazar_2016 | 326 | Expression | yes | read counts (2) | **YES** |
| ESTA_BACSU_Nutschel_2020 | 2,172 | Stability | yes | per-variant SE/SD (2) | **YES** |
| F7YBW8_MESOW_Aakre_2015 | 9,192 | OrganismalFitness | yes | point scores only | **NO** |
| F7YBW8_MESOW_Ding_2023 | 7,922 | OrganismalFitness | yes | point scores only | **NO** |
| FECA_ECOLI_Tsuboyama_2023_2D1U | 1,886 | Stability | yes | point scores only | **NO** |
| FKBP3_HUMAN_Tsuboyama_2023_2KFV | 1,237 | Stability | yes | point scores only | **NO** |
| GAL4_YEAST_Kitzman_2015 | 1,195 | OrganismalFitness | yes | point scores only | **NO** |
| GCN4_YEAST_Staller_2018 | 2,638 | Binding | yes | replicate scores (4) | **YES** |
| GDIA_HUMAN_Silverstein_2021 | 1,154 | OrganismalFitness | yes | replicate scores (1), read counts (1) | **YES** |
| GFP_AEQVI_Sarkisyan_2016 | 51,714 | Activity | yes | point scores only | **NO** |
| GLPA_HUMAN_Elazar_2016 | 245 | Expression | yes | read counts (2) | **YES** |
| GRB2_HUMAN_Faure_2021 | 63,366 | OrganismalFitness | yes | per-variant SE/SD (1) | **YES** |
| HCP_LAMBD_Tsuboyama_2023_2L6Q | 1,040 | Stability | yes | point scores only | **NO** |
| HECD1_HUMAN_Tsuboyama_2023_3DKM | 5,586 | Stability | yes | point scores only | **NO** |
| HEM3_HUMAN_Loggerenberg_2023 | 5,689 | Activity | yes | per-variant SE/SD (1) | **YES** |
| HIS7_YEAST_Pokusaeva_2019 | 496,137 | OrganismalFitness | yes | point scores only | **NO** |
| HMDH_HUMAN_Jiang_2019 | 16,853 | OrganismalFitness | yes | per-variant SE/SD (2) | **YES** |
| HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2 | 2,252 | OrganismalFitness | yes | point scores only | **NO** |
| HSP82_YEAST_Flynn_2019 | 13,294 | OrganismalFitness | yes | replicate scores (2) | **YES** |
| HSP82_YEAST_Mishra_2016 | 4,323 | OrganismalFitness | yes | point scores only | **NO** |
| HXK4_HUMAN_Gersing_2022_activity | 8,570 | OrganismalFitness | yes | per-variant SE/SD (2) | **YES** |
| HXK4_HUMAN_Gersing_2023_abundance | 8,396 | Expression | yes | per-variant SE/SD (1) | **YES** |
| I6TAH8_I68A0_Doud_2015 | 9,462 | OrganismalFitness | yes | point scores only | **NO** |
| IF1_ECOLI_Kelsic_2016 | 1,367 | OrganismalFitness | yes | point scores only | **NO** |
| ILF3_HUMAN_Tsuboyama_2023_2L33 | 1,329 | Stability | yes | point scores only | **NO** |
| ISDH_STAAW_Tsuboyama_2023_2LHR | 1,944 | Stability | yes | point scores only | **NO** |
| KCNE1_HUMAN_Muhammad_2023_expression | 2,339 | Expression | yes | per-variant SE/SD (2) | **YES** |
| KCNE1_HUMAN_Muhammad_2023_function | 2,315 | Activity | yes | per-variant SE/SD (2) | **YES** |
| KCNH2_HUMAN_Kozek_2020 | 200 | Activity | yes | point scores only | **NO** |
| KCNJ2_MOUSE_Coyote-Maestas_2022_function | 6,963 | Activity | yes | per-variant SE/SD (1) | **YES** |
| KCNJ2_MOUSE_Coyote-Maestas_2022_surface | 6,917 | Expression | yes | per-variant SE/SD (1) | **YES** |
| KKA2_KLEPN_Melnikov_2014 | 4,960 | OrganismalFitness | yes | point scores only | **NO** |
| LGK_LIPST_Klesmith_2015 | 7,890 | Activity | yes | point scores only | **NO** |
| LYAM1_HUMAN_Elazar_2016 | 359 | Expression | yes | read counts (2) | **YES** |
| MAFG_MOUSE_Tsuboyama_2023_1K1V | 1,429 | Stability | yes | point scores only | **NO** |
| MBD11_ARATH_Tsuboyama_2023_6ACV | 2,116 | Stability | yes | point scores only | **NO** |
| MET_HUMAN_Estevam_2023 | 5,393 | Activity | yes | per-variant SE/SD (2) | **YES** |
| MK01_HUMAN_Brenan_2016 | 6,809 | OrganismalFitness | yes | point scores only | **NO** |
| MLAC_ECOLI_MacRae_2023 | 4,007 | OrganismalFitness | yes | point scores only | **NO** |
| MSH2_HUMAN_Jia_2020 | 16,749 | OrganismalFitness | yes | point scores only | **NO** |
| MTH3_HAEAE_RockahShmuel_2015 | 1,777 | OrganismalFitness | yes | point scores only | **NO** |
| MTHR_HUMAN_Weile_2021 | 12,464 | OrganismalFitness | yes | per-variant SE/SD (2) | **YES** |
| MYO3_YEAST_Tsuboyama_2023_2BTT | 3,297 | Stability | yes | point scores only | **NO** |
| NCAP_I34A1_Doud_2015 | 9,462 | OrganismalFitness | yes | point scores only | **NO** |
| NKX31_HUMAN_Tsuboyama_2023_2L9R | 2,482 | Stability | yes | point scores only | **NO** |
| NPC1_HUMAN_Erwood_2022_HEK293T | 637 | Activity | yes | point scores only | **NO** |
| NPC1_HUMAN_Erwood_2022_RPE1 | 63 | Activity | yes | point scores only | **NO** |
| NRAM_I33A0_Jiang_2016 | 298 | OrganismalFitness | yes | point scores only | **NO** |
| NUD15_HUMAN_Suiter_2020 | 2,844 | Expression | yes | per-variant SE/SD (2) | **YES** |
| NUSA_ECOLI_Tsuboyama_2023_1WCL | 2,028 | Stability | yes | point scores only | **NO** |
| NUSG_MYCTU_Tsuboyama_2023_2MI6 | 1,380 | Stability | yes | point scores only | **NO** |
| OBSCN_HUMAN_Tsuboyama_2023_1V1C | 3,197 | Stability | yes | point scores only | **NO** |
| ODP2_GEOSE_Tsuboyama_2023_1W4G | 1,134 | Stability | yes | point scores only | **NO** |
| OPSD_HUMAN_Wan_2019 | 165 | Expression | yes | per-variant SE/SD (1) | **YES** |
| OTC_HUMAN_Lo_2023 | 1,570 | Activity | yes | point scores only | **NO** |
| OTU7A_HUMAN_Tsuboyama_2023_2L2D | 635 | Stability | yes | point scores only | **NO** |
| OXDA_RHOTO_Vanella_2023_activity | 6,396 | Activity | yes | point scores only | **NO** |
| OXDA_RHOTO_Vanella_2023_expression | 6,769 | Expression | yes | point scores only | **NO** |
| P53_HUMAN_Giacomelli_2018_Null_Etoposide | 7,467 | OrganismalFitness | no | processed scores only | **NO** |
| P53_HUMAN_Giacomelli_2018_Null_Nutlin | 7,467 | OrganismalFitness | no | processed scores only | **NO** |
| P53_HUMAN_Giacomelli_2018_WT_Nutlin | 7,467 | OrganismalFitness | no | processed scores only | **NO** |
| P53_HUMAN_Kotler_2018 | 1,048 | OrganismalFitness | yes | point scores only | **NO** |
| P84126_THETH_Chan_2017 | 1,519 | OrganismalFitness | yes | point scores only | **NO** |
| PABP_YEAST_Melamed_2013 | 37,708 | OrganismalFitness | yes | point scores only | **NO** |
| PAI1_HUMAN_Huttinger_2021 | 5,345 | Activity | yes | point scores only | **NO** |
| PA_I34A1_Wu_2015 | 1,820 | OrganismalFitness | yes | point scores only | **NO** |
| PHOT_CHLRE_Chen_2023 | 167,529 | Activity | yes | replicate scores (3) | **YES** |
| PIN1_HUMAN_Tsuboyama_2023_1I6C | 802 | Stability | yes | point scores only | **NO** |
| PITX2_HUMAN_Tsuboyama_2023_2L7M | 1,824 | Stability | yes | point scores only | **NO** |
| PKN1_HUMAN_Tsuboyama_2023_1URF | 1,301 | Stability | yes | point scores only | **NO** |
| POLG_CXB3N_Mattenberger_2021 | 15,711 | OrganismalFitness | yes | point scores only | **NO** |
| POLG_DEN26_Suphatrakul_2023 | 16,897 | OrganismalFitness | yes | per-variant SE/SD (1) | **YES** |
| POLG_HCVJF_Qi_2014 | 1,630 | OrganismalFitness | yes | point scores only | **NO** |
| POLG_PESV_Tsuboyama_2023_2MXD | 5,130 | Stability | yes | point scores only | **NO** |
| PPARG_HUMAN_Majithia_2016 | 9,576 | Activity | yes | point scores only | **NO** |
| PPM1D_HUMAN_Miller_2022 | 7,889 | OrganismalFitness | yes | point scores only | **NO** |
| PR40A_HUMAN_Tsuboyama_2023_1UZC | 2,033 | Stability | yes | point scores only | **NO** |
| PRKN_HUMAN_Clausen_2023 | 8,756 | Expression | yes | per-variant SE/SD (1) | **YES** |
| PSAE_PICP2_Tsuboyama_2023_1PSE | 1,579 | Stability | yes | point scores only | **NO** |
| PTEN_HUMAN_Matreyek_2021 | 5,083 | Expression | yes | read counts (3) | **YES** |
| PTEN_HUMAN_Mighell_2018 | 7,260 | Activity | yes | point scores only | **NO** |
| Q2N0S5_9HIV1_Haddox_2018 | 12,729 | OrganismalFitness | yes | point scores only | **NO** |
| Q53Z42_HUMAN_McShan_2019_binding-TAPBPR | 3,344 | Binding | yes | point scores only | **NO** |
| Q53Z42_HUMAN_McShan_2019_expression | 3,344 | Expression | yes | point scores only | **NO** |
| Q59976_STRSQ_Romero_2015 | 2,999 | Activity | yes | point scores only | **NO** |
| Q6WV12_9MAXI_Somermeyer_2022 | 31,401 | Activity | yes | replicate scores (2), read counts (1) | **YES** |
| Q837P4_ENTFA_Meier_2023 | 697 | Activity | yes | point scores only | **NO** |
| Q837P5_ENTFA_Meier_2023 | 747 | Activity | yes | point scores only | **NO** |
| Q8WTC7_9CNID_Somermeyer_2022 | 33,510 | Activity | yes | replicate scores (2), read counts (1) | **YES** |
| R1AB_SARS2_Flynn_2022 | 5,725 | OrganismalFitness | yes | point scores only | **NO** |
| RAD_ANTMA_Tsuboyama_2023_2CJJ | 912 | Stability | yes | point scores only | **NO** |
| RAF1_HUMAN_Zinkus-Boltz_2019 | 297 | OrganismalFitness | yes | point scores only | **NO** |
| RASH_HUMAN_Bandaru_2017 | 3,134 | Activity | yes | point scores only | **NO** |
| RASK_HUMAN_Weng_2022_abundance | 26,012 | Expression | yes | per-variant SE/SD (1) | **YES** |
| RASK_HUMAN_Weng_2022_binding-DARPin_K55 | 24,873 | Binding | yes | per-variant SE/SD (1) | **YES** |
| RBP1_HUMAN_Tsuboyama_2023_2KWH | 1,332 | Stability | yes | point scores only | **NO** |
| RCD1_ARATH_Tsuboyama_2023_5OAO | 1,261 | Stability | yes | point scores only | **NO** |
| RCRO_LAMBD_Tsuboyama_2023_1ORC | 2,278 | Stability | yes | point scores only | **NO** |
| RD23A_HUMAN_Tsuboyama_2023_1IFY | 1,019 | Stability | yes | point scores only | **NO** |
| RDRP_I33A0_Li_2023 | 12,003 | OrganismalFitness | yes | point scores only | **NO** |
| REV_HV1H2_Fernandes_2016 | 2,147 | OrganismalFitness | yes | point scores only | **NO** |
| RFAH_ECOLI_Tsuboyama_2023_2LCL | 1,326 | Stability | yes | point scores only | **NO** |
| RL20_AQUAE_Tsuboyama_2023_1GYZ | 1,461 | Stability | yes | point scores only | **NO** |
| RL40A_YEAST_Mavor_2016 | 1,253 | OrganismalFitness | yes | point scores only | **NO** |
| RL40A_YEAST_Roscoe_2013 | 1,195 | OrganismalFitness | yes | point scores only | **NO** |
| RL40A_YEAST_Roscoe_2014 | 1,380 | Activity | yes | point scores only | **NO** |
| RNC_ECOLI_Weeks_2023 | 4,277 | Activity | yes | read counts (5), per-variant SE/SD (2) | **YES** |
| RPC1_BP434_Tsuboyama_2023_1R69 | 1,459 | Stability | yes | point scores only | **NO** |
| RPC1_LAMBD_Li_2019_high-expression | 351 | Activity | yes | point scores only | **NO** |
| RPC1_LAMBD_Li_2019_low-expression | 351 | Activity | yes | point scores only | **NO** |
| RS15_GEOSE_Tsuboyama_2023_1A32 | 1,195 | Stability | yes | point scores only | **NO** |
| S22A1_HUMAN_Yee_2023_abundance | 9,803 | Expression | yes | per-variant SE/SD (3) | **YES** |
| S22A1_HUMAN_Yee_2023_activity | 10,094 | Activity | yes | per-variant SE/SD (3) | **YES** |
| SAV1_MOUSE_Tsuboyama_2023_2YSB | 965 | Stability | yes | point scores only | **NO** |
| SBI_STAAM_Tsuboyama_2023_2JVG | 1,025 | Stability | yes | point scores only | **NO** |
| SC6A4_HUMAN_Young_2021 | 11,576 | Activity | yes | read counts (1) | **YES** |
| SCIN_STAAR_Tsuboyama_2023_2QFF | 1,212 | Stability | yes | point scores only | **NO** |
| SCN5A_HUMAN_Glazer_2019 | 224 | OrganismalFitness | yes | point scores only | **NO** |
| SDA_BACSU_Tsuboyama_2023_1PV0 | 2,770 | Stability | yes | point scores only | **NO** |
| SERC_HUMAN_Xie_2023 | 1,914 | OrganismalFitness | yes | per-variant SE/SD (1) | **YES** |
| SHOC2_HUMAN_Kwon_2022 | 10,972 | OrganismalFitness | yes | point scores only | **NO** |
| SOX30_HUMAN_Tsuboyama_2023_7JJK | 1,010 | Stability | yes | point scores only | **NO** |
| SPA_STAAU_Tsuboyama_2023_1LP1 | 2,105 | Stability | yes | point scores only | **NO** |
| SPG1_STRSG_Olson_2014 | 536,962 | Binding | yes | read counts (2) | **YES** |
| SPG1_STRSG_Wu_2016 | 149,360 | Binding | yes | read counts (2) | **YES** |
| SPG2_STRSG_Tsuboyama_2023_5UBS | 1,451 | Stability | yes | point scores only | **NO** |
| SPIKE_SARS2_Starr_2020_binding | 3,802 | Binding | no | processed scores only | **NO** |
| SPIKE_SARS2_Starr_2020_expression | 3,798 | Expression | no | processed scores only | **NO** |
| SPTN1_CHICK_Tsuboyama_2023_1TUD | 3,201 | Stability | yes | point scores only | **NO** |
| SQSTM_MOUSE_Tsuboyama_2023_2RRU | 707 | Stability | yes | point scores only | **NO** |
| SR43C_ARATH_Tsuboyama_2023_2N88 | 1,583 | Stability | yes | point scores only | **NO** |
| SRBS1_HUMAN_Tsuboyama_2023_2O2W | 1,556 | Stability | yes | point scores only | **NO** |
| SRC_HUMAN_Ahler_2019 | 3,372 | Activity | yes | per-variant SE/SD (2) | **YES** |
| SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM | 3,637 | Activity | yes | point scores only | **NO** |
| SRC_HUMAN_Nguyen_2022 | 3,366 | OrganismalFitness | yes | point scores only | **NO** |
| SUMO1_HUMAN_Weile_2017 | 1,700 | OrganismalFitness | yes | point scores only | **NO** |
| SYUA_HUMAN_Newberry_2020 | 2,497 | OrganismalFitness | yes | point scores only | **NO** |
| TADBP_HUMAN_Bolognesi_2019 | 1,196 | OrganismalFitness | yes | read counts (1), per-variant SE/SD (1) | **YES** |
| TAT_HV1BR_Fernandes_2016 | 1,577 | OrganismalFitness | yes | point scores only | **NO** |
| TCRG1_MOUSE_Tsuboyama_2023_1E0L | 1,058 | Stability | yes | point scores only | **NO** |
| THO1_YEAST_Tsuboyama_2023_2WQG | 1,279 | Stability | yes | point scores only | **NO** |
| TNKS2_HUMAN_Tsuboyama_2023_5JRT | 1,479 | Stability | yes | point scores only | **NO** |
| TPK1_HUMAN_Weile_2017 | 3,181 | OrganismalFitness | yes | point scores only | **NO** |
| TPMT_HUMAN_Matreyek_2018 | 3,648 | Expression | yes | point scores only | **NO** |
| TPOR_HUMAN_Bridgford_2020 | 562 | OrganismalFitness | yes | replicate scores (12), per-variant SE/SD (2) | **YES** |
| TRPC_SACS2_Chan_2017 | 1,519 | OrganismalFitness | yes | point scores only | **NO** |
| TRPC_THEMA_Chan_2017 | 1,519 | OrganismalFitness | yes | point scores only | **NO** |
| UBC9_HUMAN_Weile_2017 | 2,563 | OrganismalFitness | yes | point scores only | **NO** |
| UBE4B_HUMAN_Tsuboyama_2023_3L1X | 3,622 | Stability | yes | point scores only | **NO** |
| UBE4B_MOUSE_Starita_2013 | 899 | Activity | yes | point scores only | **NO** |
| UBR5_HUMAN_Tsuboyama_2023_1I2T | 1,453 | Stability | yes | point scores only | **NO** |
| VG08_BPP22_Tsuboyama_2023_2GP8 | 723 | Stability | yes | point scores only | **NO** |
| VILI_CHICK_Tsuboyama_2023_1YU5 | 2,568 | Stability | yes | point scores only | **NO** |
| VKOR1_HUMAN_Chiasson_2020_abundance | 2,695 | Expression | no | processed scores only | **NO** |
| VKOR1_HUMAN_Chiasson_2020_activity | 697 | Activity | no | processed scores only | **NO** |
| VRPI_BPT7_Tsuboyama_2023_2WNM | 1,047 | Stability | yes | point scores only | **NO** |
| YAIA_ECOLI_Tsuboyama_2023_2KVT | 1,890 | Stability | yes | point scores only | **NO** |
| YAP1_HUMAN_Araya_2012 | 10,075 | Binding | yes | point scores only | **NO** |
| YNZC_BACSU_Tsuboyama_2023_2JVD | 2,300 | Stability | yes | point scores only | **NO** |

## Headline

- **50 of 217 assays (23.0%)** admit a computable sigma.
- **1,287,922 of 2,465,767 variants (52.2%)**.
- 9 benchmark assays ship no raw file at all.

Coverage by `coarse_selection_type`, which is the unit ProteinGym actually aggregates over (each group weighted equally):

| group | usable | total | % |
|---|---:|---:|---:|
| OrganismalFitness | 14 | 77 | 18% |
| Stability | 2 | 66 | 3% |
| Activity | 15 | 43 | 35% |
| Expression | 12 | 18 | 67% |
| Binding | 7 | 13 | 54% |

## Verdict

**Below half. The direction closes.**

23.0% of assays admit a computable sigma, against the roughly-half threshold. The
52.2% variant figure does not rescue it and should not be quoted as if it did:
ProteinGym computes a metric **per assay** and averages within functional groups,
so an assay of 536,962 variants counts exactly as much as one of 200. The
relevant denominator is assays, and by variants the figure is carried by three
datasets alone -- SPG1_STRSG_Olson_2014 (536,962), PHOT_CHLRE_Chen_2023
(167,529) and SPG1_STRSG_Wu_2016 (149,360).

The group breakdown is worse than the headline and is what actually kills it.
Since each `coarse_selection_type` is weighted equally in the final score, every
group must be renormalizable for a leaderboard-wide correction. **Stability has 2
usable assays out of 66 (3%)** while contributing one fifth of the final metric.
A renormalized Stability average resting on 2 assays is not a correction, it is a
different benchmark.

Cross-referencing MaveDB cannot close the gap cheaply either. MaveDB does expose
per-score-set column metadata, and score sets carrying `sd`, `se` or count
columns exist. But ProteinGym's 46-column reference file contains no MaveDB URN,
DOI or PubMed identifier, so mapping the 167 gap assays would require manual
literature lookup per assay, and would still only recover assays whose depositors
uploaded uncertainty columns.

What remains possible is a **subset** audit on the 50 usable assays, which is a
different and much weaker claim than the one proposed: not "the leaderboard
reorders once corrected" but "on a 23% subset skewed toward Expression and
Binding and away from Stability, corrected and uncorrected rankings differ by X".
That subset is not representative of the benchmark it would be claiming to
correct.
