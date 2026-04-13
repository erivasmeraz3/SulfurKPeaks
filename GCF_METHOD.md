# Gaussian Curve Fitting (GCF) Method for S K-edge XANES

## Overview

SulfurKPeaks implements the convergence-optimized Gaussian Curve Fitting (GCF)
method for quantifying sulfur functional groups in natural organic matter from
sulfur K-edge X-ray Absorption Near Edge Structure (XANES) spectroscopy.

The method decomposes an S K-edge spectrum into six Gaussian functions
representing distinct sulfur functionalities and two arctangent step functions
representing the continuum absorption edge. After spectral decomposition,
Gaussian areas are corrected for the energy-dependent X-ray absorption
cross-section to obtain the true fractional amounts of each sulfur species.

**Primary reference:**

Manceau A. & Nagy K.L. (2012) "Quantitative analysis of sulfur functional
groups in natural organic matter by XANES spectroscopy." *Geochimica et
Cosmochimica Acta* 99, 206-223.
doi:[10.1016/j.gca.2012.09.033](https://doi.org/10.1016/j.gca.2012.09.033)


## The Spectral Model: 6 Gaussians + 2 Arctangents

### Peak Assignments

Six types of sulfur functionality are determinable from S-XANES spectra of
natural organic matter (Vairavamurthy, 1998; Manceau & Nagy, 2012, Section 1):

| Peak | Energy (eV) | Assignment       | EOS    | Allowed shift |
|------|-------------|------------------|--------|---------------|
| 1    | 2473.2      | Exocyclic S      | ~+0.2  | -0.2 eV       |
| 2    | 2474.4      | Heterocyclic S   | ~+0.7  | +0.3 eV       |
| 3    | 2476.4      | Sulfoxide        | ~+2.0  | Fixed          |
| 4    | 2479.6      | Sulfone          | ~+4.0  | Fixed          |
| 5    | 2481.3      | Sulfonate        | ~+5.0  | Fixed          |
| 6    | 2482.75     | Sulfate          | ~+6.0  | Fixed          |

Energy calibration: elemental sulfur absorption maximum at 2472.70 eV,
inflection point at 2472.0 eV (Manceau & Nagy, 2012, Section 2.2). The
uncertainty in energy is 0.10 eV.

**Exocyclic sulfur** (EOS ~0 to +0.5) includes thiols (R-SH), thioethers
(R-S-R'), and disulfides (R-S-S-R'), which are the dominant forms of reduced
sulfur in most humic substances. Cysteine, cystine, and methionine are
representative compounds.

**Heterocyclic sulfur** (EOS ~+0.6 to +0.8) includes thiophenic structures
where sulfur is incorporated into an aromatic ring. Benzo[b]thiophene and
dibenzothiophene are representative compounds.

Note: Several constitutive exocyclic moieties (e.g., cystine) exhibit split
white lines that the GCF method decomposes as a mixture of exocyclic and
heterocyclic S. This causes an artificial increase in the heterocyclic/exocyclic
ratio, especially when heterocyclic S is minor (Manceau & Nagy, 2012,
Section 4). This is an inherent limitation of the GCF approach.

### FWHM Constraints

The widths of all Gaussian functions are **covaried** (constrained to share
one or two common values), which is critical for reducing parameter
correlations and avoiding physically unrealistic decompositions (Manceau &
Nagy, 2012, Section 3.1.2).

Two FWHM groups are defined:

- **Reduced group** (peaks 1-3: Exocyclic, Heterocyclic, Sulfoxide) — shared
  FWHM, denoted `red_fwhm`
- **Oxidized group** (peaks 4-6: Sulfone, Sulfonate, Sulfate) — shared FWHM,
  denoted `ox_fwhm`

The grouping into reduced and oxidized reflects the distinct electronic
environments: low-oxidation-state species have narrower peaks due to fewer
coordinating oxygens, while high-oxidation-state species have broader peaks
from more extensive bonding to oxygen (Manceau & Nagy, 2012, Section 3.1.1).

In practice, most humic substance spectra can be fit with a **single FWHM**
for all six peaks. Six of eight IHSS reference materials used a single FWHM
(range 2.12-2.39 eV). Only Elliott Soil HA and FA required two groups
(reduced 1.55-1.75 eV, oxidized 1.94-2.05 eV) due to their distinct speciation
(Manceau & Nagy, 2012, Table EA-4).

The theoretical validation spectrum in Table EA-2 used FWHM = 1.60 eV for all
peaks.

### Peak Center Constraints

The convergence-optimized procedure (Manceau & Nagy, 2012, Section 3.1.2)
fixes the positions of peaks 3-6 (Sulfoxide, Sulfone, Sulfonate, Sulfate) at
their nominal energies. This eliminates unnecessary degrees of freedom without
significantly affecting the goodness of fit (NSS). Only two peaks are allowed
to shift:

- **Exocyclic**: shifts left by at most -0.2 eV
- **Heterocyclic**: shifts right by at most +0.3 eV

When all six peak positions were varied independently (as in Xia et al., 1998),
the exocyclic fraction was overestimated by 23% and heterocyclic underestimated
by 32%, at least partly because the widths of the two reduced components were
also varied independently (Manceau & Nagy, 2012, Section 3.1.3).


## Baseline: Double Arctangent

### Physical Basis

Each sulfur functionality produces both a Gaussian white-line peak (the
1s -> 3p electronic transition) and an arctangent step function (the 1s ->
continuum transition). In the theoretical model, the true spectrum is
6G + 6A — six Gaussians and six individual arctangent steps.

The fitting model approximates these six step functions with only **two
arctangents** (6G + 2A). Section 3.1.1 of Manceau & Nagy (2012) demonstrates
that this approximation recovers the correct sulfur fractions within 0.5% when
the arctangent positions are properly constrained (Model-fit 1 in Table EA-2,
NSS = 7.3 x 10^-6).

### Position Constraints

The arctangent positions are the most critical constraints in the GCF model.
Incorrect placement is the main source of systematic error (Manceau & Nagy,
2012, Sections 3.1.1-3.1.2):

**First arctangent (A1):** Must be placed **between the heterocyclic and
sulfoxide peaks** (2474.4 < A1 < 2476.4 eV). If A1 is placed to the right
of sulfoxide (Model-fit 2 in Table EA-2), the sulfoxide fraction doubles
from 5.0% to 10.9%. The fitted range across IHSS samples is 2475.51-2475.85 eV
(Table EA-4).

**Second arctangent (A2):** Position depends on the sulfate/sulfonate ratio:

- When sulfate > sulfonate: A2 above sulfate (A2 > 2482.75 eV)
- When sulfonate > sulfate: A2 between sulfonate and sulfate
  (2481.3 < A2 < 2482.75 eV)

The fitted range across IHSS samples is 2482.30-2484.40 eV (Table EA-4).
The lowest value (2482.30, ES-HA) corresponds to a sulfonate-dominant sample
(24.3% sulfonate vs. 17.8% sulfate).

In several previous studies, the first arctangent was placed either below or
above the correct range (Huffman et al., 1991, 1995; Taghiei et al., 1992;
Skyllberg et al., 2000; Szulczewski et al., 2001; Martinez et al., 2002;
Solomon et al., 2003; Wiltfong et al., 2005; Pattanaik et al., 2007; Huggins
et al., 2009), leading to erroneous results.

### Width Constraints

Both arctangent widths are **covaried** (constrained equal to each other). The
fitted width across IHSS samples ranges from 0.42 to 1.01 eV FWHM (Table EA-4).

Fixing the width at an inappropriate value degrades the fit: Model-fit 4 in
Table EA-2 held the width at half the optimal value (0.62 vs. 1.25 eV FWHM),
increasing NSS by 5x to 3.7 x 10^-5. Section 3.1.1 concludes that the widths
should be allowed to vary freely, subject to the covariance constraint.


## Cross-Section Correction (Scaling Factor)

### Why Correction Is Needed

The fractions of different sulfur functionalities are **not** directly
proportional to the Gaussian areas. The probability of the 1s -> 3p transition
(the absorption cross-section) increases approximately linearly with the
density of low-lying unoccupied states, and thus with oxidation state (Waldo
et al., 1991). Without correction, oxidized species (sulfate, sulfonate)
would be systematically overestimated and reduced species (exocyclic,
heterocyclic) underestimated.

The correction is applied as:

    corrected_area_i = raw_gaussian_area_i / scaling_factor(E_i)

where the scaling factor is an energy-dependent function derived from
calibration measurements on pure reference compounds.

### Published Calibration Curves

Five calibration curves have been published, with slopes differing by nearly
fivefold (Manceau & Nagy, 2012, Section 3.1.3, Fig. 2):

| Curve                      | Slope  | Detection | Notes                          |
|----------------------------|--------|-----------|--------------------------------|
| Huffman et al., 1991 (#1)  | Steepest | —       | Highest slope                  |
| Manceau & Nagy, 2012 (#2)  | High   | TEY       | This study                     |
| Sarret et al., 1999 (#4)   | Medium | TEY       | —                              |
| Prietzel et al., 2011 (#5) | Lower  | Fluorescence | Likely affected by overabsorption |
| Waldo et al., 1991 (#6)    | Lowest | Fluorescence | Reproduced by Xia et al. (1998) |

The Orthous-Daunay et al. (2010) curve (#7) was omitted because the data were
recorded in fluorescence-yield detection mode without correcting for
overabsorption.

### Why the Curves Differ

The primary reason for the fivefold variation in slope is **overabsorption**
(also called self-absorption). This is a systematic measurement artifact that
occurs in fluorescence-yield detection when:

1. **Sulfur concentration is high** (>~20,000 ppm in an organic matrix) and
   homogeneously distributed (Manceau & Nagy, 2012, Section 3.6.1), or
2. **Sulfur is nanoparticulate** — even at low bulk concentrations, intense
   fluorescence from concentrated grains is suppressed non-linearly (Section
   3.6.2). A 0.1 um FeS2 grain causes 10% signal attenuation. Pickering et
   al. (2001) calculated 17% attenuation for 0.2 um spheres of elemental
   sulfur.

Overabsorption distorts peak amplitudes non-linearly: peaks with normalized
absorbance less than 1 are enhanced, while those greater than 1 are attenuated
(Manceau and Gates, 1997; Manceau et al., 2002). This flattens the calibration
slope because high-energy (oxidized) peaks are preferentially suppressed.

The Waldo and Prietzel curves have lower slopes than the Sarret curve and the
curve from this study, likely in part because they were measured in
fluorescence-yield detection mode instead of total electron yield (TEY)
(Manceau & Nagy, 2012, Section 3.1.3).

### Impact on Quantification

A lower calibration slope overestimates oxidized species and underestimates
reduced species. Comparing the two most extreme curves (Huffman vs. Waldo)
across the eight IHSS samples (Table EA-4):

- Exocyclic S fraction differs by **7.3 +/- 1.3%** of total S
- Sulfate fraction differs by **3.8 +/- 1.6%** of total S
- Sulfoxide is essentially independent of the calibration curve

Functionalities with intermediate oxidation states (sulfoxide, sulfone) are
less sensitive to calibration curve choice because their scaling factors are
closer to the normalization point.

### The Generic Curve (Default)

To minimize uncertainty, Manceau & Nagy (2012) recommend calibrating with the
**mean of the three steepest curves** (Huffman, this study, Sarret), which are
least affected by overabsorption. This yields the generic curve:

    scaling_factor(E) = 0.36841 * E - 909.97

normalized to 1.0 at elemental sulfur (E = 2472.70 eV).

The confidence interval of this calibration procedure is estimated as:

    delta_Si = |Si(Huffman) - Si(Sarret)| / 2

This gives a mean precision of 1.6 +/- 0.2 atom% for exocyclic S and
0.7 +/- 0.3 atom% for sulfate across the eight IHSS samples.

### Using a Custom Calibration Curve

Users who measure their own reference compounds can derive a custom calibration
curve using the `derive_calibration_curve()` function in the fitting engine.
The procedure is:

1. Fit each pure standard compound with the S K-edge model
2. Record the dominant peak's fitted energy and raw Gaussian area
3. Pass those measurements to `derive_calibration_curve()`
4. Use the returned slope/intercept in `calculate_peak_areas()`

A minimum of 2 standards spanning a range of oxidation states is required;
4 or more are recommended. Standards should be measured in **total electron
yield (TEY)** mode to avoid overabsorption, or corrected for overabsorption
if measured in fluorescence yield.


## Equivalent Solutions and the Underdetermined Problem

The GCF decomposition is mathematically **ill-posed** — the regression has
more parameters than can be uniquely constrained by the data. Table EA-3
demonstrates this with six statistically equivalent fits to the ES-HA spectrum,
all with similar NSS (~2-3 x 10^-4) but substantially different sulfur
fractions.

The convergence-optimized constraints (fixed peak positions, covaried widths,
constrained arctangent placement) are designed to select the physically
realistic solution from among the family of equivalent mathematical solutions.

Key observations from the six ES-HA model-fits (Manceau & Nagy, 2012,
Section 3.1.1, Fig. 3, Table EA-3):

- **Model-fit 1** (convergence-optimized): A1 below sulfoxide, peaks 3-6
  fixed, two FWHM groups. Gives results consistent with the independent LCF
  method.

- **Model-fit 2**: A1 placed to the right of sulfoxide, all peak positions
  optimized. Despite similar NSS (1.9 x 10^-4 vs. 2.8 x 10^-4), the sulfoxide
  fraction changes dramatically.

- **Model-fit 4**: Arctangent FWHM fixed at 2x the optimal width (1.0 vs.
  0.42 eV), heights fixed at optimal values. Arrows in Fig. 3c-e point to
  regions of significant fit error in the mid-energy range.

- **Model-fit 5**: A1 height fixed at 0.50 (instead of 0.69). Despite having
  the lowest NSS (1.6 x 10^-4), the decomposition is physically unrealistic.

The lesson: **goodness-of-fit alone cannot distinguish correct from incorrect
decompositions**. The physical constraints are essential.


## When GCF Fails: Dominant Reduced Sulfur Species

The GCF method requires that no single reduced sulfur species dominates the
spectrum. When one or two particular reduced species prevail, their broad
post-edge oscillations overlap with the arctangent baseline and the Gaussians
of intermediate species (sulfoxide, sulfone), producing spurious peaks
(Manceau & Nagy, 2012, Section 3.5).

Examples:

- **Protein-rich humic material**: Dominated by cysteine/cystine. GCF
  gives 52.7% exocyclic + 30.5% heterocyclic; LCF gives 83.0% exocyclic +
  4.2% heterocyclic. However, the total reduced sulfur agrees within 5%.

- **Carbonaceous chondrite**: Dominated by pyrrhotite. The asymmetric
  main peak requires two Gaussians, showing the fundamental limit of the
  single-Gaussian-per-species assumption.

Diagnostic indicators of this failure mode include: an upward slope between
2476-2480 eV, anomalously large sulfoxide/sulfone fractions, and a poor
match between GCF and LCF reduced sulfur partitioning.

For these sample types, the Linear Combination Fitting (LCF) method should
be used instead.


## Overabsorption

### Detection

Overabsorption in fluorescence-yield data can be difficult to detect.
Comparison between transmission and fluorescence spectra is **not** a reliable
test because hole effect in transmission distorts the signal in the same way
as overabsorption in fluorescence (Stern and Kim, 1981; Manceau et al., 2002;
Manceau & Nagy, 2012, Section 3.6.3).

The recommended approach is to measure micro-XANES spectra of several tiny
particles and compare the normalized amplitudes of their peak maxima. If the
amplitudes are consistent across particles of different size, overabsorption is
negligible (Section 3.6.2).

### Empirical Correction

An empirical correction function for overabsorbed data is given by Eq. (28) of
Manceau et al. (2002) and applied in Manceau & Nagy (2012, Section 3.6.3):

    y_OA = y / (1 - beta + beta * y)

where y_OA is the measured (distorted) spectrum, y is the true spectrum, and
beta is a dimensionless parameter measuring the strength of the overabsorption
effect. The parameter beta depends on the absorption coefficients below the
edge, above the edge, and at the fluorescence energy, as well as the incidence
and takeoff angles.

This correction can be applied to sample spectra affected by overabsorption.
However, Manceau & Nagy (2012) caution that "it is preferable to avoid or at
least minimize these problems at the sample preparation and measurement stages"
rather than relying on post-hoc correction.

### Impact on Calibration Curves

Using a calibration curve derived from overabsorbed reference spectra
propagates the error into all quantification results. The effect is systematic:
overabsorption flattens the calibration slope, causing oxidized species to be
overestimated and reduced species underestimated. This is believed to explain
the lower slopes of the Waldo et al. (1991) and Prietzel et al. (2011) curves
(Section 3.1.3).


## Summary of Constraints Implemented in SulfurKPeaks

The following constraints are applied by default, matching the
convergence-optimized procedure of Manceau & Nagy (2012, Section 3.1.2):

| Parameter | Constraint | Reference |
|-----------|-----------|-----------|
| Peak centers 3-6 | Fixed at nominal | Section 3.1.2 |
| Peak center 1 (Exocyclic) | Varies up to -0.2 eV | Section 3.1.2 |
| Peak center 2 (Heterocyclic) | Varies up to +0.3 eV | Section 3.1.2 |
| Gaussian FWHM | Covaried in 1 or 2 groups | Sections 3.1.1-3.1.2 |
| A1 position | Between heterocyclic and sulfoxide | Section 3.1.2 |
| A2 position | Above sulfate (or between sulfonate/sulfate) | Section 3.1.2 |
| A1, A2 widths | Covaried (constrained equal) | Section 3.1.1 |
| Cross-section correction | Generic calibration curve | Section 3.1.3 |
| Energy range | 2466-2489 eV | Section 2.2 |

All constraints can be adjusted in the Peak Configuration tab of the GUI.


## References

Huffman G.P., Mitra S., Huggins F.E., Shah N., Vaidya S. and Lu F. (1991)
Quantitative analysis of all major forms of sulfur in coal by X-ray absorption
fine structure spectroscopy. *Energy Fuels* 5, 574-581.

Manceau A. and Gates W.P. (1997) Surface structural model for ferrihydrite.
*Clays Clay Miner.* 45, 448-460.

Manceau A., Marcus M.A. and Tamura N. (2002) Quantitative speciation of heavy
metals in soils and sediments by synchrotron X-ray techniques. In:
*Applications of Synchrotron Radiation in Low-Temperature Geochemistry and
Environmental Science* (eds. P.A. Fenter, M.L. Rivers, N.C. Sturchio and S.R.
Sutton). Reviews in Mineralogy and Geochemistry 49, pp. 341-428.

Manceau A. and Nagy K.L. (2012) Quantitative analysis of sulfur functional
groups in natural organic matter by XANES spectroscopy. *Geochim. Cosmochim.
Acta* 99, 206-223.

Orthous-Daunay F.R., Quirico E., Lemelle L., Beck P., deAndrade V., Simionovici
A. and Derenne S. (2010) Speciation of sulfur in the insoluble organic matter
from carbonaceous chondrites by XANES spectroscopy. *Earth Planet. Sci. Lett.*
300, 321-328.

Pickering I.J., Prince R.C., Diber T., Tober G., George G.N. and Watt G.D.
(2001) Sulfur K-edge X-ray absorption spectroscopy for determining the chemical
speciation of sulfur in biological systems. *Federation Proc.* 15, A493.

Prietzel J., Botzaki A., Tyufekchieva N., Kalbe M., Schwerin J. and Stemmler
S.J. (2011) Sulfur speciation in soil by S K-edge XANES spectroscopy:
comparison of spectral deconvolution and linear combination fitting. *Environ.
Sci. Technol.* 45, 2878-2886.

Sarret G., Connan J., Kasrai M., Bancroft G.M., Charrie-Duhaut A., Lemoine S.,
Adam P., Albrecht P. and Eybert-Berard L. (1999) Chemical forms of sulfur in
geological and archeological asphaltenes from Middle East, France, and Spain
determined by sulfur K- and L-edge X-ray absorption near edge structure
spectroscopy. *Geochim. Cosmochim. Acta* 63, 3767-3779.

Vairavamurthy A. (1998) Using X-ray absorption to probe sulfur oxidation
states in complex molecules. *Spectrochim. Acta Part A* 54, 2009-2017.

Waldo G.S., Carlson R.M.K., Moldowan J.M., Peters K.E. and Penner-Hahn J.E.
(1991) Sulfur speciation in heavy petroleums: information from X-ray absorption
near-edge structure. *Geochim. Cosmochim. Acta* 55, 801-814.

Xia K., Weesner F., Bleam W., Bloom P.R., Skyllberg U. and Helmke P.A. (1998)
XANES studies of oxidation states of sulfur in aquatic and soil humic
substances. *Soil Sci. Soc. Am. J.* 62, 1240-1246.
