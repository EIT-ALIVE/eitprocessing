---
title: 'eitprocessing: a Python package for analysis of Electrical Impedance Tomography data'
tags:
  - Python
  - Electrical Impedance Tomography
authors:
  - name: Peter Somhorst
    orcid: 0000-0003-3490-2080
    equal-contrib: true
    affiliation: 1
  - name: Dani L. Bodor
    equal-contrib: true
    orcid: 0000-0003-2109-2349
    affiliation: 2
  - name: Juliette Francovich
    affiliation: 1
    orcid: 0009-0004-0976-5082
  - name: Jantine J. Wisse
    affiliation: 1
    orcid: 0009-0006-6552-5459
  - name: Walter Baccinelli
    affiliation: 2
    orcid: 0000-0001-8888-4792
  - name: Annemijn H. Jonkman
    orcid: 0000-0001-8778-5135
    affiliation: 1

affiliations:
 - name: Department of Adult Intensive Care, Erasmus MC, Rotterdam, The Netherlands
   index: 1
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 2
date: 13 January 2025
bibliography: paper.bib
---

## Summary

Electrical Impedance Tomography (EIT) is a promising non-invasive, radiation-free technology for
monitoring the respiratory system. EIT is mostly used to optimize ventilator settings to the respiratory mechanics of
mechanically ventilated patients in the Intensive
Care Unit. While EIT is gaining popularity, the complexity of data processing, analysis and
interpretation hampers standardization, validation and widespread adoption. Commercial software is
closed and opaque, while custom research software is often ad-hoc, single use, and unverified.
`eitprocessing` offers a standardized, open, and highly expandable pipeline for the processing and
analysis of EIT and respiration related data.

## State of the field

Acute respiratory failure is the most common reason for admission to the intensive care unit (ICU),
and can be caused by e.g., infection, trauma, heart failure, or complications during elective
surgery. Patients with severely injured lungs and critically low levels of arterial oxygen require
life-saving breathing support with mechanical ventilation [@tobin1998]. Although mechanical
ventilation is the cornerstone of supportive therapy in the ICU, it is a double-edged sword:
inadequate mechanical ventilator assist exacerbates lung injury and inflammation, and worsens
outcomes [@Slutsky2013;@Amato2015]. ICU mortality for patients with acute respiratory failure
remains high [~40%, @Bellani2016]; these numbers increased drastically during the COVID-19
pandemic. To ameliorate the risk of death and long-term morbidity of the critically ill, we need
mechanical ventilation strategies that are lung-protective and tailored to the individual patient’s
respiratory physiology [@Goligher2020;@Goligher2020b]. However, there are currently no simple,
reliable, and readily accessible tools available to clinicians at the bedside to identify the
beneficial and harmful effects of adaptations in mechanical ventilator support [@Jonkman2022].

A very promising technology to change clinical practice in ICU patients is EIT [@Frerichs2016]. EIT is gaining worldwide popularity as a bedside non-invasive
radiation-free tool for lung imaging. Using a belt fitted with electrodes placed around the chest, it
continuously visualizes real-time changes in lung volume. These changes reflect tidal ventilation,
changes in lung volume due to ventilator settings, and adaptations due to variations in lung
characteristics caused by improved or worsening lung mechanics. In contrast to static anatomical imaging techniques such as computed tomography
scan, EIT
provides dynamic information on lung ventilation. As such, EIT can monitor at the bedside the
direct impact of mechanical ventilation on the lung, help with personalizing mechanical
ventilation, and assist in clinical decision-making.
Personalizing mechanical ventilation using EIT monitoring and diagnostics may ameliorate the risk
of death and long-term morbidity, and may substantially reduce the burden on our healthcare system.

## Statement of need

The perspective that EIT will become an important standard monitoring technique is shared by
international experts [@Frerichs2016;@Wisse2024-sl]. Both @Frerichs2016 and @Wisse2024-sl emphasize
the importance of standardized techniques, terminology, and consensus regarding the application of
EIT.
Validated methods to implement EIT-based parameters in routine care are
still lacking. Standardized implementation of EIT-based parameters is further limited
as the availability of both bedside and offline analysis tools depends on the type of EIT device used. Advanced
image and signal analysis could overcome certain challenges but also requires complex
post-processing (including detection/removal of common artifacts) that is time-consuming and requires specific technical expertise
that is often not present in clinical practice. This currently hampers reproducibility of research
findings and clinical implementation. The current limitations of EIT analysis stresses the importance of close collaboration
between physicians, clinical researchers and engineers in order to identify clinical needs, to
develop and validate new algorithms, and to facilitate clinical implementation [@Scaramuzzo2024-ob].

Currently, some open source EIT software packages are available [@liu2018pyeit;@EIDORS2005]. These, however, all focus on reconstruction of
voltage data to images, bypassing the clinically used reconstruction algorithms implemented in CE-approved devices, and
don't include tools for the analysis of reconstructed EIT image data.

`eitprocessing` offers a standardized, open, and highly expandable library of tools for loading,
filtering, segmentation and analysis of reconstructed EIT data as well as related waveform or sparse data.
`eitprocessing` is compatible with data from the three most-used clinically available EIT devices, as well as
from related data sources, such as mechanical ventilators and dedicated pressure devices. It includes commonly used methods for filtering and segmentation. The
authors continuously develop and implement further algorithms for analysis. The
international community has been invited to use and contribute to the software.

## Key features

`eitprocessing` aims to simplify and standardize loading, pre-processing, analysis and reporting
of respiration-related datasets.
Notebooks demonstrating these features are available in the repository.

### Loading

`eitprocessing` supports the loading of EIT data exported from the Dräger Pulmovista (`.bin`
files), Timpel Enlight (`.txt` files) and Sentec LuMon (`.zri` files) devices. Non-EIT data saved
in the data files are also loaded.

### Data containers

The main data container in `eitprocessing` is the `Sequence`. A sequence represents a single
continuous measurement of data in a single subject, and can contain data from different sources. Sequences
can be sliced --- by time or index --- and concatenated. All data contained in the sequence are
sliced and concatenated accordingly.

`eitprocessing` currently supports four types of dataset. The most important type is `EITData`,
which contains the electrical impedance of individual pixels as three-dimensional data --- (generally)
32 rows by 32 columns over time. Each frame of 32 by 32 pixels represents the impedance in a
transverse plane through the thorax at the corresponding time. `ContinuousData` has one-dimensional data points at
predictable intervals with a fixed sample frequency. Examples are airway pressure measured by a
mechanical ventilator or a global impedance signal.  `SparseData` has one-dimensional data points at
unpredictable intervals and no set sample frequency. An example is the tidal volume measured by a
mechanical ventilator, registered at the end of each breath. `IntervalData` has one-dimensional data
points that are valid for a time interval. An example is the position of a subject, e.g., supine
for the first part of a measurement and prone for the second part.

### Pre-processing

`eitprocessing` currently has implementations for the following pre-processing steps:

- high-pass, low-pass, band-pass or band-stop Butterworth filters, as well as a Multiple Digital Notch filter [@Wisse2024-wi];
- calculation of the global or regional impedance as the sum of the impedance of all or a subset of pixels;
- a moving averager using convolution with a given window;
- region of interest selection using predefined or custom masks;
- functional lung space detection using the tidal impedance variation, amplitude, or the Watershed method;
- automatic detection of the start, middle (end-inspiration) and end of breaths on a
  global/regional and pixel level;
- automatic detection of the respiratory and heart rate from pixel impedance values.

### Analysis

`eitprocessing` currently has implementations for the following parameters:

- end-expiratory lung impedance on a global/regional and pixel level;
- tidal impedance variation on a global/regional and pixel level.

## Visualization

`eitprocessing` includes several visualization methods to simplify and standardize visual output. Examples are:

- showing pixel maps, e.g., tidal impedance variation, changes in EELI, pendelluft, etc.;
- show the effect of filtering methods in the frequency domain.

## Future perspective

`eitprocessing` is ready for use in offline analysis of EIT and respiratory related data. Our team
is actively working on expanding the features of the software.

Several features are in active development. Examples are:

- provenance tracking of data processing steps;
- more advanced filtering methods, e.g., empirical mode
  decomposition and wavelet transforms;
- quantification of pendelluft;
- expansion of visualization methods.

Moreover, we plan to extend `eitprocessing` with standardized workflows to summarize and report
analysis results.

## References
