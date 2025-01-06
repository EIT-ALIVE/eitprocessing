---
title: 'eitprocessing: a Python package for analysis of Electrial Impedance Tomography data'
tags:
  - Python
  - Electrical Impedance Tomography
authors:
  - name: Peter Somhorst
    orcid: 0000-0003-3490-2080
    equal-contrib: true
    affiliation: 1
  - name: Dani Bodor
    equal-contrib: true
    orcid: 0000-0003-2109-2349
    affiliation: 2
  - name: Juliette Francovich
    affiliation: 1
    orcid: 0000-0000-0000-0000
  - name: Jantine Wisse-Smit
    affiliation: 1
    orcid: 0000-0000-0000-0000
  - name: Walter Baccinelli
    affiliation: 2
    orcid: 0000-0001-8888-4792
  - name: Annemijn Jonkman
    orcid: 0000-0001-8778-5135
    affiliation: 1
  
affiliations:
 - name: Department of Intensive Care adults, Erasmus MC, Rotterdam, The Netherlands
   index: 1
 - name: Netherlands eScience Center, Amsterdam, The Netherlands 
   index: 2
date: 6 January 2025
bibliography: paper.bib
---

## Summary

Electrical Impedance Tomography (EIT) is a promising non-invasive, radiation-free technology for
monitoring the respiratory system of patients who undergo mechanical ventilation in the Intensive
Care Unit. While EIT is gaining popularity, the complexity of data processing, analysis and
interpretation hamper standardization, validation and widespread adoption. Commercial software is
closed and opaque, while custom research software is often ad-hoc, single use and unverified.
`eitprocessing` offers a standardized, open, and highly expandable pipeline for the processing and
analysis of EIT and related data.

## Statement of need

Acute respiratory failure is the most common reason for admission to the intensive care unit (ICU),
and can be caused by e.g., infection, trauma, heart failure, or complications during elective
surgery. Patients with severely injured lungs and critically low levels of arterial oxygen require
life-saving breathing support with mechanical ventilation [@tobin1998]. Although mechanical
ventilation is the cornerstone of supportive therapy in the ICU, it is a double-edge sword:
inadequate mechanical ventilator assist exacerbates lung injury and inflammation, and worsens
outcomes [@Slutsky2013;@Amato2015]. ICU mortality for patients with acute respiratory failure
remains high [~40%, @Bellani2016]; these numbers increased drastically during the COVID-19
pandemic. To ameliorate the risk of death and long-term morbidity of the critically ill, we need
mechanical ventilation strategies that are lung-protective and tailored to the individual patient’s
respiratory physiology [@Goligher2020;@Goligher2020b]. However, there are yet no simple and
reliable and readily accessible techniques available to clinicians at the bedside to identify the
beneficial and harmful effects of adaptations in mechanical ventilator support [@Jonkman2022].

A very promising technology to change clinical practice in ICU patients is Electrical Impedance
Tomography (EIT) [@Frerichs2016]. EIT is gaining popularity worldwide as a bedside non-invasive
radiation-free lung imaging tool: using a belt mounted with electrodes placed around the chest, it
continuously and real-time visualizes changes in lung volume owing to adaptations in ventilator
pressures or because of changes in lung characteristics, resulting from worsening or improving lung
function. In contrast to static anatomical imaging techniques such as computed tomography scan, EIT
provides dynamic information on lung ventilation. As such, EIT can monitor at the bedside the
direct impact of mechanical ventilation on the lung and provides important information to assist in
the clinical decision-making. Personalizing mechanical ventilation using EIT monitoring and
diagnostics may ameliorate the risk of death and long-term morbidity, and may substantially reduce
the burden on our healthcare system.

The perspective that EIT will become an important standard monitoring technique is shared by
international experts [@Frerichs2016;Wisse2024-sl]. Both @Frerichs2016 and @Wisse2024-sl emphasize
the importance of standardized techniques, terminology, and consensus regarding applications was
extensively discussed. However, validated methods to implement EIT information in routine care are
still lacking and synchronizing EIT data with other bedside respiratory monitoring modalities is
often practically impossible. Standardized implementation of EIT information is further limited
because the availability of bedside analysis tools depends on the type of EIT device used. Advanced
image and signal analysis could overcome certain challenges but also requires complex
post-processing (including detection/removal of common artifacts) and specific technical expertise
that is often not present in clinical practice. This currently hampers reproducibility of research
findings and clinical implementation. Moreover, it stresses the importance of close collaboration
between physicians, clinical researchers and engineers in order to identify clinical needs, to
develop and validate new algorithms, and to facilitate clinical implementation [@Scaramuzzo2024-ob].

`eitprocessing` offers a standardized, open, and highly expandable library of tools for loading,
filtering, segmentation and analysis of EIT data as well as related waveform or sparse data.
`eitprocessing` can work with data from the three main clinically available EIT devices, as well as
from related data sources. It includes commonly used methods for filtering and segmentation. The
authors continuously develop further algorithms for analysis for current and future projects. The
international community has been invited to use and contribute to the software.

## Key features

`eitprocessing` aims to simplify and standardize loading, pre-processing, analysis and reporting
while working with different respiration-related datasets. 

### Loading

`eitprocessing` supports the loading of EIT data exported from the Dräger Pulmovista (`.bin`
files), Timpel Enlight (`.csv` files) and Sentec LuMon (`.zri`) devices. Non-EIT data saved in the
data files are also loaded.

### Data containers

The main data container in `eitprocessing` is the sequence. A sequence represents a single
continuous measurement of data in a single subject, and can contain multiple datasets. Sequences
can be sliced --- by time or index --- and concatenated. Contained datasets are sliced and
concatenated accordingly.

`eitprocessing` currently supports four types of dataset. Continuous data has one-dimensional data
points at predictable intervals with a fixed sample frequency. Examples are airway pressure
measured by a mechanical ventilator or a global impedance signal. EIT data is also continuous, but
contains three-dimensional data points, (generally) 32 rows by 32 columns by time. Sparse data has
one-dimensional data points at unpredictable intervals and no set sample frequency. An example is
the tidal volume measured by a mechanical ventilator, registered at the end of each breath.
Interval data has one-dimensional data points that are valid for a time interval. An example is the
position of a subject, e.g., supine for the first part of a measurement and prone for the second
part.

### Pre-processing

`eitprocessing` currently has implementations for the following pre-processing steps:

- high-pass, low-pass, band-pass of band-stop Butterworth filters;
- a moving averager using convolution with a given window;
- automatic detection of the start, middle (end-inspiration) and end of breaths on a
  global/regional and pixel level.

### Analysis

`eitprocessing` currently has implementations for the following parameters:

- end-expiratory lung impedance on a global/regional and pixel level;
- tidal impedance variation on a global/regional and pixel level.

## Future perspective

Several features are in active development. Examples are:

- more advanced filtering methods, using a combination of Butterworth filters, empirical mode
  decomposition or wavelet transforms;
- automatic detection of respiratory and heart rate from pixel impedance values.

Moreover, we plan to extend `eitprocessing` with standardized workflows to summarize and report
analysis results.

## References
