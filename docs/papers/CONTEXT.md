# Papers — Citation Index

This file is the single source of truth for academic and authoritative
sources cited in the case studies under `docs/drafts/`. It is maintained
by the `citation-manager` skill. Each entry below corresponds to one PDF
in `docs/papers/`. Drafts cite these via markdown footnote markers
(`[^slug]`), and the §11 References section of each draft is generated
from the markers found in that draft.

---

## ambience-d4-1 — AmBIENCe D4.1: Database of grey-box model parameter values for EU building typologies

- **Citation:** Jankovic, I., Fernandez, X., & Diriken, J. (2021).
  *Database of grey-box model parameter values for EU building
  typologies* (Deliverable D4.1, V02). AmBIENCe project, EU Horizon 2020
  GA No 847054. BPIE / VITO. December 2021.
- **File:** [ambience-d4-1.pdf](ambience-d4-1.pdf)
- **Summary:** EU27-wide database of grey-box thermal model parameters
  for residential and non-residential reference buildings, derived
  forward from TABULA and Hotmaps stock data via standardised assumptions
  (shoe-box, one-or-few-zone models). Parameters are forward-simulated,
  not measurement-identified. Includes Spanish multi-family residential
  archetypes by construction period.
- **Supports claims:**
  - EU-wide reference grey-box parameter sets exist for residential
    typologies including Spanish multi-family stock
  - Forward-derivation from typology + envelope characteristics is a
    standard route when measurement data is unavailable
  - Reference parameter values vary substantially with construction
    period, climate zone, and envelope refurbishment state
- **Cited in:**
  - — not yet cited

---

## bacher2011 — Identifying suitable models for the heat dynamics of buildings

- **Citation:** Bacher, P., & Madsen, H. (2011). Identifying suitable
  models for the heat dynamics of buildings. *Energy and Buildings*,
  43(7), 1511–1522. DOI: 10.1016/j.enbuild.2011.02.005
- **File:** [bacher2011.pdf](bacher2011.pdf)
- **Summary:** Establishes a forward-selection procedure for identifying
  the appropriate order of grey-box RC models for building heat dynamics,
  fitted via maximum likelihood with a Kalman filter. Demonstrated on a
  120 m² single-storey building at the Risø DTU experimental energy
  system in Denmark. Canonical reference for residential thermal grey-box
  modelling.
- **Supports claims:**
  - Grey-box RC modelling is a standard, well-validated methodology for
    residential building thermal dynamics
  - Multiple RC model orders (1R1C, 2R2C, higher-order) exist with known
    trade-offs in fidelity vs. parameter count
  - Parameter identification from measured time-series data is itself a
    non-trivial problem requiring maximum-likelihood / Kalman-filter
    techniques and statistical model selection
- **Cited in:**
  - — not yet cited

---

## cte-db-he-da1 — CTE DA DB-HE/1: Cálculo de parámetros característicos de la envolvente

- **Citation:** Ministerio de Transportes, Movilidad y Agenda Urbana
  (2020). *Documento de Apoyo al Documento Básico DB-HE Ahorro de
  energía — DA DB-HE/1: Cálculo de parámetros característicos de la
  envolvente.* Código Técnico de la Edificación. Madrid: Dirección
  General de Arquitectura, Vivienda y Suelo. (Version: enero 2020;
  prior: febrero 2015.)
- **File:** [cte-db-he-da1.pdf](cte-db-he-da1.pdf)
- **Summary:** Official Spanish technical guidance document under the
  *Código Técnico de la Edificación* (CTE) DB-HE energy-saving section.
  Specifies the calculation procedure for the thermal envelope's
  characteristic parameters: thermal transmittance (U-values), solar
  transmittance of semi-transparent elements, and total thermal
  resistance of layered constructions. References UNE EN ISO 6946:2012
  and related standards. This is the *method* document, not a parameter
  table.
- **Supports claims:**
  - There is a formal, mandatory Spanish technical procedure for
    deriving envelope thermal parameters
  - Standardised methods exist to derive lumped envelope resistances
    from layered construction descriptions
  - Spain has a national-code framework for building thermal
    characterisation that supports engineering parameter estimation
    when measurement data is unavailable
- **Cited in:**
  - — not yet cited

---

## pean2018 — Configurations of model predictive control to exploit energy flexibility in building thermal loads

- **Citation:** Péan, T., Salom, J., & Costa-Castelló, R. (2018).
  Configurations of model predictive control to exploit energy
  flexibility in building thermal loads. In *Proc. 57th IEEE Conference
  on Decision and Control (CDC)*, Miami, FL, USA, pp. 3177–3182.
  DOI: 10.1109/CDC.2018.8619095
- **File:** [pean2018.pdf](pean2018.pdf)
- **Summary:** Develops an MPC framework for residential heating /
  cooling loads with three objective formulations (thermal energy,
  electrical energy, electricity cost). The 2R2C ("R2C2") grey-box
  building model is identified via PRBS excitation of a validated
  TRNSYS model. The study case is *"a flat for a family of four
  members, within a multi-family building block, typical of the Spanish
  and Mediterranean climate areas"* — a refurbished apartment with 12 cm
  added insulation, located in Catalonia (IREC, Sant Adrià de Besòs,
  Barcelona).
- **Identified parameters (Table 1, p. 2):**
  - $C_{int}$ = 0.26 kWh/K, $C_w$ = 19.1 kWh/K
  - $R_{int}$ = 0.42 K/kW, $R_w$ = 8.86 K/kW
  - Solar aperture $gA$ = 1.92 m²
  - (Plus TES tank and heat-pump COP coefficients, not relevant to a
    pure heating-only LP without storage tank.)
- **Supports claims:**
  - A 2R2C grey-box thermal model with identified parameters has been
    published for a Mediterranean / Spanish multi-family residential
    apartment
  - Reference 2R2C parameter values for a refurbished Catalan apartment
    are documented and peer-reviewed
  - MPC for residential thermal flexibility against electricity prices
    is an established research direction in the Spanish context
  - The 2R2C model structure (one fast air-temperature state, one slow
    envelope state) is a standard order for residential MPC applications
  - Parameter values published here represent a *refurbished* envelope
    (12 cm extra insulation) — direct transfer to an unrefurbished
    apartment requires explicit adjustment of $R_w$
- **Cited in:**
  - — not yet cited

---

## tabula-spain — Use of Building Typologies for Energy Performance Assessment: Spain

- **Citation:** Valencian Institute of Building (IVE) (2011). *Use of
  Building Typologies for Energy Performance Assessment of National
  Building Stock: Existent Experiences in Spain.* TABULA project report.
  Valencia: Instituto Valenciano de la Edificación.
- **File:** [ES_TABULA_Report_IVE.pdf](ES_TABULA_Report_IVE.pdf)
- **Summary:** Spanish national report to the EU TABULA project,
  documenting the Spanish residential building stock, climate-zone
  classification under the *Código Técnico de la Edificación* (CTE),
  and existing typology-based energy-assessment frameworks (Rehenergía,
  Retrofit, CERMA+). Establishes the typology framework rather than
  publishing tabulated grey-box RC parameters directly.
- **Supports claims:**
  - Spanish residential building stock spans diverse construction
    periods and typologies with thermal characteristics that vary by
    climate zone
  - Spain has a formal climate-zone classification (CTE) relevant to
    building thermal performance
  - National-typology methodologies exist for energy performance
    assessment of Spanish residential buildings
- **Cited in:**
  - — not yet cited
