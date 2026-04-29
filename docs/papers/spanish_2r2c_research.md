# Authoritative 2R2C Parameter Sources for a Barcelona (CTE C2) Residential Apartment

## Honest assessment up front

**Yes — a directly-fitting peer-reviewed source exists.** Péan, Salom & Costa-Castelló (2018, IEEE CDC) publish a fully identified 2R2C (R2C2) grey-box model whose study case is explicitly *"a flat for a family of four members, within a multi-family building block, typical of the Spanish and Mediterranean climate areas"*, with all four parameters (C_int, C_w, R_int, R_w) reported numerically in their Table 1. This is, to the best of what is verifiable on the open web, the single best-fitting source for the request: Mediterranean apartment + 2R2C structure + numerical R/C values, peer-reviewed, from a Barcelona-based group (IREC + UPC). A later open-access journal paper by the same authors (Sustainable Cities & Society, 2019) re-uses the same study case. Caveats: the Péan model represents a *refurbished* apartment (12 cm extra insulation), so for an unrefurbished CTE-C2 apartment one would still need to scale R_ext upward; and the "C_int = 0.26 kWh/K" value lumps mostly air, while C_w = 19.1 kWh/K lumps the envelope mass. Beyond Péan, the most defensible fall-back is to combine CTE DB-HE U-values with a typology-derived geometry (TABULA-ES) — this is widely done in the Spanish literature and is honest if explicitly framed as an engineering estimate, not a measurement-identified parameter set.

---

## Ranked source list

### 1. Péan, Salom & Costa-Castelló (2018) — *the direct hit*
- **Citation:** Péan, T., Salom, J., Costa-Castelló, R. (2018). "Configurations of model predictive control to exploit energy flexibility in building thermal loads." *57th IEEE Conference on Decision and Control (CDC)*, Miami, FL, pp. 3177–3182. DOI: 10.1109/CDC.2018.8619095.
- **Open-access:** https://upcommons.upc.edu/handle/2117/127600 (full PDF on UPCommons).
- **What it publishes:** Identified 2R2C parameters (their Table 1): **C_int ≈ 0.26 kWh/K, C_w ≈ 19.1 kWh/K, R_int ≈ 0.42 K/kW, R_w ≈ 8.86 K/kW**, plus solar aperture A_w ≈ 1.92 m². Identification was done via PRBS excitation of a validated TRNSYS model.
- **Building type / climate:** Refurbished apartment (added 12 cm insulation) in a multi-family block, "typical of the Spanish and Mediterranean climate areas" (IREC, Barcelona).
- **Authority:** Peer-reviewed IEEE conference; lead author's PhD work at UPC/IREC under INCITE H2020.
- **Defensibility:** Strong. The parameters can defensibly seed a Barcelona apartment 2R2C model, with the explicit caveat that they represent a *refurbished* envelope. For a typical unrefurbished CTE-C2 apartment, scale R_w downward (envelope ~2–3× more conductive) and leave C close to as-published.

### 2. Péan, Costa-Castelló & Salom (2019) — journal version, same study case
- **Citation:** Péan, T., Costa-Castelló, R., Salom, J. (2019). "Price and carbon-based energy flexibility of residential heating and cooling loads using model predictive control." *Sustainable Cities and Society* **50**, 101579. DOI: 10.1016/j.scs.2019.101579.
- **What it publishes:** Re-uses the same R2C2 building model and parameters as #1, with extended results. Confirms "R2C2 had good accuracy for modelling the temperature profiles and heating demand of a residential dwelling … representative of the building stock of Catalonia" (as cited by Pascual et al., IBPSA 2021).
- **Why ranked second:** Adds peer-reviewed journal authority and corroboration, but does not add an independent identification.

### 3. Mont Lecocq, Pascual & Salom (2024/2025) — IREC follow-up, multiple residential archetypes
- **Citation:** Mont Lecocq, E., Pascual, J., Salom, J. (2024). "Development of physically coherent grey-box models for residential buildings using a simplified adjustment method." *Energy and Buildings* **325**, 115004. DOI: 10.1016/j.enbuild.2024.115004 (also as SSRN preprint 4946700, https://ssrn.com/abstract=4946700).
- **What it publishes:** R2C2 models for several residential archetypes plus a documented method to adjust the four parameters for envelope variations (insulation, glazing, etc.) — directly useful to convert the Péan refurbished-flat baseline into an as-built CTE-C2 case.
- **Authority:** Peer-reviewed *Energy and Buildings*; same IREC group.
- **Defensibility:** High — this is the most up-to-date Catalan/Mediterranean residential R2C2 reference and addresses precisely the renovation-state adjustment gap of source #1. Verify the published parameter table once accessed.

### 4. AmBIENCe Project Deliverable D4.1 (BPIE/VITO, 2021) — EU-27 grey-box database (includes Spain)
- **Citation:** Jankovic, I., Fernandez, X., Diriken, J. (2021). *Database of grey-box model parameter values for EU building typologies* (Deliverable D4.1, AmBIENCe H2020 GA 847054). BPIE / VITO. https://ambience-project.eu/wp-content/uploads/2022/02/AmBIENCe_D4.1_Database-of-grey-box-model-parameter-values-for-EU-building-typologies-update-version-2-submitted.pdf
- **What it publishes:** Country-specific RC parameters (forward-derived from TABULA + Hotmaps via Modelica/IDEAS white-box simulations and PRBS identification). Includes Spanish multi-family residential archetypes by construction period; the dataset is publicly downloadable and contains "Zone D / Zone A / Zone Wall B" grey-box variants — Zone Wall B is functionally a 2R2C structure.
- **Authority:** Institutional H2020 deliverable, methodologically transparent, but values are *forward-simulated* not measurement-identified.
- **Defensibility:** Medium-strong as a sanity check / second source; weaker than #1–3 because it is not measurement-based and the Spanish reference cities used in TABULA may not equal Barcelona.

### 5. CTE DB-HE + TABULA-ES — defensible fall-back if measurement values are rejected
- **Citations:**
  - Ministerio de Fomento (2019/2022). *Documento Básico HE Ahorro de Energía*, Código Técnico de la Edificación. Tables 3.1.1.a-HE1 (U-value limits per climate zone, including C2). https://www.codigotecnico.org/
  - DA DB-HE/1 *Cálculo de parámetros característicos de la envolvente*: https://www.codigotecnico.org/pdf/Documentos/HE/DA_DB-HE-1_Calculo_de_parametros_caracteristicos_de_la_envolvente.pdf
  - Episcope/TABULA Spain (IVE, 2011/2014): national typology with envelope geometry per period.
- **What it publishes:** U-values (W/m²K) and surface resistances for each climate zone (C2 includes Barcelona); TABULA gives wall/roof/floor/window areas per archetype. R_ext (≈ 1/(U·A_env)) and C_wall (Σ ρ·c·V·thickness/area) follow by standard lumping (e.g. ISO 13790 5R1C reduction or VDI 6007).
- **Authority:** Mandatory Spanish technical code; TABULA is the EU-recognized typology framework.
- **Defensibility:** This route does **not** publish 2R2C parameters directly — every paper that uses it documents its lumping assumptions. It is a defensible engineering estimate when explicitly labelled as such, and is the standard approach when no measurement-identified set is available. It should be reported as "derived from CTE U-values with assumptions X, Y, Z," never as a measured parameter set.

---

## Recommendation

Anchor the parameter set on **Péan et al. 2018 (#1)** as the primary citation (it is the only verifiable peer-reviewed publication of identified 2R2C parameters for a Spanish-Mediterranean multi-family apartment), apply the **Mont Lecocq et al. 2024 (#3)** adjustment method to translate from the refurbished baseline to the target CTE-C2 envelope state, and use **AmBIENCe D4.1 (#4)** and the **CTE/TABULA route (#5)** as independent cross-checks. If only one citation is permissible, source #1 remains the defensibly correct choice — provided the refurbishment caveat is stated in the methods section.