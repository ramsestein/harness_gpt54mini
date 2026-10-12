# Clinical context for the AnesLLM decision task (verbatim excerpts)

Every quoted block below is copied literally from the published source named under it. Nothing has been paraphrased. Page numbers are PDF pages. Provenance, licences, glyph repairs and SHA-256 hashes: `guideline_context.sources.json`.


## Where the data come from (VitalDB)

> Data from the brain monitor (BIS VistaTM, Medtronic, Dublin, Ireland) were recorded in most patients undergoing general anesthesia and sedation/analgesia. Data from the target-controlled infusion pumps (Orchestra® Base Primea with module DPS, Fresenius Kabi AG, Bad Homburg, Germany) were recorded in all patients undergoing intravenous anesthesia and balanced anesthesia. The infusion pump data also includes infusion histories of various intravenous drugs.

<sub>[vdb_01] Lee & Jung, VitalDB v1.0.0 (PhysioNet), Methods (data acquisition), p. 3</sub>


## Propofol: target-controlled infusion, pharmacokinetics and time course

> A TCI pump contains a microprocessor programmed with pharmacokinetic models for relevant drugs. The user selects the drug and pharmacokinetic model to be used by that TCI pump and inputs the patient characteristics (covariates) such as body weight and age, and the target plasma or ‘brain’ (effect-site) concentration, with the pump determining the initial bolus and subsequent infusion rates. The two most commonly used adult propofol models are Marsh [4] and Schnider [5, 6].

<sub>[tiva_01] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Target-controlled infusions, p. 3</sub>

> Plasma drug concentrations in individual patients are unlikely to be identical to those predicted by the pharmacokinetic model and displayed by the TCI pump. The mean difference between estimated and measured concentrations is usually less than 25% [10], but, if the patient differs from the population in which the model was developed, the difference may be considerably greater. In such circumstances, TCI pumps can be a useful tool for titrating a propofol infusion to effect (clinical effect or the desired effect on the EEG as measured by a pEEG monitor), but the predicted propofol concentration cannot be assumed to be accurate.

<sub>[tiva_02] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), How relevant is the pharmacokinetic model to my patient?, p. 3</sub>

> It should be remembered that even when using drugs with the fastest pharmacokinetic profiles, a simple change in an infusion rate is associated with a significant delay before plasma concentrations change appreciably; this lag is even greater for effect-site concentrations and clinical effect. On starting a propofol infusion at a fixed rate without an initial bolus, concentrations rise very slowly and only reach near steady-state conditions after several hours (Fig. 1). If no loading dose is given, then administration by fixed infusion rate will be initially associated with inadequate concentrations. On the other hand, after some time at a fixed infusion rate, the concentrations may rise to excessive levels. Likewise, if the infusion rate is decreased, plasma concentrations will change slowly. In contrast, remifentanil achieves around 75% of steady-state concentration after 5 min, with 100% reached after 15–20 min.

<sub>[tiva_03] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Manual infusions, p. 4</sub>

> A clinical calibration of the individual patient’s response to propofol is recommended during induction and maintenance of anaesthesia. The drug concentration achieved should be sufficient to produce loss of consciousness and prevent movement in response to noxious stimuli. However, the concentration should not be excessive, as this may cause marked hypotension and delayed recovery from anaesthesia. There is no plasma or effect-site concentration that is appropriate for all patients. Rather, the concentration required will depend on inter-individual patient variation, other drugs administered and the degree of surgical stimulus.

<sub>[tiva_04] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Choosing an appropriate target drug concentration for a patient, p. 4</sub>

> The brain propofol concentration required for adequate anaesthesia during surgery is influenced by the magnitude of the surgical stimulus. An effective regional anaesthetic block will reduce the propofol concentration required.

<sub>[tiva_05] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Degree of surgical stimulus, p. 5</sub>

> During maintenance of anaesthesia, target concentrations of 3.0–6.0 µg.ml-1 (without opioids) or 2.5–4.0 µg.ml-1 (with opioids) are typical. Higher initial targets may be required for anxious and ‘robust’ individuals, whereas lower targets are appropriate for older, frail or unwell patients.

<sub>[tiva_06] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Typical target concentrations in routine practice, p. 5</sub>

> As with other rapidly acting intravenous anesthetic agents, the half-time of the blood-brain equilibration is approximately 1 to 3 minutes, accounting for the rate of induction of anesthesia.

<sub>[fda_01] DIPRIVAN (propofol) US prescribing information, 2022, 12 Clinical Pharmacology, p. 22</sub>

> Pharmacodynamic properties of propofol are dependent upon the therapeutic blood propofol concentrations. Steady-state propofol blood concentrations are generally proportional to infusion rates. Undesirable side effects, such as cardiorespiratory depression, are likely to occur at higher blood concentrations which result from bolus dosing or rapid increases in infusion rates. An adequate interval (3 to 5 minutes) must be allowed between dose adjustments in order to assess clinical effects.

<sub>[fda_02] DIPRIVAN (propofol) US prescribing information, 2022, 12.2 Pharmacodynamics, p. 22</sub>

> In adults, anesthesia can be maintained by administering DIPRIVAN by infusion or intermittent intravenous bolus injection. The patient's clinical response will determine the infusion rate or the amount and frequency of incremental injections Continuous Infusion DIPRIVAN 100 mcg/kg/min to 200 mcg/kg/min administered in a variable rate infusion with 60% to 70% nitrous oxide and oxygen provides anesthesia for patients undergoing general surgery. Maintenance by infusion of DIPRIVAN should immediately follow the induction dose in order to provide satisfactory or continuous anesthesia during the induction phase. During this initial period following the induction dose, higher rates of infusion are generally required (150 mcg/kg/min to 200 mcg/kg/min) for the first 10 minutes to 15 minutes. Infusion rates should subsequently be decreased 30% to 50% during the first half-hour of maintenance. Generally, rates of 50 mcg/kg/min to 100 mcg/kg/min in adults should be achieved during maintenance in order to optimize recovery times. Other drugs that cause CNS depression (hypnotics/sedatives, inhalational anesthetics, and opioids) can increase the CNS depression induced by propofol.

<sub>[fda_03] DIPRIVAN (propofol) US prescribing information, 2022, 2.3 Maintenance of General Anesthesia (Adult Patients), p. 6</sub>


## Depth of hypnosis and processed EEG (BIS)

> The BIS system uses a disposable 4-electrode sensor on the patient's forehead to measure electrical activity in the brain before using a proprietary algorithm to process the EEG data and calculate a number between 0 (absence of brain electrical activity) and 100 (wide awake). This provides a direct measure of the patient's response to anaesthetic drugs. The target range of BIS values during general anaesthesia is 40 to 60; this range indicates a low probability of awareness with recall.

<sub>[nice_01] NICE HTG292 (replaces DG6), 2012, 4.1 Bispectral Index, p. 11</sub>

> EEG-based depth of anaesthesia monitors are designed to indicate the probability of consciousness with explicit recall in patients receiving general anaesthetics, and to aid the tailoring of anaesthetic dose to the individual patient to avoid inadequate or excessively deep levels of anaesthesia. Measuring a patient's response to anaesthesia is important clinically because individual variation in response to anaesthetics can occasionally lead to inadequate or excessively deep levels of anaesthesia. An inadequate level of anaesthesia can result in patient awareness during surgery, which can cause post-traumatic stress disorder in some patients. Conversely, an excessively deep level of anaesthesia can result in prolonged recovery and has been linked to an increased risk of postoperative adverse outcomes, including myocardial infarction, stroke and cognitive dysfunction in older patients.

<sub>[nice_02] NICE HTG292 (replaces DG6), 2012, 3.1 The problem addressed, p. 7</sub>

> The use of electroencephalography (EEG)-based depth of anaesthesia monitors is recommended as an option during any type of general anaesthesia in patients considered at higher risk of adverse outcomes. This includes patients at higher risk of unintended awareness and patients at higher risk of excessively deep anaesthesia. The Bispectral Index (BIS) depth of anaesthesia monitor is therefore recommended as an option in these patients. 1.2 The use of EEG-based depth of anaesthesia monitors is also recommended as an option in all patients receiving total intravenous anaesthesia. The BIS monitor is therefore recommended as an option in these patients.

<sub>[nice_03] NICE HTG292 (replaces DG6), 2012, 1 Recommendations (1.1-1.2), p. 4</sub>

> Groups of patients who are considered at higher risk of unintended awareness during general anaesthesia include patients with high opiate or high alcohol use, patients with airway problems, and patients with previous experience of accidental awareness during surgery. The risk of unintended awareness is also raised by the concomitant use of muscle relaxants, particularly with total intravenous anaesthesia. Older patients, patients with comorbidities and those undergoing certain types of surgery are also considered at higher risk of unintended awareness because they are at greater risk of haemodynamic instability during surgery. Therefore, lower levels of anaesthetic are often used to prevent adverse effects on the cardiovascular system, which can result in these patient groups receiving inadequate levels of anaesthesia.

<sub>[nice_04] NICE HTG292 (replaces DG6), 2012, 3.7, p. 8</sub>

> For example, the EEG waveform may be displayed together with measures of the EEG signal quality, EMG activity and degree of burst suppression. Optimal use of a pEEG monitor involves using all the information it provides together with the information from other patient monitors, clinical judgement and experience. A pEEG index value may be a useful extra piece of information, but it should be considered along with all the other available information before making a judgement about whether anaesthetic dose should be adjusted.

<sub>[tiva_07] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Monitoring the patient during TIVA, p. 8</sub>

> If ketamine is given, then paradoxical increases in pEEG index value may occur [35].

<sub>[tiva_08] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Rapid sequence induction, p. 9</sub>

> Recommendation 5.1 We suggest Index-based EEG monitoring depth of anaesthesia guidance to decrease the risk of POD.

<sub>[esaic_01] Aldecoa et al., Eur J Anaesthesiol 2024 (ESAIC postoperative delirium guideline), Recommendation 5.1 (PICO 14), p. 13</sub>

> We suggest multiparameter, intraoperative EEG monitoring (burst suppression, density spectral array, DSA) during anaesthesia to decrease the risk of POD.

<sub>[esaic_02] Aldecoa et al., Eur J Anaesthesiol 2024 (ESAIC postoperative delirium guideline), Recommendation 5.2 (PICO 15), p. 13</sub>

> Index values may be less reliable in older people with reduced cortical electrical activity.

<sub>[esaic_03] Aldecoa et al., Eur J Anaesthesiol 2024 (ESAIC postoperative delirium guideline), PICO 14, evidence summary, p. 13</sub>

> In older patients, deep anaesthesia frequently causes burst suppression activity in the raw EEG,150 which has been identified as a risk factor for POD.129,130,140 – 143 This EEG pattern – an isoelectric line with intermittent bursts – can easily be identified in the raw EEG if presented on the monitors. Thus, anaesthesiologists should ensure that the monitors they use provide raw EEG traces.

<sub>[esaic_04] Aldecoa et al., Eur J Anaesthesiol 2024 (ESAIC postoperative delirium guideline), PICO 15, discussion, p. 15</sub>

> Based on adequate staff training, the anaesthesiologists should likely avoid burst suppression in their patients.

<sub>[esaic_05] Aldecoa et al., Eur J Anaesthesiol 2024 (ESAIC postoperative delirium guideline), Summary and future directions, p. 22</sub>


## Remifentanil and propofol-opioid interaction

> The administration of opioids, benzodiazepines, ketamine, α2-adrenoceptor agonists, magnesium and nitrous oxide result in a marked reduction in the required brain propofol concentration [13, 14]. Synergy of effect occurs between propofol and opioids. Opioids reduce the propofol dose required to produce loss of consciousness and, in particular, to obtund movement and haemodynamic responses to noxious stimuli [15]. A remifentanil infusion is often used in conjunction with propofol infusion. The rapid offset of effect after stopping remifentanil enables doses to be given that reduce propofol requirements by approximately 50% (Table 1) without causing prolonged respiratory depression after surgery. However, intra-operative remifentanil does not provide postoperative analgesia. Higher dose remifentanil infusions (target concentration above 5 ng.ml-1 or infusions above 0.2 µg.kg-1.min-1) may cause acute opioid tolerance or opioid-induced hyperalgesia but the clinical significance of this is uncertain [16].

<sub>[tiva_09] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Other drugs administered, p. 5</sub>

> Where TCI remifentanil is administered with propofol, target remifentanil concentrations of 2–6 ng.ml-1 are commonly used (equivalent to manual infusion rates of approximately of 0.08–0.25 µg.kg-1.min-1) and will usually necessitate ventilation of the patient’s lungs, as spontaneous ventilation is uncommon with concentrations > 1.5 ng.ml-1 in adults. During maintenance of anaesthesia, the target propofol concentration and opioid administration should be adjusted, using clinical judgement supported by the observation of clinical signs, and supplemented by the use of a pEEG device if the patient has received a neuromuscular blocking drug.

<sub>[tiva_10] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Typical target concentrations in routine practice, p. 5</sub>

> Pharmacodynamics: The analgesic effects of ULTIVA are rapid in onset and offset. Its effects and side effects are dose dependent and similar to other µ-opioids. ULTIVA in humans has a rapid blood-brain equilibration half-time of 1 ± 1 minutes (mean ± SD) and a rapid onset of action. The pharmacodynamic effects of ULTIVA closely follow the measured blood concentrations, allowing direct correlation between dose, blood levels, and response. Blood concentration decreases 50% in 3 to 6 minutes after a 1-minute infusion or after prolonged continuous infusion due to rapid distribution and elimination processes and is independent of duration of drug administration. Recovery from the effects of ULTIVA occurs rapidly (within 5 to 10 minutes). New steady-state concentrations occur within 5 to 10 minutes after changes in infusion rate.

<sub>[ult_01] ULTIVA (remifentanil) US prescribing information, 2004, Clinical Pharmacology: Pharmacodynamics, p. 1</sub>

> Anesthesia: ULTIVA is synergistic with the activity of hypnotics (propofol and thiopental), inhaled anesthetics, and benzodiazepines (see CLINICAL TRIALS, PRECAUTIONS, and DOSAGE AND ADMINISTRATION).

<sub>[ult_02] ULTIVA (remifentanil) US prescribing information, 2004, Clinical Pharmacology: Pharmacodynamics (Anesthesia), p. 3</sub>

> ULTIVA is synergistic with other anesthetics; therefore, clinicians may need to reduce doses of thiopental, propofol, isoflurane, and midazolam by up to 75% with the coadministration of ULTIVA. The administration of ULTIVA must be individualized based on the patient’s response.

<sub>[ult_03] ULTIVA (remifentanil) US prescribing information, 2004, Dosage and Administration: During General Anesthesia, p. 22</sub>

> During Maintenance of Anesthesia: After endotracheal intubation, the infusion rate of ULTIVA should be decreased in accordance with the dosing guidelines in Tables 10 (adults) and 11 (pediatric patients). Due to the fast onset and short duration of action of ULTIVA, the rate of administration during anesthesia can be titrated upward in 25% to 100% increments in adult patients or up to 50% increments in pediatric patients, or downward in 25% to 50% decrements every 2 to 5 minutes to attain the desired level of µ-opioid effect. In response to light anesthesia or transient episodes of intense surgical stress, supplemental bolus doses of 1 mcg/kg may be administered every 2 to 5 minutes. At infusion rates >1 mcg/kg/min, increases in the concomitant anesthetic agents should be considered to increase the depth of anesthesia.

<sub>[ult_04] ULTIVA (remifentanil) US prescribing information, 2004, Dosage and Administration: During Maintenance of Anesthesia, p. 23</sub>


## Arterial pressure management and anaesthetic-induced hypotension

> Consensus statement 1: Intraoperative mean arterial pressures <60–70 mm Hg or systolic arterial pressures <90–100 mm Hg are associated with acute kidney injury, myocardial injury, myocardial infarction, and death. Injury is a function of hypotension severity and duration (high-quality evidence). Consensus recommendation 1: We recommend keeping intraoperative mean arterial pressure ≥60 mm Hg in at-risk patients (strong recommendation, moderate-quality evidence).

<sub>[poqi_01] Saugel et al., Br J Anaesth 2024 (POQI arterial pressure consensus), Consensus statement 1 / recommendation 1, p. 2</sub>

> In summary, there are clear associations between intraoperative hypotension and renal and myocardial injury. However, trial data remain sparse and suffer methodologic limitations. The extent to which the associations are causal thus remains largely unknown; consequently, the acceptable lower limit of intraoperative arterial pressure remains uncertain. Nonetheless, available information suggests that keeping intraoperative mean arterial pressure ≥60 mm Hg is advisable.

<sub>[poqi_02] Saugel et al., Br J Anaesth 2024 (POQI arterial pressure consensus), Consensus recommendation 1, summary, p. 4</sub>

> Consensus recommendation 3: We recommend that treatment of hypotension be based on presumed underlying causes including vasodilation, hypovolaemia, bradycardia, and low cardiac output.

<sub>[poqi_03] Saugel et al., Br J Anaesth 2024 (POQI arterial pressure consensus), Table 1 (consensus recommendation 3), p. 3</sub>

> Hypotension can result from several underlying physiological processes acting alone or in combination.39 Vasodilation, hypovolaemia, bradycardia, and low cardiac output are all modifiable causes of hypotension.39 Vasodilation can be reversed by vasopressors such as phenylephrine or norepinephrine. Hypovolaemia can be treated with intravascular fluid administration using a variety of different types of fluids including crystalloid and colloid solutions or blood. Bradycardia is typically pharmacologically managed with anticholinergic agents such as atropine or glycopyrronium; when not responsive to these agents, epinephrine or isoprenaline might be necessary. A pacemaker can be used to manage profound bradycardia. Finally, low cardiac output due to acute or chronic myocardial dysfunction can be treated with positive inotropic agents such as dobutamine or epinephrine.

<sub>[poqi_04] Saugel et al., Br J Anaesth 2024 (POQI arterial pressure consensus), Consensus recommendation 3, p. 4</sub>

> Consensus statement 2: In general surgery, there is no association between intraoperative systolic arterial pressures between 120 and 200 mm Hg and acute kidney injury and myocardial injury (low-quality evidence). Consensus recommendation 4: If intraoperative hypertension is treated, we recommend caution to avoid hypotension (strong recommendation, moderate-quality evidence).

<sub>[poqi_05] Saugel et al., Br J Anaesth 2024 (POQI arterial pressure consensus), Consensus statement 2 / recommendation 4, p. 4</sub>

> Some degree of hypertension should prompt treatment. However, given that reasonable levels of hypertension appear relatively benign, available data suggest that hypertension should be treated incrementally, if at all, with the goal of avoiding hypotension, which undoubtedly causes organ injury at some level.

<sub>[poqi_06] Saugel et al., Br J Anaesth 2024 (POQI arterial pressure consensus), Consensus recommendation 4, p. 4</sub>

> Multiple factors contribute to intraoperative hypotension including anaesthetic drugs (which induce vasodilation, myocardial depression, or bradycardia), bleeding (resulting in intravascular hypovolaemia), mechanical ventilation (causing high intrathoracic pressure impairing venous return), and patient positioning. Consequently, there are various endotypes of intraoperative hypotension characterised by different underlying haemodynamic alterations. Identified endotypes include myocardial depression, bradycardia, vasodilation, and hypovolaemia.39 It remains unknown whether various causes of hypotension affect the association between hypotension and organ injury, although it seems likely that the degree of hypotension is more important than its cause. In contrast, treatment of hypotension should be directed to the presumed cause(s) because cause-specific treatments are more likely to be effective than generic treatments.

<sub>[poqi_07] Saugel et al., Br J Anaesth 2024 (POQI arterial pressure consensus), Open research questions: causes of hypotension, p. 5</sub>

> As with other anesthetic and sedative-hypnotic agents, DIPRIVAN in healthy patients causes a decrease in blood pressure that is secondary to decreases in preload (ventricular filling volume at the end of the diastole) and afterload (arterial resistance at the beginning of the systole). The magnitude of these changes is proportional to the blood and effect site concentrations achieved. These concentrations depend upon the dose and speed of the induction and maintenance infusion rates. In addition, lower heart rates are observed during maintenance with DIPRIVAN, possibly due to reduction of the sympathetic activity and/or resetting of the baroreceptor reflexes. Therefore, anticholinergic agents should be administered when increases in vagal tone are anticipated.

<sub>[fda_04] DIPRIVAN (propofol) US prescribing information, 2022, 2.2 Induction of General Anesthesia, p. 6</sub>

> Anaesthetic agents can affect the body's physiology, in particular, the cardiovascular system. Adverse outcomes of excessively deep general anaesthesia include prolonged recovery, particularly in people with a high BMI. In severe cases or in at-risk patient groups (for example, older patients, patients with liver disease, and patients with poor cardiovascular function), excessively deep anaesthesia can result in haemodynamic instability and respiratory complications (which can be fatal without cardiorespiratory support). Inappropriately deep anaesthesia has also been linked to an increased risk of post-operative complications such as myocardial infarction and stroke in older patients. There is some evidence to suggest a link between longer term morbidity (for example, cognitive dysfunction) and mortality, and the depth of anaesthesia.

<sub>[nice_05] NICE HTG292 (replaces DG6), 2012, 3.6, p. 8</sub>

> Hemodynamics: In premedicated patients undergoing anesthesia, 1-minute infusions of <2 mcg/kg of ULTIVA cause dose-dependent hypotension and bradycardia. While additional doses >2 mcg/kg (up to 30 mcg/kg) do not produce any further decreases in heart rate or blood pressure, the duration of the hemodynamic change is increased in proportion to the blood concentrations achieved. Peak hemodynamic effects occur within 3 to 5 minutes of a single dose of ULTIVA or an infusion rate increase. Glycopyrrolate, atropine, and vagolytic neuromuscular blocking agents attenuate the hemodynamic effects associated with ULTIVA. When appropriate, bradycardia and hypotension can be reversed by reduction of the rate of infusion of ULTIVA, or the dose of concurrent anesthetics, or by the administration of fluids or vasopressors.

<sub>[ult_05] ULTIVA (remifentanil) US prescribing information, 2004, Clinical Pharmacology: Pharmacodynamics (Hemodynamics), p. 2</sub>


## Older, frail and ASA III-IV patients

> Starting target concentrations should be chosen depending on the characteristics of the patient, co-administered drugs and clinical situation. Older, frail or unwell patients may benefit from setting a low initial target propofol concentration, and making repeated small incremental increases.

<sub>[tiva_11] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Recommendation 3, p. 2</sub>

> In general, older patients require a lower brain anaesthetic drug concentration than younger patients [6], but there is considerable variation between individuals of the same age and overlap between patients of different ages.

<sub>[tiva_12] Nimmo et al., Anaesthesia 2019 (AAGBI/SIVA TIVA guidelines), Inter-individual variation, p. 4</sub>

> A lower induction dose and a slower maintenance rate of administration should be used in elderly, debilitated, or ASA-PS III or IV patients [see Dosage and Administration (2)]. Patients should be continuously monitored for early signs of hypotension and/or bradycardia.

<sub>[fda_05] DIPRIVAN (propofol) US prescribing information, 2022, 5.12 Use in the Elderly, Debilitated, or ASA-PS III or IV Patients, p. 14</sub>

> With increasing patient age, the dose of propofol needed to achieve a defined anesthetic end point (dose-requirement) decreases. This does not appear to be an age-related change in pharmacodynamics or brain sensitivity, as measured by EEG burst suppression. With increasing patient age, pharmacokinetic changes are such that, for a given intravenous bolus dose, higher peak plasma concentrations occur, which can explain the decreased dose requirement. These higher peak plasma concentrations in the elderly can predispose patients to cardiorespiratory effects including hypotension, apnea, airway obstruction, and/or arterial oxygen desaturation. The higher plasma levels reflect an age-related decrease in volume of distribution and intercompartmental clearance. Lower doses are therefore recommended for initiation and maintenance of sedation and anesthesia in elderly patients [see Dosage and Administration (2)].

<sub>[fda_06] DIPRIVAN (propofol) US prescribing information, 2022, 12.3 Pharmacokinetics (Geriatrics), p. 24</sub>

> Age: The pharmacodynamic activity of ULTIVA (as measured by the EC50 for development of delta waves on the EEG) increases with increasing age. The EC50 of remifentanil for this measure was 50% less in patients over 65 years of age when compared to healthy volunteers (25 years of age) (see DOSAGE AND ADMINISTRATION).

<sub>[ult_06] ULTIVA (remifentanil) US prescribing information, 2004, Clinical Pharmacology: Pharmacodynamics (Age), p. 3</sub>

> While the effective biological half-life of remifentanil is unchanged, elderly patients have been shown to be twice as sensitive as the younger population to the pharmacodynamic effects of remifentanil. The recommended starting dose of ULTIVA should be decreased by 50% in patients over 65 years of age (see CLINICAL PHARMACOLOGY and DOSAGE AND ADMINISTRATION).

<sub>[ult_07] ULTIVA (remifentanil) US prescribing information, 2004, Precautions: Geriatric Use, p. 14</sub>

> Elderly: The clearance of remifentanil is reduced (approximately 25%) in the elderly (>65 years of age) compared to young adults (average 25 years of age). However, remifentanil blood concentrations fall as rapidly after termination of administration in the elderly as in young adults.

<sub>[ult_08] ULTIVA (remifentanil) US prescribing information, 2004, Clinical Pharmacology: Special Populations (Elderly), p. 5</sub>
