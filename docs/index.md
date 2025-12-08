---
layout: page
title: TinyODOM-EX
permalink: /
exclude: true
---

Joseph Zales ([GitHub](https://github.com/Joseph-Q-Zales))  

<div style="text-align: center">
  <img src="./assets/img/TinyODOM-EX logo.png" alt="logo" width="500" />
</div>

<style>
.triple-download-btn {
  display: flex;
  border: 1px solid #0366d6;
  border-radius: 6px;
  overflow: hidden;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 1.5rem 0;
}

/* Each third of the button */
.triple-download-btn a {
  flex: 1;
  text-decoration: none;
  text-align: center;
  padding: 0.6rem 0.75rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
  line-height: 1.1;
  cursor: pointer;
}

/* Small "download" / "view on" text */
.triple-download-btn .label-small {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.8;
}

/* Big "ZIP file" / "TAR Ball" / "GitHub" text */
.triple-download-btn .label-big {
  font-size: 0.95rem;
  font-weight: 600;
}

/* Colors per segment */
.triple-download-btn a:nth-child(1),
.triple-download-btn a:nth-child(2) {
  background: #f6f8fa;
  color: #0366d6;
}

.triple-download-btn a:nth-child(2) {
  border-left: 1px solid #e1e4e8;
  border-right: 1px solid #e1e4e8;
}

.triple-download-btn a:nth-child(3) {
  background: #24292e;
  color: #ffffff;
}

/* Hover state */
.triple-download-btn a:hover {
  background: #0366d6;
  color: #ffffff;
}
</style>

<div class="triple-download-btn">
  <a href="https://github.com/Joseph-Q-Zales/ECM202A_2025Fall_Project_1/zipball/main">
    <span class="label-small">Download</span>
    <span class="label-big">ZIP file</span>
  </a>

  <a href="https://github.com/Joseph-Q-Zales/ECM202A_2025Fall_Project_1/tarball/main">
    <span class="label-small">Download</span>
    <span class="label-big">TAR Ball</span>
  </a>

  <a href="https://github.com/Joseph-Q-Zales/ECM202A_2025Fall_Project_1">
    <span class="label-small">View on</span>
    <span class="label-big">GitHub</span>
  </a>
</div>

---

## 📝 **Abstract**

- Problem: inertial odometry in GPS denied settings, TinyODOM showed on device neural odometry on MCUs but ignored energy
- Approach: extend TinyODOM NAS with Optuna, hardware in the loop BLE33 measurements, and a modernized TFLite Micro plus Arduino toolchain
- Key results: accuracy latency memory vs TinyODOM baselines on BLE33, plus first energy aware NAS curves and lessons from the failed RP2040 path
*- Probs write at the end*

## **Slides**

- [Midterm Checkpoint Slides](http://https://docs.google.com/presentation/d/1UtyWay7o1q8KnlKmfcb9YPXjvYXP0kHzi5wXddbCKxM/edit?usp=sharing)  
- [Final Presentation Slides](http://)

---
# Table of Contents:
- [Table of Contents:](#table-of-contents)
- [**1. Introduction**](#1-introduction)
    - [**1.1 Motivation \& Objective**](#11-motivation--objective)
    - [**1.2 State of the Art \& Its Limitations**](#12-state-of-the-art--its-limitations)
    - [**1.3 Novelty \& Rationale**](#13-novelty--rationale)
    - [**1.4 Potential Impact**](#14-potential-impact)
    - [**1.5 Challenges**](#15-challenges)
    - [**1.6 Metrics of Success**](#16-metrics-of-success)
- [**2. Related Work**](#2-related-work)
- [**3. Technical Approach**](#3-technical-approach)
    - [**3.1 TinyODOM-EX System Architecture**](#31-tinyodom-ex-system-architecture)
    - [**3.2 Dataset and Windowing Pipeline**](#32-dataset-and-windowing-pipeline)
    - [**3.3 NAS Objective, Search Space, and Training Procedure**](#33-nas-objective-search-space-and-training-procedure)
    - [**3.4 Hardware in the Loop Measurement and Implementation**](#34-hardware-in-the-loop-measurement-and-implementation)
    - [**3.5 Key Design Decisions and Tradeoffs**](#35-key-design-decisions-and-tradeoffs)
- [**4. Evaluation \& Results**](#4-evaluation--results)
    - [**4.1 Experimental Studies and Metrics**](#41-experimental-studies-and-metrics)
    - [**4.2 Reproducing TinyODOM on BLE33 (Study 1)**](#42-reproducing-tinyodom-on-ble33-study-1)
      - [**4.2.1 Baseline Model Quality**](#421-baseline-model-quality)
      - [**4.2.2 Baseline Latency and Model Size Profile**](#422-baseline-latency-and-model-size-profile)
    - [**4.3 Single-Objective Energy-Aware NAS on BLE33 (Studies 1 and 2)**](#43-single-objective-energy-aware-nas-on-ble33-studies-1-and-2)
      - [**4.3.1 Effect of Energy Logging on NAS Outcome**](#431-effect-of-energy-logging-on-nas-outcome)
      - [**4.3.2 Empirical Relationship Between Energy and Latency**](#432-empirical-relationship-between-energy-and-latency)
    - [**4.4 Multi-Objective NAS: Accuracy vs Latency (Study 3)**](#44-multi-objective-nas-accuracy-vs-latency-study-3)
      - [**4.4.1 Pareto Front and Convergence**](#441-pareto-front-and-convergence)
    - [**4.4.2 Hyperparameter Sensitivity for Latency-Oriented Search**](#442-hyperparameter-sensitivity-for-latency-oriented-search)
    - [**4.5 Multi-Objective NAS: Accuracy vs Energy (Study 4)**](#45-multi-objective-nas-accuracy-vs-energy-study-4)
      - [**4.5.1 Pareto Front in Accuracy–Energy Space**](#451-pareto-front-in-accuracyenergy-space)
    - [**4.5.2 Energy-Oriented Hyperparameter Structure**](#452-energy-oriented-hyperparameter-structure)
    - [**4.6 Cross-Study Comparison and NAS Trial Budget**](#46-cross-study-comparison-and-nas-trial-budget)
      - [**4.6.1 Summary of Best Models Across Studies**](#461-summary-of-best-models-across-studies)
      - [**4.6.2 NAS Trial Budget and Practical Convergence**](#462-nas-trial-budget-and-practical-convergence)
    - [**4.x Multi-Objective NAS**](#4x-multi-objective-nas)
    - [**4.x NAS Trial Budget**](#4x-nas-trial-budget)
- [**5. Discussion \& Conclusions**](#5-discussion--conclusions)
    - [**5.1 Summary of Key Findings**](#51-summary-of-key-findings)
    - [**5.2 Lessons from Energy-Aware NAS and HIL Infrastructure**](#52-lessons-from-energy-aware-nas-and-hil-infrastructure)
    - [**5.3 Limitations and Threats to Validity**](#53-limitations-and-threats-to-validity)
    - [**5.4 Future Work**](#54-future-work)
    - [**5.5 Final Conclusion**](#55-final-conclusion)
- [**6. References**](#6-references)
- [**7. Supplementary Material**](#7-supplementary-material)
    - [**7.1. Datasets**](#71-datasets)
    - [**7.2. Software**](#72-software)
- [🧭 **Guidelines for a Strong Project Website**](#-guidelines-for-a-strong-project-website)
- [📊 **Minimum vs. Excellent Rubric**](#-minimum-vs-excellent-rubric)
- [Project Abstract](#project-abstract)
    - [Project Video](#project-video)
- [Project Motivation](#project-motivation)
- [System Block Diagram](#system-block-diagram)

---

# **1. Introduction**

- short paragraph about whats to come
- TinyODOM-EX in oneshot: what was built, what was measured and why. Probs write at the end

### **1.1 Motivation & Objective**  

- Need for robust inertial odometry when GPS is unavailable, especially for low power embedded systems that cannot offload compute
- TinyODOM showed that hardware aware NAS can produce deployable models for inertial odometry, but did not optimize for energy [cite here]
- Objective: design and evaluate an energy aware NAS pipeline for inertial odometry on microcontrollers, focusing on BLE33 and targeting real deployment constraints

### **1.2 State of the Art & Its Limitations**  
- Classical IMU fusion and odometry: Kalman style filters and hand designed models, sensitive to noise and bias and often tuned for desktops or phones
- Learning based inertial odometry: works like TinyODOM and similar deep inertial nav systems, accurate but usually tune architectures as black boxes and typically ignore power
- Black box and one for all NAS methods: general purpose optimizers like Mango and other tuners, often treat hardware as a coarse constraint and lack explicit energy objectives on microcontroller
*Include citations here for some of the above*

### **1.3 Novelty & Rationale**  
- Extend TinyODOM with an explicit energy objective 
  - add hardware in the loop measurement of current, voltage, and energy per inference on BLE33
  - optimize for accuracy, latency, memory, and energy instead of accuracy alone 
- Modernize and refactor the original TinyODOM codebase
  - replace the monolithic notebook with a modular TinyODOM EX stack: NAS client, HIL server, shared utilities, config files, firmware
  - move to TensorFlow 2 and TFLite Micro and Arduino CLI so that experiments are reproducible and easier to rerun on new machines
- Introduce a multi objective NAS setup based on Optuna
  - use Optuna pruning and multi objective capabilities to explore TCN architectures under strict memory and latency limits
  - design objective functions that incorporate measured BLE33 performance rather than only proxy metrics like FLOPs or parameter count
- Provide an end to end measurement pipeline on a real microcontroller
  - build a reliable BLE33 harness for automated flashing and telemetry
  - collect ground truth energy and latency measurements during NAS runs instead of only one off benchmarks

### **1.4 Potential Impact**  
- Provide a modular TinyODOM EX codebase that separates data prep, NAS logic, HIL control, and firmware, so other projects can reuse pieces without rewriting everything
- Enable repeatable hardware aware NAS experiments by treating the BLE33 measurement setup and scripts as a drop in module for future models or datasets
- Lower the barrier for adding new devices by isolating board specific code in the HIL server and firmware harness, so a future RP2040 or NPU port mostly touches one layer (note potential challenges depending on board flashing)
- Offer a concrete reference for structuring energy aware TinyML experiments, including how to couple Optuna, TFLite Micro, and Arduino CLI in a way that survives toolchain changes

### **1.5 Challenges**  
- RP2040 board instability under heavy flashing
  - repeated hard faults and failure to re enter BOOTSEL mode, blocking unattended NAS runs
  - A possible mitigation for *future* RP2040-based NAS runs is to attach a second microcontroller (e.g. a Pico running Picoprobe) configured to pull the target’s RUN pin low (i.e. reset) or to drive SWD reset, allowing automated recovery when the board fails to re-enter BOOTSEL after a hard fault [RUN-pin reset docs](https://forums.raspberrypi.com/viewtopic.php?t=340911&utm_source=chatgpt.com), [Pico-as-debug-probe workflow](https://raspberry-projects.com/pi/microcontrollers/programming-debugging-devices/debugging-using-another-pico?utm_source=chatgpt.com)
- Modernizing the entire software stack while keeping parity with TinyODOM
  - Python and TensorFlow upgrades, TFLite Micro integration, Arduino CLI setup, and new firmware harnesses that all have to agree on formats and interfaces
- Debugging across GPU and HIL processes
  - tracing failures that only show up on device during NAS runs, correlating ZeroMQ logs, serial output, and Optuna trial state

### **1.6 Metrics of Success**  
- Quantitative: accuracy metrics from OxIOD (RMSE, trajectory error), latency per inference on BLE33, memory footprint, and energy per inference
- System level: stable unattended HIL runs over many NAS trials without manual intervention or board recovery
- Scientific: clear visualization of tradeoffs between accuracy, latency, memory, and energy, plus comparison to TinyODOM style baselines

---

# **2. Related Work**

- Deep inertial odometry and navigation: TinyODOM, OxIOD dataset, other IMU only or phone based inertial odometry systems and their focus on accuracy over energy
- Hardware aware NAS and TinyML: works that co optimize models for microcontrollers using memory and latency constraints, but without explicit energy measurement
- Optimization methods and architectures: black box optimizers including Optuna, earlier tuners such as Mango, and core TCN papers that justify the chosen search space

---

# **3. Technical Approach**

- End to end NAS loop, sample TCN hyperparameters, train candidate models on GPU, export to TFLite Micro, deploy to BLE33, measure latency and energy, feed metrics back to Optuna
- Modular split between NAS client, HIL server, dataset prep, and firmware, use YAML config and shared utilities so experiments are repeatable and easy to rerun
- BLE33 as the primary device under test, document the RP2040 path as a negative result and as context for design choices in the final system

### **3.1 TinyODOM-EX System Architecture**
- Split pipeline into GPU side NAS client and embedded HIL server
  - NAS client samples hyperparameters, trains candidate TCNs, evaluates on validation data, and requests hardware measurements from the HIL server
  - ~8m combinations
- HIL server handles TFLite export, C plus plus generation, Arduino CLI compilation, uploading to BLE33, and collection of latency and energy metrics over serial
- Use ZeroMQ between NAS client and HIL server and store configuration and roles in YAML and shared utilities so runs are reproducible and can move between machines without code edits

### **3.2 Dataset and Windowing Pipeline**
- Use OxIOD dataset as the base inertial odometry corpus and choose train, validation, and test splits that are comparable to the original TinyODOM setup
- Preprocessing: resample to the chosen sampling rate, generate windows with fixed window size and stride, normalize sensor channels, and apply any simple augmentations or filtering
- Use limited budgeting and calibration passes: small fixed subset of windows for calibration, quantization, and warm up runs so that HIL costs stay manageable during NAS

### **3.3 NAS Objective, Search Space, and Training Procedure**
 
The latency budget in the scoring function is set to 200 ms, derived from the OxIOD streaming configuration rather than chosen arbitrarily. The IMU is sampled at 100 Hz, so each sample corresponds to 10 ms. Following TinyODOM, the NAS uses windows of 200 samples (2 s of history) with a stride of 20 samples between consecutive windows [1]. A new window therefore arrives every \(20 \times 10\ \text{ms} = 200\ \text{ms}\). To keep up with this stream in real time, each model must produce one prediction per window within at most 200 ms of compute latency. This bound is used as the latency budget \(L_b\) in the scoring function and in the latency penalty term for both energy aware and latency only scoring runs.

- Temporal convolutional network search space
  - depth, kernel sizes, dilation pattern, channel counts, residual connections, and heads that predict velocity components
- Objective design
  - construct a scalar Optuna objective from validation errors plus latency and memory, and extend it to include energy either as a penalty term or as part of a multi objective front
  - Note the addition of seeding with default configuration after score function, not energy aware study
- Training procedure
  - loss terms and optimization settings, number of epochs per trial, early stopping and Optuna pruning strategy, and how failed HIL calls or invalid models are marked and handled inside the NAS loop

### **3.4 Hardware in the Loop Measurement and Implementation**
- Hardware setup
  - Arduino Nano 33 BLE Sense and INA228 power monitor, physical wiring and USB power path, measurement setup used to obtain current, voltage, and energy per inference on BLE33
- Firmware harness
  - evolution from the original Mbed latency harness to Arduino sketches with consistent TFLite Micro setup and telemetry
  - separate energy instrumented and latency only firmware variants to balance measurement richness with tensor arena limits
- Software stack for HIL
  - Python 3.11, TensorFlow 2 and TFLite Micro, Arduino CLI tooling, ZeroMQ based RPC between NAS client and HIL server, and debugging tools used during BLE33 and RP2040 bring up

### **3.5 Key Design Decisions and Tradeoffs**
- Choosing Optuna over Mango and other tuners
  - support for multi objective search, pruning, and tight integration with Python based experiments
- Separating firmware into energy instrumented and latency only sketches
  - tradeoff between richer measurements and available tensor arena space on BLE33 and impact on maximum model size
- Dropping RP2040 from final experiments
  - operational reliability problems and the need for fully unattended NAS runs outweighed the benefit of a second DUT, so final results focus on BLE33 while still documenting the RP2040 failure mode and possible future mitigations
  - left time for multi-objective NAS runs

---

# **4. Evaluation & Results**

- Summarize the four experimental studies and common metrics
- Present baseline reproduction of TinyODOM on BLE33 as a reference point
- Evaluate the effect of adding energy logging in single-objective NAS
- Analyze multi-objective fronts for accuracy–latency and accuracy–energy
- Quantify NAS convergence and trial budget sufficiency

---

### **4.1 Experimental Studies and Metrics**

- Overview of the four studies
  - Study 1: Single-objective NAS, BLE33, score without energy logging
  - Study 2: Single-objective NAS, BLE33, score with energy logging
  - Study 3: Multi-objective NAS, accuracy vs latency
  - Study 4: Multi-objective NAS, accuracy vs energy per inference
- Common evaluation metrics
  - OxIOD splits and velocity / trajectory error metrics (RMSE, trajectory drift)
  - Latency per inference on BLE33, flash usage, RAM usage
  - Energy per inference from HIL measurement pipeline
- Pointer back to Section 3 for experimental setup
  - HIL infrastructure, number of trials, key hyperparameters

---

### **4.2 Reproducing TinyODOM on BLE33 (Study 1)**

#### **4.2.1 Baseline Model Quality**

- Compare TinyODOM-like model performance on BLE33 to original TinyODOM results [1]
- Figures
  - Trajectory overlays: model vs ground truth and bias-corrected IMU integration baseline
  - Velocity time-series and/or error-over-time plots for representative sequences
  - Predicted vs true velocity histograms to assess calibration
- Table of baseline metrics
  - RMSE per axis, trajectory error, summary across key OxIOD sequences
- Interpretation
  - Where the learned model improves over the classical baseline
  - Failure modes and typical error patterns

#### **4.2.2 Baseline Latency and Model Size Profile**

- Report typical latency and model size for the best TinyODOM-like model(s)
  - Table with: latency per inference, flash usage, RAM usage for 1–2 representative models
- Brief note on constraint handling
  - By design, the NAS loop prunes or discards trials that exceed BLE33 flash/RAM limits
  - As a result, all models discussed in this section satisfy deployment constraints by construction
- Interpretation
  - Position baseline TinyODOM-like models on the latency / size spectrum
  - Use this as a reference point for later energy-aware and multi-objective models


---

### **4.3 Single-Objective Energy-Aware NAS on BLE33 (Studies 1 and 2)**

The single-objective NAS runs use a scalar score that combines accuracy, memory, latency, and (optionally) energy per inference. For each trial, the validation velocity RMSE in x and y is converted into an accuracy term $\(A = -(R_x + R_y)\)$, a small resource term encodes relative RAM and flash usage, and latency and energy penalties are applied when the BLE33 hardware measurements exceed a 200 ms latency budget and an implicit 10 mJ energy target. The overall score is

$$
\[
\text{score} = A + 0.01 \, M - \text{lat\_pen} - \text{energy\_pen},
\]
$$

** NOTE TO SELF: MAYBE ALREADY DISCUSSED EARLIER AND IF SO, DELETE HERE AND JUST REFERENCE

where the energy penalty is active only in the energy-aware run. Higher scores correspond to lower validation error and fewer constraint violations. Scores in all but a handful of trials were negative.

#### **4.3.1 Effect of Energy Logging on NAS Outcome**

<figure style="text-align: left">
  <figcaption style="font-size: 0.9em; color: #555; margin-bottom: 4px;">
    <strong>Table X.</strong> Heuristic contributions of each term to the absolute scalar score for the single-objective NAS runs on BLE33 (75 trials each). The score combines accuracy, resource usage, latency, and optional energy penalties.
  </figcaption>
</figure>


| Term          | No-energy score (%) | Energy-aware score (%) |
|--------------|---------------------|------------------------|
| model_acc$   | 77.4                | 55.1                   |
| latency_term | 21.9                | 9.9                    |
| energy_term  | --                  | 34.7                   |
| resource_term| 0.7                 | 0.4                    |


Table X compares how the scalar score is distributed across components in the no-energy and energy-aware single-objective runs on BLE33. In the no-energy setting, the score is dominated by the accuracy term: $$model\_acc$$ accounts for about 77% of the absolute score, $$latency\_term$$ contributes roughly 22%, and the resource term is negligible. When energy measurements are enabled, the energy penalty becomes a first-class objective and takes about 35% of the score magnitude. The share of $$model\_acc$$ drops to roughly 55% and the latency contribution falls below 10%, while the resource term remains insignificant.

The mean $$model\_acc$$ value becomes slightly more negative in the energy-aware run, indicating a modest loss in validation accuracy as the optimizer trades off some performance for lower energy per inference. The mean latency penalty remains almost unchanged between the two runs, which is consistent with the BLE33 latency budget being satisfied by most trials in both cases. Overall, these statistics confirm that enabling the energy term rebalances the single-objective score toward energy per inference without turning it into a pure energy minimization problem: accuracy still carries the largest weight, but energy now plays a comparable role to latency in shaping the search.


- Contrast Study 1 (no energy logging) vs Study 2 (energy logging added to score)
- Figures
  - Stacked score contribution plots (accuracy, resource, latency, energy) with score overlay
  - Optuna optimization history: best score / RMSE vs trial index for both studies
- Interpretation
  - How adding energy logging changes which trials are favored
  - Evidence of convergence or plateauing in both cases
  - Discussion of any shift in accuracy–latency–resource tradeoff when energy is included

#### **4.3.2 Empirical Relationship Between Energy and Latency**

- Quantify how tightly energy and latency are coupled on BLE33
- Figures
  - Energy vs latency scatter plots, optionally with linear fit and correlation coefficient
  - HIL stability plot: latency and energy for repeated runs of the same model
- Interpretation
  - Measurement repeatability for both latency and energy
  - When latency is a good proxy for energy and where it breaks down
  - Outlier models that are faster but not proportionally more energy efficient

---

### **4.4 Multi-Objective NAS: Accuracy vs Latency (Study 3)**

#### **4.4.1 Pareto Front and Convergence**

<figure style="text-align: left">
  <img src="./assets/plots/MO_no_E_optuna_pareto_front.png"
       alt="Accuracy–latency Pareto front for multi-objective NAS without energy term"
       width="650" />
  <figcaption style="font-size: 0.9em; color: #555; margin-top: 4px;">
    <strong>Figure X.</strong> Accuracy–latency Pareto front for the multi-objective NAS run on the BLE33 without an explicit energy term. Blue points are individual trials, plotted by latency and aggregate  RMSE. The red curve marks the Pareto-optimal set. The vertical dashed line at 200ms indicates the real-time latency budget implied by the 100Hz sampling rate and a stride of 20 samples between windows.
  </figcaption>
</figure>

Figure X shows the tradeoff between aggregate RMSE over \(v_x\) and \(v_y\) and on-device latency in the accuracy–latency multi objective run. The cloud of blue points indicates that the search explored a wide range of models, from very fast but inaccurate configurations to slower and more accurate ones. The red Pareto curve traces the nondominated set (i.e. the best possible trade-off trials). Moving along this curve from left to right trades higher latency for lower error. The front drops steeply as latency increases from tens of milliseconds to roughly the 150–200 ms range, then flattens as latency approaches several hundred milliseconds.

The vertical 200 ms line marks the real-time budget implied by the streaming configuration (see Section [3.3](#33-nas-objective-search-space-and-training-procedure) for details). Several Pareto points lie to the left of this line with aggregate RMSE close to the global minimum, which shows that satisfying the real-time constraint is not restrictive for this dataset and search space. Slower models beyond the budget provide only modest additional accuracy gains compared to the best models that already meet the budget. This knee occurs prior to the real-time boundary suggesting that most of the benefit of the accuracy–latency tradeoff is doable.


- Evaluate tradeoff between accuracy (RMSE) and latency
- Figures
  - Accuracy–latency Pareto front (non-dominated points highlighted)
  - Hypervolume progression vs completed trials with reference point
  - Optuna parallel coordinates: key hyperparameters vs accuracy and latency
- Interpretation
  - Shape and “knee” of the accuracy–latency front
  - Whether the front is still moving at the end of the trial budget
  - Hyperparameter patterns along the Pareto front

### **4.4.2 Hyperparameter Sensitivity for Latency-Oriented Search**

<figure style="text-align: left">
  <img src="./assets/plots/MO_no_E_collated_hyperparam_plots_grid_Latency.png"
       alt="Multiple Objective hyperparamameters plots nb filters vs accuracy / latency"
       width="700" />
  <figcaption style="font-size: 0.9em; color: #555; margin-top: 4px;">
    <strong>Figure x.</strong> Hyperparameter sensitivity in the accuracy–latency multi-objective run on the BLE33. Top: nb_filters versus vector RMSE (left, with a logarithmic best-fit curve) and latency per inference (right, with a linear best-fit line). Bottom: kernel_size versus vector RMSE (left) and latency (right).
  </figcaption>
</figure>

Figure X shows how nb_filters and kernel_size correlate with accuracy and latency in the accuracy versus latency multi objective run. The top row indicates that nb_filters is a strong driver of both accuracy and latency. The nb_filters–RMSE panel includes a simple logarithmic best-fit curve, which highlights a diminishing returns pattern: error falls quickly as channel count increases from very small models, then tapers off once nb_filters reaches the mid range. The nb_filters–latency panel includes a linear best-fit line, which emphasizes the approximately linear growth of latency with channel count despite some scatter. Together, these two trends show that the most accurate models are also among the slowest, and that increasing nb_filters beyond the mid range mostly increases cost while providing only modest additional accuracy.
The bottom row shows that kernel_size is a much weaker knob. Good and bad models are spread across the kernel sizes explored, and there is no clear monotonic trend between kernel_size and either RMSE or latency. Some kernel sizes contain both low error and high error models, and latency varies widely within each kernel size. This is consistent with kernel_size behaving as a secondary design choice once the receptive field is sufficient, while nb_filters primarily controls both model quality and computational cost.

---

### **4.5 Multi-Objective NAS: Accuracy vs Energy (Study 4)**

#### **4.5.1 Pareto Front in Accuracy–Energy Space**

<figure style="text-align: left">
  <img src="./assets/plots/MO_EA_optuna_pareto_front.png"
       alt="Accuracy–energy Pareto front for multi-objective energy-aware NAS"
       width="650" />
  <figcaption style="font-size: 0.9em; color: #555; margin-top: 4px;">
    <strong>Figure X.</strong> Accuracy–energy Pareto front for the multi-objective NAS run with an explicit energy objective. Blue points show individual trials plotted by energy per inference (mJ) and aggregate RMSE. The red curve denotes the Pareto-optimal set. The target energy used in the scoring function, 10mJ, corresponds to running within the 200ms latency budget at a nominal power of 50mW.
  </figcaption>
</figure>


Figure X shows the tradeoff in the accuracy–energy space for the energy-aware multi objective run. As in the latency plot in section [4.4.1](#441-pareto-front-and-convergence), the blue points indicate the search covers a broad spectrum of designs, while the red curve again highlights the nondominated trials. Again, similar to the accuracy-latency plot, the Pareto front has a steep initial segment moving from the lowest energy models to more expensive trials initially yields large reductions in aggregate RMSE. Around a mid range energy level, near the 10mJ target implied by the 200ms latency budget and a 50mW nominal power draw, the curve indicates diminishing returns.

This shape suggests a natural operating regime for deployment. Very low energy models exist, but they incur substantial error. Increasing energy per inference up to the mid range buys most of the available accuracy improvement, while pushing to very high energy models produces only small additional gains. Together with the accuracy–latency Pareto in Figure X in section [4.4.1](#441-pareto-front-and-convergence), these curve show that the search space contains models that are both reasonably accurate, energy efficient and operate in a real-time setting.


- Evaluate tradeoff between accuracy (RMSE) and energy per inference
- Figures
  - Accuracy–energy Pareto front (non-dominated points)
  - Latency vs accuracy Pareto constructed by joining HIL logs and Optuna trials
  - Hypervolume progression for the accuracy–energy study
- Interpretation
  - Comparison of the accuracy–energy front to the accuracy–latency front
  - Whether optimizing for energy yields different architectures than optimizing for latency
  - Evidence of diminishing returns in hypervolume vs trials

### **4.5.2 Energy-Oriented Hyperparameter Structure**

<figure style="text-align: left">
  <img src="./assets/plots/MO_EA_collated_hyperparam_plots_nb_filters_only_Energy.png"
       alt="Multiple Objective Energy Aware hyperparameters plots nb filters vs accuracy / energy"
       width="700" />
  <figcaption style="font-size: 0.9em; color: #555; margin-top: 4px;">
    <strong>Figure x.</strong> Effect of nb_filters on accuracy and energy per inference in the accuracy–energy multi objective NAS run on the BLE33. Left: nb_filters versus vector RMSE with a logarithmic best-fit curve. Right: nb_filters versus energy per inference with a linear best-fit line..
  </figcaption>
</figure>

Figure X shows that for the energy aware multi objective study, nb_filters are still the dominant hyperparameter. The nb_filters–RMSE panel includes a logarithmic best-fit curve and shows that increasing channel count from very small models to the mid range generally reduces error, after which gains tend to taper off. The nb_filters–energy panel includes a linear best-fit line and illustrates that energy per inference increases approximately linearly with nb_filters, with moderate spread due to other architectural choices. Together, these plots indicate that channel count controls a smooth accuracy–energy tradeoff: larger nb_filters provide some accuracy improvements but at a predictable energy cost.
Kernel_size versus RMSE and energy was also examined (plots not shown to avoid redundancy) and, as in the accuracy–latency run, did not exhibit strong structure. This reinforces that kernel_size behaves as a secondary knob in the current search space, while nb_filters primarily determines both model quality and resource usage.
  
  ---

### **4.6 Cross-Study Comparison and NAS Trial Budget**

#### **4.6.1 Summary of Best Models Across Studies**

- Table summarizing selected models from each study
  - For each: RMSE, latency, energy, flash, RAM
- Comparison
  - Baseline TinyODOM-like model vs single-objective energy-aware vs multi-objective models
  - Tradeoff between simplicity of objective and quality of resulting models

#### **4.6.2 NAS Trial Budget and Practical Convergence**

- Use all studies to comment on how many trials are “enough”
- Figures
  - Optuna optimization history curves across studies
  - Hypervolume progression curves for Study 3 and Study 4 on a shared axis
- Interpretation
  - Points where additional trials give minimal improvement
  - Practical guidance: recommended trial counts for future runs on similar hardware
  - Discussion of remaining uncertainty and where more budget might still help

### **4.x Multi-Objective NAS**
- Pareto Frontiers for both latency and EA studies

### **4.x NAS Trial Budget**
- **Goal: choose NAS trial budgets from convergence, not arbitrary counts**
  - Question: how many NAS trials are actually needed for this search space and objectives?
  - Approach: look at convergence of (1) best objective value over trials (single-objective) and (2) hypervolume over trials (multi-objective).

- **Single-objective NAS, score without energy awareness**
  - Optimization history shows rapid improvement in the first ~10–20 trials.
  - Best objective value is already reached by roughly trial 20–25.
  - Relative improvement of best value from 25→50 and 50→75 trials is effectively 0 percent.
  - Interpretation: the search has converged by ~25 trials. The remaining trials only resample around an already discovered optimum.

- **Single-objective NAS, score with energy awareness**
  - Similar behavior: best value improves early and then stays flat.
  - Again, no measurable improvement in best value after about ~25 trials.
  - Conclusion: for this scoring function and search space, ~30–40 trials is a safe upper bound.

- **Effect of seeding with a default configuration**
  - Later single-objective runs start with a manually seeded “default” model (e.g., TinyODOM-like baseline).
  - In those runs the seeded trial is already as good as or better than everything the optimizer finds later.
  - This makes the best-value curve flat, and the plots show that NAS never beats the baseline even with many extra trials.
  - Interpretation: the default model is already near-optimal under the chosen score, and additional trials serve mainly to confirm that.

- **Multi-objective NAS with energy awareness (e.g., energy vs accuracy)**
  - Hypervolume increases sharply over the first ~10–15 trials.
  - By ~20–30 trials, hypervolume is within about 1 percent of its final value.
  - From ~30 to 110 trials, only tiny hypervolume gains are observed.
  - Interpretation: the Pareto front is essentially discovered by ~20–30 trials; further trials slightly refine but do not change the frontier.

- **Multi-objective NAS without energy awareness (e.g., energy vs latency or latency vs accuracy)**
  - Same qualitative pattern: rapid hypervolume growth at the beginning, then an early plateau.
  - Hypervolume after ~15–20 trials is very close to the final value at 100+ trials.
  - Conclusion: 40–60 trials would have been more than sufficient for this multi-objective run.

- **Overall conclusions about trial budgets**
  - For **single-objective** studies in this project, convergence occurs by ~25 trials. A budget of **30–40 trials** is a conservative choice.
  - For **multi-objective** studies, the Pareto hypervolume converges by ~20–30 trials. A budget of **40–60 trials** is a conservative choice.
  - The larger budgets used in the final experiments (75 and 135 trials) are therefore more than sufficient and mostly spend time in the diminishing-returns regime.

- **Potential stopping criteria for future runs**
  - Single-objective: stop if the best score has not improved for the last 15–20 trials.
  - Multi-objective: stop if hypervolume has improved by less than ~1 percent over the last 20 trials.
  - These criteria formalize the convergence behavior observed in the plots.


---
# **5. Discussion & Conclusions**

### **5.1 Summary of Key Findings**

- Brief recap of TinyODOM-EX contributions
  - Reproduction of TinyODOM-style accuracy on BLE33 within a fully automated HIL NAS loop.
  - Successful incorporation of energy logging into both single-objective and multi-objective NAS.
  - Observation of clear accuracy–latency–energy tradeoffs and identification of nb_filters as the dominant hyperparameter.
- High-level statement of what the results collectively show about energy-aware NAS on microcontrollers.

---

### **5.2 Lessons from Energy-Aware NAS and HIL Infrastructure**

- Objectives and optimization behavior
  - Single-objective composite scoring is simple to use but hides the shape of the tradeoff surface.
  - Multi-objective NAS (accuracy–latency and accuracy–energy) exposes Pareto fronts and hypervolume progression, enabling a more explicit view of tradeoffs.
  - Direct optimization for energy per inference can shift the Pareto front relative to optimization for latency alone.

- Search space behavior
  - nb_filters acts as the main capacity and cost knob:
    - Very small models have high RMSE.
    - Mid-range nb_filters provide a good accuracy–cost compromise.
    - Larger nb_filters show diminishing accuracy gains while increasing latency and energy.
  - kernel_size has comparatively weak structure within the explored range and behaves as a secondary design choice once the receptive field is sufficient.
  - PCA projections of hyperparameters do not exhibit strong clustering, suggesting that useful structure resides in higher-dimensional combinations captured better by per-parameter plots and importance scores.

- Measurement and latency–energy relationship
  - Latency and energy per inference are strongly correlated on BLE33 but not in a perfectly one-to-one fashion; outliers motivate direct energy measurement.
  - HIL stability experiments show that repeated runs of the same model have low variance in both latency and energy, supporting the reliability of measurements.

- HIL infrastructure and trial budget
  - GPU-side NAS client and HIL server split works effectively with Optuna as the search engine.
  - Enforcing flash/RAM feasibility inside the HIL loop ensures that all reported models are deployable on BLE33 by construction.
  - Optimization histories and hypervolume curves indicate approximate saturation points, providing empirical guidance on how many trials are needed before returns diminish.

---

### **5.3 Limitations and Threats to Validity**

- Hardware scope
  - Final quantitative results are obtained on a single device under test (BLE33).
  - RP2040 exploration is limited by board instability; conclusions about energy/latency structure are specific to BLE33.

- Dataset scope
  - All evaluations use the OxIOD dataset with a particular train/validation/test split.
  - Cross-dataset generalization and performance in other environments are not measured.

- Search space and training budget
  - NAS is restricted to a TCN-based search space with specific hyperparameters; other architectures (for example, RNNs, transformers, pruned/quantized variants) are not explored.
  - Each trial uses a fixed training budget; the impact of more aggressive training or early stopping strategies is not fully characterized.

- Measurement assumptions
  - Energy measurements are taken at the board level under a specific harness, supply path, and workload.
  - Different clocking schemes, peripheral usage, or deployment scenarios may shift absolute latency and energy values.

---

### **5.4 Future Work**

- Broaden hardware and accelerators
  - Extend TinyODOM-EX to additional microcontrollers and boards with integrated NPUs or accelerators.
  - Use these platforms to test whether the approximately linear latency–energy relationship observed on BLE33 persists on more optimized hardware or whether accelerators change the shape of the tradeoff.

- Extend objectives and model space
  - Incorporate richer multi-objective formulations (for example, three-way accuracy–latency–energy fronts, or energy-normalized error).
  - Expand the search space to include alternative temporal architectures and explicit quantization or pruning strategies.

- Improve tooling and portability
  - Evolve TinyODOM-EX into a reusable “tool” that can be pointed at different projects with minimal changes.
  - Support multiple compiler and uploader backends (for example, Arduino CLI, PlatformIO, vendor-specific SDKs) behind a common HIL interface so that new boards can be added by implementing a single backend.
  - Provide configuration templates and scripts that make it easy to plug in new datasets, search spaces, and devices.

- Toward real deployments
  - Integrate NAS-selected models into end-to-end systems (such as small robots or wearables) to evaluate navigation performance beyond offline OxIOD metrics.
  - Revisit RP2040-class boards with an external debug/reset controller or similar recovery mechanism to enable reliable multi-device NAS.

---

### **5.5 Final Conclusion**

- Single paragraph that:
  - Restates what TinyODOM-EX is and what problem it addresses.
  - Highlights the main technical takeaway about energy-aware NAS for inertial odometry on microcontrollers.
  - Emphasizes the intended impact of the modular codebase and HIL pipeline as a reference for future energy-aware TinyML experiments.

---

# **6. References**

*Provide full citations for all sources (academic papers, websites, etc.) referenced and all software and datasets uses.*

---

# **7. Supplementary Material**

### **7.1. Datasets**

- OxIOD: source, URL, sensor modalities, collection settings, and which subsets used
- Data format: raw IMU streams, trajectories, and any intermediate outputs the pipeline writes (windowed tensors, cached datasets)
- Preprocessing steps: extraction, normalization, window generation, split restoration, and how prepare_oxiod.py makes this reproducible for other users
  
### **7.2. Software**

- External libraries: TensorFlow and TFLite Micro versions, Optuna, ZeroMQ, Arduino CLI, and any plotting or logging libraries
  - include talk of the using the shell scripts and having everything internal to both the conda environment and the folder
- Internal modules: NAS client, HIL server, shared utilities, config files, firmware sketches, and dataset prep scripts, with short one line roles for each

---

> [!NOTE] 
> Read and then delete the material from this line onwards.

# 🧭 **Guidelines for a Strong Project Website**

- Include multiple clear, labeled figures in every major section.  
- Keep the writing accessible; explain acronyms and algorithms.  
- Use structured subsections for clarity.  
- Link to code or datasets whenever possible.  
- Ensure reproducibility by describing parameters, versions, and preprocessing.  
- Maintain visual consistency across the site.

---

# 📊 **Minimum vs. Excellent Rubric**

| **Component**        | **Minimum (B/C-level)**                                         | **Excellent (A-level)**                                                                 |
|----------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Introduction**     | Vague motivation; little structure                             | Clear motivation; structured subsections; strong narrative                                |
| **Related Work**     | 1–2 citations; shallow summary                                 | 5–12 citations; synthesized comparison; clear gap identification                          |
| **Technical Approach** | Text-only; unclear pipeline                                  | Architecture diagram, visuals, pseudocode, design rationale                               |
| **Evaluation**       | Small or unclear results; few figures                          | Multiple well-labeled plots, baselines, ablations, and analysis                           |
| **Discussion**       | Repeats results; little insight                                | Insightful synthesis; limitations; future directions                                      |
| **Figures**          | Few or low-quality visuals                                     | High-quality diagrams, plots, qualitative examples, consistent style                      |
| **Website Presentation** | Minimal formatting; rough writing                           | Clean layout, good formatting, polished writing, hyperlinks, readable organization        |
| **Reproducibility**  | Missing dataset/software details                               | Clear dataset description, preprocessing, parameters, software environment, instructions   |


<div style="text-align: center">
  <img src="./assets/img/UCLA_Samueli_ECE_block-1.png" alt="logo" width="300" />
</div>

# Project Abstract
<!-- The project abstract should be a short (< 200 words) summary of what your project does -->

<div style="text-align: left">
  <img src="./assets/img/Logo.png" alt="logo" width="100" />
</div>

### Project Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/y5Qfcjh6fBQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# Project Motivation
<!-- In the project motivation explain the background behind why you chose this project. -->

# System Block Diagram
