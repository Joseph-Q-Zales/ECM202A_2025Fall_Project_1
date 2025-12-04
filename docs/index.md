---
layout: page
title: TinyODOM-EX(tension)
permalink: /
exclude: true
---
<sub>*(Optional: Replace with a conceptual figure or meaningful image.)*</sub>
<div style="text-align: center">
  <img src="./assets/img/UCLA_Samueli_ECE_block-1.png" alt="logo" width="500" />
</div>


Joseph Zales (jzales (at) ucla.edu, [GitHub](https://github.com/Joseph-Q-Zales))  

<sub>*(Optional: Replace with a conceptual figure or meaningful image.)*</sub>
<div style="text-align: center">
  <img src="./assets/img/TinyODOM-EX logo.png" alt="logo" width="500" />
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
- [**5. Discussion \& Conclusions**](#5-discussion--conclusions)
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
- Temporal convolutional network search space
  - depth, kernel sizes, dilation pattern, channel counts, residual connections, and heads that predict velocity components
- Objective design
  - construct a scalar Optuna objective from validation errors plus latency and memory, and extend it to include energy either as a penalty term or as part of a multi objective front
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

*Present experimental results with clarity and professionalism.

Include:

- Plots (accuracy, latency, energy, error curves)  
- Tables (comparisons with baselines)  
- Qualitative visualizations (spectrograms, heatmaps, bounding boxes, screenshots)  
- Ablation studies  
- Error analysis / failure cases

Each figure should have a caption and a short interpretation.*

---

# **5. Discussion & Conclusions**

*Synthesize the main insights from your work.

- What worked well and why?  
- What didn’t work and why?  
- What limitations remain?  
- What would you explore next if you had more time?  

This should synthesize—not merely repeat—your results.
*
---

# **6. References**

*Provide full citations for all sources (academic papers, websites, etc.) referenced and all software and datasets uses.*

---

# **7. Supplementary Material**

### **7.1. Datasets**

- Describe OxIOD: source, URL, sensor modalities, collection settings, and which subsets you used
- Data format: raw IMU streams, trajectories, and any intermediate outputs your pipeline writes (windowed tensors, cached datasets)
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
