⚙️ ApexSentry — Hybrid Edge-to-Cloud Equipment Triage

An intelligent industrial diagnostics and telemetry triage system built for the AMD Developer Hackathon: ACT III (Track 1: Intelligent Industry).

ApexSentry couples high-speed, local computer vision at the machine edge with cloud-accelerated reasoning models running on AMD ROCm / AMD Radeon Cloud infrastructure.

📌 Problem Statement

In heavy manufacturing and automation, unplanned machinery downtime costs an average of $260,000 per hour. Plant operators face two critical bottlenecks:

Isolated Defect Flagging: Conventional optical inspection draws bounding boxes around surface cracks but cannot interpret root causes or prescribe immediate corrective actions.

Network Saturation: Streaming continuous, uncompressed 4K video feeds from dozens of factory-floor cameras to cloud servers incurs extreme network costs and high latency.

💡 Solution Architecture

ApexSentry resolves this tradeoff via a dual-tier Edge-to-Cloud pipeline:

[Local Camera / Telemetry Sensors]
                 │
                 ▼
   ┌─────────────────────────────┐
   │     Tier 1: Edge Vision     │ ◄── Local OpenCV Pipeline (~11 ms)
   │  - Micro-crack boundary box │
   │  - Optical noise filtering  │
   │  - Defect surface area (%)  │
   └─────────────┬───────────────┘
                 │
                 ▼ (Sends compact JSON metadata; 99% bandwidth reduction)
   ┌─────────────────────────────┐
   │    Tier 2: AMD Cloud LLM    │ ◄── AMD Token Factory / ROCm GPU Cloud
   │  - Physical failure triage  │     (DeepSeek-V4-Flash / Gemma-4-31B-IT)
   │  - Root-cause hypothesis    │
   │  - Immediate safe shutdown  │
   └─────────────┬───────────────┘
                 │
                 ▼
 [Structured Engineering Dispatch & Safety LOTO Action]


Edge Tier (Local Processing): Fast OpenCV contour routines inspect component surfaces (turbine blades, hydraulic pistons, conveyor rollers), calculate fault density, and isolate defect regions in real time without sending raw video across external networks.

Cloud Tier (AMD ROCm Acceleration): Compact anomaly telemetry is transmitted to the AMD Token Factory endpoint (developer.amd.com.cn/radeon/api/v1), invoking open models accelerated on AMD GPU hardware to synthesize immediate emergency containment protocols.

🚀 Key Features

Sub-15ms Edge Vision Isolation: Local OpenCV algorithms extract defect boundaries and calculate surface degradation percentages in real time.

Direct AMD Cloud Acceleration: Calls live AMD Token Factory endpoints running open-weight models on dedicated AMD GPU infrastructure.

Standardized Operator Protocols: Synthesizes severity ratings (Critical, High, Advisory), failure mechanics, and OSHA/LOTO safety steps.

Multimodal Telemetry Plotting: Correlates time-series sensor streams (vibration RMS, line pressure, operating temperature) alongside visual fracture maps.

🛠️ Tech Stack

Front-End & UI: Streamlit

Edge Computer Vision: OpenCV (cv2), NumPy, Pillow

Telemetry & Data Handling: Pandas

Cloud Infrastructure & Acceleration: AMD Token Factory, AMD Radeon Cloud / ROCm vLLM

Models Tested: DeepSeek-V4-Flash / Gemma-4-31B-IT (Google Partner Track)
