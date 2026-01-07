# Predictive Healthcare Capacity Management  
**The HealSight Initiative – MedOptix Analytics**

## Project Overview
This project focuses on predictive healthcare capacity management to help hospitals move from reactive operations to proactive planning. The solution forecasts patient inflows and supports better bed allocation and staffing decisions using historical hospital data and predictive analytics.

The work is inspired by MedOptix Analytics’ vision of building data-powered hospitals that anticipate demand rather than react to it, with a focus on public healthcare systems in Nordic countries.

---

## Business Problem
Hospitals operating under reactive capacity management face persistent operational inefficiencies, particularly during peak demand periods. Key challenges include:

- Prolonged patient waiting times  
- Inefficient bed utilization  
- Emergency department overcrowding during surges  
- Unpredictable staff scheduling and high overtime  
- Limited visibility into real-time patient flow  

These issues negatively impact patient experience, staff wellbeing, and operational costs.

---

## Project Objective
Use predictive modelling to forecast patient admissions ahead of time, enabling data-driven decisions around:

- Bed and ward allocation  
- Emergency overflow prevention  
- Proactive staffing and scheduling  
- Hospital capacity planning  

The primary aim is to support hospital administrators with forward-looking insights rather than retrospective reporting.

---

## Data Used
The solution integrates multiple hospital data sources, including:

- **Admissions data**
  - Historical patient inflow records  

- **Capacity data**
  - Bed availability and utilization  

- **Operational data**
  - Staffing levels and overtime  

- **Contextual factors**
  - Seasonal patterns and demand surges  

Data was cleaned, structured, and prepared using SQL and Python prior to modelling.

---

## Modelling Approach
Predictive models were developed to forecast patient admissions over short- to medium-term horizons (7, 14, and 30 days ahead).

Key characteristics:
- Time-aware forecasting of patient inflows  
- Emphasis on interpretability and operational relevance  
- Models tracked and versioned using MLflow  

The modelling approach is designed to support planning decisions rather than act as an automated control system.

---

## Deployment & Visualization
The solution includes a lightweight dashboard layer to make predictions accessible to non-technical users.

- **Backend**: Python-based predictive models  
- **Model tracking**: MLflow  
- **Packaging & deployment**: Docker  
- **Visualization**: Power BI dashboards  

Dashboards present forecast trends, bed utilization indicators, and staffing KPIs to support operational decision-making.

---

## Results & Impact
The project reports measurable operational improvements following deployment and validation:

- **88% bed utilization efficiency**, increased from 68%  
- **Emergency overflow incidents reduced** from 32 to 11 per month  
- **Staff overtime costs reduced** from €125,000 to €90,000  
- **88% prediction accuracy** achieved for patient inflow forecasts  

These results demonstrate the potential of predictive analytics to improve hospital efficiency, staff wellbeing, and patient experience when integrated into operational workflows.

---

## Technology Stack
- **Python** (Pandas, Scikit-learn)  
- **SQL** (data querying and integration)  
- **MLflow** (experiment tracking and model registry)  
- **Docker** (deployment and portability)  
- **Power BI** (visual analytics and dashboards)  

---

## Notes
This repository focuses on the analytics and modelling components of the solution. Further impact assessment would depend on continued live deployment and long-term operational monitoring.
