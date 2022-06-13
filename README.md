# TMB_JM
# A joint model considering measurement errors for optimally identifying tumor mutation burden threshold

It is a multi-endpoint joint model as a generalized method for inferring the TMB thresholds, facilitating consistent statistical inference using an iterative numerical estimation procedure considering misspecified covariates. The model considered the discrete tumor response endpoint in addition to the continuous time-to-event (TTE) endpoint simultaneously to optimize the division from a comprehensive perspective.

## Usage:
**Input**:  
* Patients' observations (containing ORR and TTE endpoints, as well as other clinical indicators) and the corresponding TMB values.  

**Output**: 
* The optimal TMB thresholds for the dichotomy.  

**NOTE1:**
The calculation criteria for the composite prognosis indexes can be adjusted according to the clinical characteristics of the cancer species to be analyzed.
