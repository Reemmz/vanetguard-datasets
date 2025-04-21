# VANETGuard Datasets

This dataset supports the **VANETGuard** project — an advanced trust management system for Vehicular Ad-Hoc Networks (VANETs).  
It detects malicious vehicles using entropy-based anomaly detection, Bayesian reputation updates, and trust propagation.

---

## 📁 Dataset Structure

This release includes 24 scenario folders combining:
- **Vehicle Counts**: 20, 40, 80, 160, 320, 400
- **Malicious Percentages**: 5%, 10%, 20%, 40%

Each folder contains:
- `datasetDen.csv`: ETSI-compliant DENM messages (with simulated malicious behavior)
- `datasetCam.csv`: CAM messages from all vehicles
- `initial_reputations.csv`: Initial trust values
- `malicious_sources.txt`: List of misbehaving vehicle IDs

---

## 🔗 Download

📦 **[Download the Full Dataset (172 MB)](https://github.com/Reemmz/vanetguard-datasets/releases/latest/download/generated_datasets.zip)**

---

## 🧠 Use Case

These datasets are designed for:
- Trust/reputation model evaluation
- VANET anomaly and intrusion detection
- Simulating VANETGuard without running full OMNeT++ simulations

Compatible with:
- `v2v.py` or any trust engine
- Python-based analytics and detection pipelines

---

## 📄 License

This dataset is licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**.  
You may use, modify, and share the data with appropriate credit.

🔗 [View License](https://creativecommons.org/licenses/by/4.0/)

---

## 📚 Citation

If you use this dataset, please cite:

> Reemmz, _VANETGuard: Trust-Based Detection System for VANETs_, 2025.  
> [GitHub Repo](https://github.com/Reemmz/vanetguard-datasets)

---

