import random

class RiskAssessor:
    def __init__(self):
        # Dummy models for risk prediction
        pass

    def predict_off_target_risk(self, sequence, technology):
        """
        Simulates a CNN model predicting off-target probability.
        """
        base_risk = {"CRISPR": 0.8, "Base Editing": 0.4, "Prime Editing": 0.2, "RNA Editing": 0.1}
        risk_score = base_risk.get(technology, 0.5) * random.uniform(0.8, 1.2)
        return min(max(risk_score, 0.0), 1.0)

    def predict_functional_disruption(self, locus):
        """
        Simulates predicting the probability of disrupting essential gene functions.
        """
        # Dummy logic based on locus name length
        return min(len(locus) / 20.0, 0.9)

    def assess_risk(self, sequence, locus, technology):
        off_target = self.predict_off_target_risk(sequence, technology)
        func_disruption = self.predict_functional_disruption(locus)
        
        # Calculate an overall risk score
        overall_risk = (off_target * 0.6) + (func_disruption * 0.4)
        
        risk_level = "Low"
        if overall_risk > 0.7:
            risk_level = "High"
        elif overall_risk > 0.4:
            risk_level = "Medium"
            
        return {
            "technology": technology,
            "off_target_probability": round(off_target, 3),
            "functional_disruption_probability": round(func_disruption, 3),
            "overall_risk_score": round(overall_risk, 3),
            "risk_level": risk_level,
            "uncertainty_interval": f"±{round(random.uniform(0.05, 0.15), 2)}"
        }

if __name__ == "__main__":
    assessor = RiskAssessor()
    report = assessor.assess_risk("ATGCGTACGTAGCTAG", "BRCA1_exon11", "CRISPR")
    print("Risk Assessment Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
