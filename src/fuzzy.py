import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')

risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.4])
confidence['medium'] = fuzz.trimf(confidence.universe, [0.2, 0.5, 0.8])
confidence['high'] = fuzz.trimf(confidence.universe, [0.6, 1, 1])

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

rule1 = ctrl.Rule(confidence['low'], risk['low'])
rule2 = ctrl.Rule(confidence['medium'], risk['medium'])
rule3 = ctrl.Rule(confidence['high'], risk['high'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

def get_risk_level(conf_value):
    risk_sim.input['confidence'] = conf_value
    risk_sim.compute()

    risk_value = risk_sim.output['risk']

    if risk_value < 30:
        return "LOW"
    elif risk_value < 75:
        return "MEDIUM"
    else:
        return "HIGH"
    