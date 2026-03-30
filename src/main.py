from model import model, X_test
from fuzzy import get_risk_level

predictions = model.predict(X_test.toarray())

for i in range(10):
    conf = predictions[i][0]
    risk = get_risk_level(conf)

    print("Confidence: ", round(conf, 2))
    print("Risk Level: ", risk)
    print("------------------")