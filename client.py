import requests
import json
import time
APIUrl = "http://localhost:8000"
def testPredictSingle():
  print("\nTesting High Risk Student (Struggling - Very Low Engagement)")
  payload = {
    "studentId": "stuRisk999",
    "VLELogs": [
      {"date": 1, "sumClick": 2, "activityType": "homepage"},
      {"date": 10, "sumClick": 1, "activityType": "resource"},
      {"date": 20, "sumClick": 1, "activityType": "resource"},
      {"date": 30, "sumClick": 1, "activityType": "forumng"},
      {"date": 40, "sumClick": 0, "activityType": "quiz"}
    ],
    "assessments": [{"score": 30.0, "dateSubmitted": 25, "dueDate": 20}],
    "studiedCredits": 120
  }
  resp = requests.post(f"{APIUrl}/predict", json=payload)
  if resp.status_code == 200:
    data = resp.json()
    print(f"Risk: {data['riskLabel']} | Prob: {data['stressProb']:.4f} | Conf: {data['confidence']}")
  else:
    print(f"Error: {resp.text}")
def testPredictNormal():
  print("\nTesting Normal Student (High Engagement - Consistent Activity)")
  VLELogs = []
  for d in range(1, 200, 10):
    VLELogs.append({"date": d, "sumClick": 45, "activityType": "homepage"})
    VLELogs.append({"date": d+2, "sumClick": 30, "activityType": "resource"})
    VLELogs.append({"date": d+5, "sumClick": 60, "activityType": "quiz"})
  payload = {
    "studentId": "stuNormal111",
    "VLELogs": VLELogs,
    "assessments": [
      {"score": 85.0, "dateSubmitted": 20, "dueDate": 25},
      {"score": 90.0, "dateSubmitted": 50, "dueDate": 55},
      {"score": 88.0, "dateSubmitted": 180, "dueDate": 190}
    ],
    "gender": 0,
    "studiedCredits": 60
  }
  resp = requests.post(f"{APIUrl}/predict", json=payload)
  if resp.status_code == 200:
    data = resp.json()
    print(f"Risk: {data['riskLabel']} | Prob: {data['stressProb']:.4f} | Conf: {data['confidence']}")
  else:
    print(f"Error: {resp.text}")
if __name__ == "__main__":
  print("=== Final API Verification ===")
  testPredictSingle()
  testPredictNormal()