import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import traceback

load_dotenv()

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASS = os.getenv("SMTP_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")

app = FastAPI(title="MLOps FastAPI Prediction Service")

model = joblib.load("model/model.pkl")


def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = TO_EMAIL

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASS)
        server.sendmail(SMTP_EMAIL, TO_EMAIL, msg.as_string())
        server.quit()
        print("✅ Email sent successfully")
    except Exception as e:
        print("❌ Email failed:", e)


class InputData(BaseModel):
    # Update features based on dataset
    feature1: float
    feature2: float


@app.post("/predict")
def predict(data: InputData):
    try:
        features = [[data.feature1, data.feature2]]
        prediction = model.predict(features)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        err = traceback.format_exc()

        # log error
        with open("error.log", "a") as f:
            f.write(err + "\n")

        send_email("❌ ALERT: API Prediction Failed", err)

        raise HTTPException(status_code=400, detail="Prediction failed")
