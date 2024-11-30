from flask import Flask, request
from twilio.rest import Client
import os

app = Flask(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = "USE YOUR TWILIO SID"  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = "GET FROM TWILIO DEV"    # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "GET FROM TWILIO DEV"      # Replace with your Twilio phone number
AUTHORITY_PHONE_NUMBER = "USE NUMBER ON TWILIO ACCOUNT FOR TRIAL OR ON PREMIUM USE ANY"   # Replace with the authority's phone number

# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/send_alert', methods=['POST'])
def send_alert():
    # Get the data from the request
    camera_name = request.form.get('camera_name')
    people_count = request.form.get('people_count')
    crowd_density = request.form.get('crowd_density')

    # Prepare the SMS content
    message_content = f"Amber Alert from {camera_name}: High crowd density detected! Current density: {crowd_density} with {people_count} people. Immediate attention required."

    # Send SMS using Twilio
    try:
        message = client.messages.create(
            body=message_content,
            from_=TWILIO_PHONE_NUMBER,
            to=AUTHORITY_PHONE_NUMBER
        )
        return f"Alert sent successfully! Message SID: {message.sid}", 200
    except Exception as e:
        return f"Failed to send SMS: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
