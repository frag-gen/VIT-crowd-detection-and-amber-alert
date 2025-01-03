<<<<<<< HEAD
from flask import Flask, request, jsonify
from twilio.rest import Client
import os

app = Flask(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = "ACbeeb50d1489a76611aabe973f8c84689"  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = "7f9485d34b030053d190b19e4597d4ec"    # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "+17756373600"      # Replace with your Twilio phone number
AUTHORITY_PHONE_NUMBER = "+916392104804"   # Replace with the authority's phone number

# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/send_alert', methods=['POST'])
def send_alert():
    try:
        # Get the data from the request
        data = request.get_json()  # Get JSON data from the request body
        camera_name = data.get('cameraName', 'Unknown Camera')
        people_count = data.get('peopleCount', 0)
        crowd_density = data.get('status', 'unknown')

        # Prepare the SMS content
        message_content = (
            f"Amber Alert from {camera_name}: High crowd density detected! "
            f"Current density: {crowd_density} with {people_count} people. Immediate attention required."
        )

        # Send SMS using Twilio
        message = client.messages.create(
            body=message_content,
            from_=TWILIO_PHONE_NUMBER,
            to=AUTHORITY_PHONE_NUMBER
        )

        return jsonify({'message': 'Alert sent successfully', 'sid': message.sid}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
=======
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
>>>>>>> 57eb7f1fa34fa39deb7752ce475d84cc8b7dafe6
