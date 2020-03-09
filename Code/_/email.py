from knockknock import email_sender

@email_sender(recipient_emails=["ant.louis@protonmail.com"], sender_email="training.aimodel@gmail.com")
def train_your_nicest_model(your_nicest_parameters):
    import time
    time.sleep(10)
    return {'loss': 0.9} # Optional return value