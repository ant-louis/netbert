import time
import smtplib
import argparse


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pwd", type=str,
                        help="Password to gmail account.")
    arguments, _ = parser.parse_known_args()
    return arguments


def create_email(output, pwd):
    """
    """
    # Connect to server
    print("Connecting to the server...")
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login('training.aimodel@gmail.com', pwd)
    print("Connected!")
    
    # Content of email
    sender = 'training.aimodel@gmail.com'
    receivers = ['ant.louis@protonmail.com']
    subject = 'Process terminates.'
    message = 'Process terminates with message:\n' + output + '\n\n Best\n Antoine'
    email = """\
    From: %s
    To: %s
    Subject: %s

    %s
    """ % (sender, ", ".join(receivers), subject, message)
    
    # Send email
    print("Sending email...")
    server.sendmail(sender, receivers, email) 
    server.close()
    print("Sent!")
    

def main(args):
    """
    """
    create_email('Test', args.pwd)



if __name__=="__main__":
    args = parse_arguments()
    main(args)
